"""
Gymnasium-compatible portfolio allocation environment.

The agent treats portfolio management as a sequential decision problem:
at each timestep it observes the current market state and outputs target
weights for each asset. The reward is the risk-adjusted return minus
transaction costs.

Design choices:
  State space: per-stock feature vector (25 indicators) concatenated with
    the current portfolio weights. Including weights in the observation gives
    the agent memory of its current positions without explicit recurrence.

  Action space: continuous weights ∈ [0, 1] summing to 1, with a cash slot
    as the last element. Cash earns zero return but avoids forced investment
    when the agent has no conviction.

  Reward: Sharpe-scaled daily return minus transaction costs.
    Using raw return as the reward trains the agent to maximise total return
    regardless of risk; scaling by a rolling volatility estimate encourages
    the agent to seek returns per unit of risk (Sharpe ratio objective).
    Transaction cost (0.1%) penalises unnecessary rebalancing.

  Episode length: full history of the shortest ticker in the universe.
    Each episode starts at a random point to prevent memorisation.

References:
  Liu et al. (2020) FinRL — original RL portfolio framework
  Mnih et al. (2015) DQN — inspiration for observation design
"""
import logging
import numpy as np

logger = logging.getLogger(__name__)

# Features used by the RL agent — must be a subset of feature_engineer.py output.
# Chosen to cover trend, momentum, volatility, and volume signals without
# including absolute price levels that don't transfer across stocks.
FEATURE_COLS = [
    # trend
    'sma20', 'sma50', 'ema12', 'ema26', 'golden_cross', 'sma20_slope',
    # macd
    'macd', 'macd_signal', 'macd_hist',
    # momentum
    'rsi', 'stoch_k', 'williams_r', 'momentum_10d', 'momentum_21d',
    # volatility / range
    'volatility', 'vol_ratio', 'bb_pct', 'atr_pct',
    # volume
    'obv_momentum', 'volume_ratio',
    # returns and position
    'daily_return', 'weekly_return', 'close_to_sma20', 'week52_position',
    # cycle/trend strength
    'adx',
    # sentiment (from FinBERT fusion step)
    'sent_score', 'sent_news_count',
]

N_FEATURES = len(FEATURE_COLS)   # 27 features — update if FEATURE_COLS changes


class PortfolioEnv:

    def __init__(self, master_data: dict, tickers: list,
                 initial_balance: float = 10_000.0, tx_cost: float = 0.001):
        try:
            from gymnasium import spaces
        except ImportError:
            raise ImportError('Run: pip install gymnasium stable-baselines3')

        from gymnasium import spaces

        # only keep tickers that have enough history to train on
        self.tickers  = [t for t in tickers if t in master_data and len(master_data[t]) > 60]
        self.n_stocks = len(self.tickers)

        if self.n_stocks == 0:
            raise ValueError('No tickers with sufficient history — run the data pipeline first')

        self.master_data     = master_data
        self.initial_balance = initial_balance
        self.tx_cost         = tx_cost

        # observation: N_FEATURES per stock + current weight per asset (stocks + cash)
        obs_dim = self.n_stocks * N_FEATURES + (self.n_stocks + 1)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # action: target weight per asset in [0, 1]; agent normalises to sum to 1
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(self.n_stocks + 1,), dtype=np.float32)

        logger.info(f'PortfolioEnv: {self.n_stocks} stocks · obs={obs_dim} · act={self.n_stocks+1}')
        self._preprocess()

    # ── Gymnasium API ──────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)

        # randomise start point so the agent doesn't memorise a single trajectory
        max_start = max(0, self.episode_len - 252 - 1)
        self.t    = np.random.randint(30, max(31, max_start)) if max_start > 30 else 30

        # start fully in cash
        self.weights         = np.zeros(self.n_stocks + 1, dtype=np.float32)
        self.weights[-1]     = 1.0
        self.portfolio_value = self.initial_balance

        return self._obs(), {}

    def step(self, action: np.ndarray):
        # normalise: raw network outputs are in [0,1] but don't sum to 1
        action = np.abs(action).astype(np.float32)
        action /= (action.sum() + 1e-8)

        stock_ret = self._returns_at(self.t)
        all_ret   = np.append(stock_ret, 0.0)   # cash slot earns zero return
        port_ret  = float(np.dot(action, all_ret))

        # transaction cost: proportional to the total weight shift (turnover)
        turnover = float(np.abs(action - self.weights).sum())
        cost     = self.tx_cost * turnover

        # Sharpe-scaled reward: encourages risk-adjusted returns not raw returns.
        # Rolling volatility is estimated from the past 20 steps' returns.
        rolling_vol = float(np.std(self.recent_rets[-20:]) + 1e-8)
        reward      = (port_ret - cost) / rolling_vol

        self.recent_rets.append(port_ret)
        self.portfolio_value *= (1 + port_ret - cost)
        self.weights  = action
        self.t       += 1

        terminated = self.t >= self.episode_len - 1
        info = {
            'portfolio_value': round(self.portfolio_value, 2),
            'step_return':     round(port_ret, 6),
        }
        return self._obs(), float(reward), terminated, False, info

    def render(self):
        print(f't={self.t}  value=${self.portfolio_value:.2f}')

    # ── Internal helpers ────────────────────────────────────────────────────

    def _obs(self) -> np.ndarray:
        parts = []
        for ticker in self.tickers:
            row = self.feat_matrix[ticker][self.t] if self.t < len(self.feat_matrix[ticker]) else np.zeros(N_FEATURES)
            parts.append(row[:N_FEATURES])
        return np.concatenate([*parts, self.weights]).astype(np.float32)

    def _returns_at(self, t: int) -> np.ndarray:
        ret = np.zeros(self.n_stocks, dtype=np.float32)
        for i, ticker in enumerate(self.tickers):
            arr = self.ret_arrays[ticker]
            if t < len(arr):
                ret[i] = float(arr[t])
        return ret

    def _preprocess(self):
        """
        Pre-build numpy feature matrices and return arrays for each ticker.
        Doing this once at environment creation avoids repeated DataFrame
        lookups during the hot loop inside step().
        """
        self.feat_matrix = {}
        self.ret_arrays  = {}
        self.recent_rets = [0.0] * 20   # initialise rolling return buffer
        min_len = 9999

        for ticker in self.tickers:
            df   = self.master_data[ticker].ffill().fillna(0.0)
            avail = [c for c in FEATURE_COLS if c in df.columns]
            X    = df[avail].values.astype(np.float32)

            # pad with zeros for any features not present in this ticker's data
            if X.shape[1] < N_FEATURES:
                X = np.pad(X, ((0, 0), (0, N_FEATURES - X.shape[1])))

            self.feat_matrix[ticker] = X
            self.ret_arrays[ticker]  = (
                df['daily_return'].fillna(0.0).values.astype(np.float32)
                if 'daily_return' in df.columns else np.zeros(len(df), dtype=np.float32)
            )
            min_len = min(min_len, len(X))

        self.episode_len = min_len
        logger.info(f'  Episode length: {self.episode_len} steps per ticker')
