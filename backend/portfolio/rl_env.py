"""
rl_env.py

Gymnasium portfolio-allocation environment for the independent PPO agent.

The agent observes, for every candidate stock, a few cross-stock-transferable
market features (momentum, volatility, RSI, relative volume, price-vs-SMA) and its own
current weights, and outputs target weights (with a cash slot). It observes at
the close of day t-1, executes at the close of day t, and earns t -> t+1, so it can never be rewarded
using a return it already saw. Reward is return net of transaction costs, scaled
by recent realised volatility.

Independence (project architecture): the PORTFOLIO leg does not consume the
forecasting models. This environment therefore takes NO model-signal input - its
observation is historical market features only. A [start_idx, end_idx] window
restricts episodes to a date range so train and test never overlap.

Reference: Schulman et al. (2017) Proximal Policy Optimization.
"""

import logging

import gymnasium as gym
import numpy as np
import pandas as pd

from backend.config.settings import RL_STATE_FEATURES, TX_COST
from backend.portfolio.transaction_costs import transaction_cost

logger = logging.getLogger(__name__)

N_REGIMES = 3  # optional bull/bear/sideways one-hot (zeros when unused)
VOL_FLOOR = 1e-3  # floor on the reward's risk scaler so early steps can't explode
MIN_EPISODE = 20  # shortest episode when sampling a random training start


class PortfolioEnv(gym.Env):
    """Simulate multi-asset allocation, equity trading costs and cash holdings."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        master_data,
        tickers,
        signals=None,
        regimes=None,
        start_idx=None,
        end_idx=None,
        initial_balance=10_000.0,
        tx_cost=TX_COST,
        deterministic_reset=False,
        strict_features=False,
    ):
        super().__init__()
        from gymnasium import spaces

        # `signals` is accepted for backward call-compatibility but intentionally
        # IGNORED: the portfolio leg is firewalled from model forecasts.
        self.tickers = [t for t in tickers if t in master_data and len(master_data[t]) > 60]
        self.n_stocks = len(self.tickers)
        if self.n_stocks == 0:
            raise ValueError("No tickers with sufficient history - run the data pipeline first")

        self.master_data = master_data
        self.regimes_in = regimes
        self.initial_balance = initial_balance
        self.tx_cost = tx_cost
        self.n_feat = len(RL_STATE_FEATURES)  # market features only (no model signal)
        # deterministic_reset: every reset starts at start_idx and runs the WHOLE
        # window - required for a defensible OOS evaluation (one fixed test period,
        # not a random sub-window per episode). strict_features: raise on a missing
        # state feature / return series instead of silently zero-filling it, so a
        # research run fails loudly rather than reporting fabricated flat data.
        self.deterministic_reset = bool(deterministic_reset)
        self.strict_features = bool(strict_features)

        self._preprocess()

        self.start_idx = 30 if start_idx is None else max(30, int(start_idx))
        # The final action earns the return into the last observation and then terminates.
        self.end_idx = self.episode_len - 1 if end_idx is None else min(int(end_idx), self.episode_len - 1)

        obs_dim = self.n_stocks * self.n_feat + N_REGIMES + (self.n_stocks + 1)
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.action_space = spaces.Box(0.0, 1.0, (self.n_stocks + 1,), np.float32)
        logger.info(
            f"PortfolioEnv: {self.n_stocks} stocks · obs={obs_dim} · "
            f"aligned dates={self.episode_len} · window=[{self.start_idx},{self.end_idx}]"
        )

    def reset(self, seed=None, options=None):
        """Reset portfolio value and the simulation cursor for a new episode."""
        super().reset(seed=seed)  # seeds self.np_random
        if self.deterministic_reset:
            # evaluation: always start at the beginning of the window and run to its
            # end, so every episode covers the SAME complete out-of-sample period.
            self.t = self.start_idx
        else:
            # Sample training starts while reserving a minimum episode length.
            hi = max(self.start_idx + 1, self.end_idx - MIN_EPISODE)
            self.t = int(self.np_random.integers(self.start_idx, hi)) if hi > self.start_idx else self.start_idx
        self.weights = np.zeros(self.n_stocks + 1, dtype=np.float32)
        self.weights[-1] = 1.0  # start fully in cash
        self.portfolio_value = self.initial_balance
        self.recent_rets = []  # grows over the episode; floored in step()
        return self._obs(), {}

    def step(self, action):
        """Execute target weights, deduct equity turnover costs and advance one session."""
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.n_stocks + 1,) or not np.isfinite(action).all():
            raise ValueError("Portfolio action must contain finite stock and cash weights")
        action = np.maximum(action, 0)
        if action.sum() == 0:
            action[-1] = 1
        action /= action.sum()

        # The observed features end at t-1; execution at t earns the return
        # realised over t -> t+1 (index t+1), never the already-known return into t.
        nxt = self.t + 1
        stock_ret = self._returns_at(nxt)
        all_ret = np.append(stock_ret, 0.0)  # cash earns zero
        port_ret = float(np.dot(action, all_ret))

        turnover = float(np.abs(action[:-1] - self.weights[:-1]).sum())
        cost = transaction_cost(turnover, self.tx_cost)

        net_ret = (1 - cost) * (1 + port_ret) - 1
        if nxt >= self.end_idx:
            terminal_equity_weight = float(np.dot(action[:-1], 1 + stock_ret)) / max(1 + port_ret, 1e-9)
            net_ret = (1 + net_ret) * (1 - transaction_cost(terminal_equity_weight, self.tx_cost)) - 1
        # risk scaler floored so the first (near-empty) buffer can't divide by ~0.
        # recent_rets stores NET returns, so numerator and denominator match.
        denom = max(float(np.std(self.recent_rets[-20:])) if len(self.recent_rets) >= 5 else VOL_FLOOR, VOL_FLOOR)
        reward = net_ret / denom

        self.recent_rets.append(net_ret)
        self.portfolio_value *= 1 + net_ret

        # WEIGHT DRIFT: after the market moves, holdings are no longer the chosen
        # weights - each grows by (1+r_i). Carry the drifted weights forward so the
        # next step's turnover/cost is measured from the real position.
        grown = action * (1.0 + all_ret)
        tot = float(grown.sum())
        self.weights = (grown / tot).astype(np.float32) if tot > 0 else action

        self.t = nxt
        terminated = self.t >= self.end_idx
        # Expose both legs explicitly: gross is before costs, net is after (matches
        # the value/reward update).
        info = {
            "portfolio_value": self.portfolio_value,
            "gross_step_return": round(port_ret, 6),
            "net_step_return": round(net_ret, 6),
        }
        return self._obs(), float(reward), terminated, False, info

    def render(self):
        """Log the current simulation session and portfolio value."""
        logger.info(f"t={self.t}  value=${self.portfolio_value:.2f}")

    def _obs(self):
        parts = []
        for ticker in self.tickers:
            feats = (
                self.feat_matrix[ticker][self.t - 1]
                if self.t < len(self.feat_matrix[ticker])
                else np.zeros(self.n_feat, dtype=np.float32)
            )
            parts.append(feats)
        regime = np.zeros(N_REGIMES, dtype=np.float32)
        if self.regime_arr is not None and self.t < len(self.regime_arr):
            r = int(self.regime_arr[self.t - 1])
            if 0 <= r < N_REGIMES:  # a valid regime -> one-hot
                regime[r] = 1.0  # out-of-range r leaves it all-zeros
                # Invalid regime codes leave the encoding at zero.
        return np.concatenate([*parts, regime, self.weights]).astype(np.float32)

    def _returns_at(self, t):
        ret = np.zeros(self.n_stocks, dtype=np.float32)
        for i, ticker in enumerate(self.tickers):
            arr = self.ret_arrays[ticker]
            if 0 <= t < len(arr):
                ret[i] = float(arr[t])
        return ret

    def _preprocess(self):
        """Align ticker arrays to their common calendar dates."""
        idx_sets = []
        for t in self.tickers:
            idx = pd.to_datetime(self.master_data[t].index)
            idx_sets.append(set(idx))
        common = sorted(set.intersection(*idx_sets)) if idx_sets else []
        self.dates = pd.DatetimeIndex(common)
        self.episode_len = len(self.dates)
        if self.episode_len < 60:
            raise ValueError("Not enough overlapping dates across tickers for an episode")

        self.feat_matrix, self.ret_arrays = {}, {}
        for ticker in self.tickers:
            df = self.master_data[ticker].copy()
            df.index = pd.to_datetime(df.index)
            df = df.reindex(self.dates)  # align to the common date grid
            missing = [c for c in RL_STATE_FEATURES if c not in df.columns]
            if missing:
                msg = f"{ticker}: missing RL state features {missing}"
                if self.strict_features:
                    raise ValueError(msg + " - refusing to zero-fill in a strict run")
                logger.warning(msg + " - zero-filling (schema gap)")
            if "daily_return" not in df.columns:
                msg = f"{ticker}: no daily_return column"
                if self.strict_features:
                    raise ValueError(msg + " - refusing to treat as 0% returns in a strict run")
                logger.warning(msg + " - treating as 0% returns (schema gap)")
            # Reindex to the fixed feature list so every feature keeps its column
            # position; a missing one becomes 0 in place rather than shifting the
            # later features into the wrong network inputs.
            X_df = df.reindex(columns=RL_STATE_FEATURES).ffill()
            if self.strict_features and X_df.isna().to_numpy().any():
                raise ValueError(
                    f"{ticker}: state features still incomplete after ffill; "
                    "refusing to zero-fill observations in a strict run"
                )
            X = X_df.fillna(0.0).to_numpy(dtype=np.float32)
            self.feat_matrix[ticker] = X
            self.ret_arrays[ticker] = (
                df["daily_return"].fillna(0.0).values.astype(np.float32)
                if "daily_return" in df.columns
                else np.zeros(self.episode_len, np.float32)
            )

        # optional regime series, aligned to the common dates when supplied
        self.regime_arr = None
        if self.regimes_in is not None:
            r = pd.Series(self.regimes_in)
            if isinstance(r.index, pd.DatetimeIndex):
                self.regime_arr = r.reindex(self.dates).ffill().fillna(0).values
            elif len(r) == self.episode_len:
                self.regime_arr = r.values
        logger.info(f"  aligned episode length: {self.episode_len} common dates")
