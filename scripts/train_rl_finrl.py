import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s')
logger = logging.getLogger(__name__)

FEATURES_DIR = Path('data/features')
MODELS_DIR   = Path('models/rl')
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# FinRL feature subset — must exist in the feature engineer output
FINRL_FEATURES = [
    'close', 'rsi', 'macd', 'momentum_10d', 'volatility',
    'sma20', 'close_to_sma20', 'daily_return', 'volume_ratio', 'sent_score',
]


def load_data(tickers: list) -> pd.DataFrame:
    frames = []
    for ticker in tickers:
        for path in [FEATURES_DIR / f'{ticker}_master.csv', FEATURES_DIR / f'{ticker}.csv']:
            if path.exists():
                df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
                df['tic'] = ticker
                frames.append(df)
                break
        else:
            logger.warning(f'{ticker}: no feature file found')

    if not frames:
        raise RuntimeError('No feature data — run the data pipeline first')

    combined = pd.concat(frames)
    combined.index.name = 'date'
    return combined.reset_index()


def train(tickers: list, timesteps: int = 100_000):
    try:
        from finrl.meta.env_portfolio_optimization.env_portfolio_optimization import StockPortfolioEnv
        from stable_baselines3 import PPO
        use_finrl = True
        logger.info('FinRL available — using StockPortfolioEnv with Sharpe reward')
    except ImportError:
        use_finrl = False
        logger.warning('FinRL not installed — falling back to custom env (pip install finrl)')

    df      = load_data(tickers)
    present = [c for c in FINRL_FEATURES if c in df.columns]
    logger.info(f'{len(df["tic"].unique())} tickers  {len(df)} rows  {len(present)} features')

    if use_finrl:
        env = StockPortfolioEnv(
            df=df,
            stock_dim=len(tickers),
            hmax=1.0,
            initial_amount=100_000,
            transaction_cost_pct=0.001,
            tech_indicator_list=present,
            reward_scaling=1e-4,      # scale rewards so gradients stay numerically stable
        )
        model = PPO('MlpPolicy', env, verbose=1,
                    learning_rate=3e-4, n_steps=2048, batch_size=64,
                    n_epochs=10, ent_coef=0.01, device='auto')
        logger.info(f'Training for {timesteps:,} timesteps')
        model.learn(total_timesteps=timesteps)
        path = str(MODELS_DIR / 'ppo_finrl_final')
        model.save(path)
        logger.info(f'Saved to {path}.zip')
    else:
        from backend.models.rl_agent import RLPortfolioAgent
        from backend.models.rl_env import PortfolioEnv
        from backend.data.fusion import FeatureFusion
        fuser = FeatureFusion()
        md    = {t: fuser.load_master(t) for t in tickers if fuser.load_master(t) is not None}
        env   = PortfolioEnv(md, tickers=list(md.keys()))
        RLPortfolioAgent().train(env, total_timesteps=timesteps)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--tickers',    nargs='+', default=['AAPL','MSFT','NVDA','GOOGL','AMZN'])
    p.add_argument('--timesteps',  type=int,  default=100_000)
    args = p.parse_args()
    logger.info(f'tickers: {args.tickers}')
    train(args.tickers, args.timesteps)
