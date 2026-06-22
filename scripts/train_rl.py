import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.data.fusion import FeatureFusion
from backend.models.rl_env import PortfolioEnv
from backend.models.rl_agent import RLPortfolioAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s', datefmt='%H:%M:%S')


def main():
    print('\nPPO Portfolio Agent')
    print('-' * 36)

    files = sorted(Path('data/universe').glob('*.json'), key=lambda f: f.stat().st_mtime)
    if not files:
        print('No universe found — run the main pipeline first')
        sys.exit(1)

    data    = json.loads(files[-1].read_text())
    tickers = data['tickers']
    print(f'  {data["market"]} / {data["index"]} — {len(tickers)} tickers')

    fuser       = FeatureFusion()
    master_data = {t: df for t in tickers
                   if (df := fuser.load_master(t)) is not None and len(df) > 60}
    print(f'  {len(master_data)} tickers with sufficient history')

    if len(master_data) < 2:
        print('  Need at least 2 tickers — run the pipeline first')
        sys.exit(1)

    env = PortfolioEnv(master_data, tickers=list(master_data.keys()))
    print(f'  obs={env.observation_space.shape}  act={env.action_space.shape}  T={env.episode_len}')

    print('\n  1. Quick  —  10 000 steps  (~2 min)')
    print('  2. Standard — 100 000 steps  (~20 min)')
    print('  3. Full  — 500 000 steps  (~90 min)')
    steps = {'1': 10_000, '2': 100_000, '3': 500_000}.get(input('  Select: ').strip(), 100_000)

    agent = RLPortfolioAgent()
    agent.train(env, total_timesteps=steps)

    print('\nEvaluating...')
    m = agent.evaluate(env, n_episodes=5)
    print(f'  avg return {m["avg_return"]:+.2f}%  sharpe {m["sharpe"]:.3f}  max DD {m["max_drawdown"]:.2f}%')
    print(f'\nSaved to models/rl/\n')


if __name__ == '__main__':
    main()
