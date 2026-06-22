import logging
from pathlib import Path

import numpy as np

logger     = logging.getLogger(__name__)
MODELS_DIR = Path('models/rl')

# continuous action space rules out DQN — PPO (Schulman et al., 2017) handles
# it natively via a Gaussian policy head and avoids the large destructive
# parameter updates that destabilise training on financial time series


class RLPortfolioAgent:

    def __init__(self):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.model = None

    def train(self, env, total_timesteps: int = 100_000) -> dict:
        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.callbacks import CheckpointCallback
            from stable_baselines3.common.monitor import Monitor
        except ImportError:
            raise ImportError('pip install stable-baselines3 gymnasium')

        env = Monitor(env)

        checkpoint = CheckpointCallback(
            save_freq=20_000,
            save_path=str(MODELS_DIR),
            name_prefix='ppo_checkpoint',
            verbose=0,
        )

        self.model = PPO(
            policy='MlpPolicy',
            env=env,
            learning_rate=3e-4,
            n_steps=2048,        # rollout length: long enough to capture multi-day trends
            batch_size=64,
            n_epochs=10,
            gamma=0.99,          # high discount: rewards 100 steps ahead still matter
            gae_lambda=0.95,     # standard GAE variance/bias trade-off
            clip_range=0.2,      # PPO clip: prevents destructive policy steps
            ent_coef=0.005,      # small entropy bonus prevents collapse to "always hold cash"
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs={'net_arch': [256, 128, 64]},  # bottleneck forces cross-stock compression
            verbose=1,
        )

        logger.info(f'PPO training: {total_timesteps:,} timesteps')
        self.model.learn(total_timesteps=total_timesteps, callback=checkpoint, progress_bar=True)
        self.save()
        return {'total_timesteps': total_timesteps, 'saved_to': str(MODELS_DIR)}

    def get_weights(self, obs: np.ndarray) -> np.ndarray | None:
        if not self._loaded():
            return None
        # deterministic=True disables exploration noise that is only needed during training
        action, _ = self.model.predict(obs, deterministic=True)
        action     = np.abs(action)
        return action / (action.sum() + 1e-8)

    def evaluate(self, env, n_episodes: int = 5) -> dict:
        if not self._loaded():
            return {'error': 'no model loaded'}

        all_returns = []
        for _ in range(n_episodes):
            obs, _  = env.reset()
            done    = False
            ep_vals = [env.portfolio_value]
            while not done:
                w = self.get_weights(obs)
                obs, _, terminated, truncated, info = env.step(w)
                done = terminated or truncated
                ep_vals.append(info.get('portfolio_value', ep_vals[-1]))
            all_returns.append((ep_vals[-1] - ep_vals[0]) / (ep_vals[0] + 1e-8))

        daily  = np.diff(ep_vals) / (np.array(ep_vals[:-1]) + 1e-8)
        sharpe = float((daily.mean() / (daily.std() + 1e-8)) * np.sqrt(252))
        peak   = ep_vals[0]
        max_dd = 0.0
        for v in ep_vals:
            peak   = max(peak, v)
            max_dd = max(max_dd, (peak - v) / (peak + 1e-8))

        return {
            'avg_return':   round(float(np.mean(all_returns)) * 100, 2),
            'sharpe':       round(sharpe, 3),
            'max_drawdown': round(max_dd * 100, 2),
        }

    def save(self):
        if self.model:
            self.model.save(str(MODELS_DIR / 'ppo_portfolio_final'))
            logger.info(f'RL agent saved to {MODELS_DIR}')

    def is_trained(self) -> bool:
        return (MODELS_DIR / 'ppo_portfolio_final.zip').exists()

    def _loaded(self) -> bool:
        if self.model:
            return True
        path = MODELS_DIR / 'ppo_portfolio_final.zip'
        if not path.exists():
            logger.warning('No RL model — train it with scripts/train_rl.py')
            return False
        try:
            from stable_baselines3 import PPO
            self.model = PPO.load(str(path.with_suffix('')))
            return True
        except Exception as e:
            logger.error(f'RL load error: {e}')
            return False
