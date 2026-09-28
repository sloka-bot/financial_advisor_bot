"""Train and evaluate a PPO allocation policy with profile-constrained stock and cash weights."""

import json
import logging
from pathlib import Path

import numpy as np

from backend.config.settings import RISK_CONSTRAINTS as RISK_LIMITS
from backend.evaluation.metrics import annualized_sharpe

logger = logging.getLogger(__name__)
MODELS_DIR = Path("models/rl")


class RLPortfolioAgent:
    """Train and evaluate a PPO allocation policy with risk-profile constraints."""

    def __init__(self):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.model = None

    def train(self, env, total_timesteps: int = 100_000, seed: int = None) -> dict:
        """Fit PPO on the training environment and persist weights and progress."""
        import time

        import numpy as np
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
        from stable_baselines3.common.monitor import Monitor

        self.training_tickers = list(env.tickers)
        env = Monitor(env)

        # Record progress and learning metrics after each 2,048-step rollout.
        class _Progress(BaseCallback):
            def __init__(self, total):
                super().__init__()
                self.total = total
                self.curve = []
                self._t0 = time.time()

            def _on_rollout_end(self):
                rews = [e.get("r") for e in self.model.ep_info_buffer] if self.model.ep_info_buffer else []
                mr = float(np.mean(rews)) if rews else float("nan")
                self.curve.append({"timesteps": int(self.num_timesteps), "ep_rew_mean": round(mr, 4)})
                logger.info(
                    f"    PPO {self.num_timesteps:,}/{self.total:,} timesteps  "
                    f"ep_rew_mean={mr:.3f}  ({time.time() - self._t0:.0f}s)"
                )

            def _on_step(self):
                return True

        # Limit PyTorch CPU threads to avoid native-runtime contention on macOS.
        torch.set_num_threads(1)
        logger.info("building PPO model (single-threaded)...")

        checkpoint = CheckpointCallback(
            save_freq=20_000, save_path=str(MODELS_DIR), name_prefix="ppo_checkpoint", verbose=0
        )
        progress = _Progress(total_timesteps)
        self.model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=2048,  # rollout length covers multi-day trends
            batch_size=64,
            n_epochs=10,
            gamma=0.99,  # weight distant rewards
            gae_lambda=0.95,
            clip_range=0.2,  # clip keeps policy updates small
            ent_coef=0.005,  # entropy bonus discourages an all-cash policy
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs={"net_arch": [256, 128, 64]},
            seed=seed,  # per-run seed
            verbose=0,
        )
        self.seed = seed
        logger.info(f"PPO training: {total_timesteps:,} timesteps (progress logged every rollout)")
        self.model.learn(total_timesteps=total_timesteps, callback=[checkpoint, progress], progress_bar=False)
        self.learning_curve = progress.curve
        self.save()
        return {"total_timesteps": total_timesteps, "saved_to": str(MODELS_DIR), "learning_curve": progress.curve}

    def get_weights(self, obs: np.ndarray):
        """Return deterministic normalised policy weights, or None without a model."""
        if not self._loaded():
            return None
        action, _ = self.model.predict(obs, deterministic=True)  # deterministic inference
        action = np.abs(action)
        return action / (action.sum() + 1e-8)

    def constrain(self, weights, risk_profile: str = "moderate") -> np.ndarray:
        """Cap equity positions and allocate residual weight to cash."""
        lim = RISK_LIMITS.get(risk_profile, RISK_LIMITS["moderate"])
        w = np.abs(np.asarray(weights, dtype=float))
        w /= w.sum() + 1e-8
        stocks, cash = w[:-1].copy(), float(w[-1])
        stocks = np.minimum(stocks, lim["max_weight"])  # cap each position
        invest_budget = max(0.0, 1.0 - lim["min_cash"])
        if stocks.sum() > invest_budget and stocks.sum() > 0:  # scale down to keep minimum cash
            stocks *= invest_budget / stocks.sum()
        cash = 1.0 - stocks.sum()
        out = np.append(stocks, cash)
        return out / (out.sum() + 1e-8)

    def build_portfolio(self, obs, risk_profile: str = "moderate"):
        """Convert the current observation into profile-constrained target weights."""
        weights = self.get_weights(obs)
        if weights is None:
            return None
        return self.constrain(weights, risk_profile)

    def rebalance(self, obs, current_weights, risk_profile: str = "moderate"):
        """Return target weights and signed changes from the current allocation."""
        target = self.build_portfolio(obs, risk_profile)
        if target is None:
            return None
        current = np.asarray(current_weights, dtype=float)
        current = current / (current.sum() + 1e-8)
        delta = target - current
        return {"target_weights": target, "delta": delta}  # positive buys, negative sells

    def evaluate(self, env, n_episodes: int = 5, risk_profile: str = "moderate") -> dict:
        """Evaluate profile-constrained actions and retain the worst drawdown across episodes."""
        if not self._loaded():
            return {"error": "no model loaded"}
        episode_returns, daily_returns, episode_mdds = [], [], []
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = False
            values = [env.portfolio_value]
            while not done:
                w = self.constrain(self.get_weights(obs), risk_profile)  # trade the constrained weights
                obs, _, terminated, truncated, info = env.step(w)
                done = terminated or truncated
                values.append(info.get("portfolio_value", values[-1]))
            values = np.asarray(values, dtype=float)
            episode_returns.append(values[-1] / (values[0] + 1e-8) - 1.0)
            daily_returns.extend(np.diff(values) / (values[:-1] + 1e-8))
            peak, mdd = values[0], 0.0
            for v in values:
                peak = max(peak, v)
                mdd = max(mdd, (peak - v) / (peak + 1e-8))
            episode_mdds.append(mdd)
        daily = np.asarray(daily_returns)
        sharpe = annualized_sharpe(daily, 252, eps=1e-8, min_periods=1)
        worst_mdd = float(max(episode_mdds)) if episode_mdds else 0.0
        return {
            "avg_return": round(float(np.mean(episode_returns)) * 100, 2),
            "sharpe": round(sharpe, 3),
            "max_drawdown": round(worst_mdd * 100, 2),  # worst across episodes
            "mean_max_drawdown": round(float(np.mean(episode_mdds)) * 100, 2) if episode_mdds else 0.0,
            "risk_profile": risk_profile,
            "n_episodes": n_episodes,
        }

    def save(self):
        """Persist the fitted PPO policy when one is available."""
        if self.model:
            self.model.save(str(MODELS_DIR / "ppo_portfolio_final"))
            (MODELS_DIR / "metadata.json").write_text(
                json.dumps(
                    {
                        "trading_protocol": "next_close_v2",
                        "tickers": getattr(self, "training_tickers", []),
                        "seed": getattr(self, "seed", None),
                    },
                    indent=2,
                )
            )
            logger.info(f"RL agent saved to {MODELS_DIR}")

    def is_trained(self) -> bool:
        """Check whether a saved PPO policy artifact exists."""
        if not (MODELS_DIR / "ppo_portfolio_final.zip").exists():
            return False
        try:
            metadata = json.loads((MODELS_DIR / "metadata.json").read_text())
        except (OSError, ValueError):
            return False
        return metadata.get("trading_protocol") == "next_close_v2"

    # Load the saved policy on first use.
    def _loaded(self) -> bool:
        if self.model:
            return True
        path = MODELS_DIR / "ppo_portfolio_final.zip"
        if not self.is_trained():
            logger.warning("No PPO model compatible with the current execution protocol")
            return False
        try:
            from stable_baselines3 import PPO

            self.model = PPO.load(str(path.with_suffix("")))
            return True
        except Exception as e:
            logger.error(f"RL load error: {e}")
            return False
