"""
Drift Monitor - detects when the live market has shifted away from training data.

Monitors two signals that indicate the model may need retraining.
  Feature drift: the live market's indicator distributions shift away from
  what the model was trained on, measured by Z-score of the mean shift.
  Performance drift: the 30-day recommendation win rate drops below a floor.

  Markets are non-stationary (Lo, 2004). A model trained in a low-volatility
  bull market will deteriorate in a high-volatility bear market even with
  no code changes. This monitor provides an early warning before recommendation
  quality visibly degrades.
"""

import json
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from backend.config.settings import PREDICTION_HORIZON

logger = logging.getLogger(__name__)

DRIFT_LOG_PATH = Path("data/drift_log.json")
OUTCOMES_PATH = Path("data/recommendation_outcomes.json")

# Thresholds
FEATURE_DRIFT_THRESHOLD = 2.0  # Z-score of mean shift vs training baseline
WIN_RATE_FLOOR = 0.45  # below this triggers retraining
ROLLING_WINDOW_DAYS = 30


class DriftMonitor:
    """Track recommendation outcomes and rolling predictive performance."""

    def __init__(self):
        DRIFT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    def check_feature_drift(self, current_df: pd.DataFrame, baseline_stats: dict) -> dict:
        """
        Compare current feature distributions against training baseline.
        Uses a simple Z-score on the mean shift, scaled by baseline std.

        baseline_stats format: {'feature_name': {'mean': float, 'std': float}}
        """
        if not baseline_stats or current_df is None or current_df.empty:
            return {"drifted": False, "features": {}, "max_z": 0.0}

        results = {}
        z_scores = []

        numeric = current_df.select_dtypes(include="number")
        for col in numeric.columns:
            if col not in baseline_stats:
                continue
            b_mean = baseline_stats[col]["mean"]
            b_std = baseline_stats[col].get("std", 1.0) or 1.0
            c_mean = float(numeric[col].mean())
            z = abs(c_mean - b_mean) / b_std
            results[col] = round(z, 3)
            z_scores.append(z)

        max_z = max(z_scores) if z_scores else 0.0
        drifted = max_z > FEATURE_DRIFT_THRESHOLD

        # Log the check
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "max_z_score": round(max_z, 3),
            "drifted": drifted,
            "top_drifted": sorted(results.items(), key=lambda x: -x[1])[:5],
        }
        self._append_log(log_entry)

        if drifted:
            logger.warning(f"Drift detected: max Z={max_z:.2f} (threshold={FEATURE_DRIFT_THRESHOLD})")

        return {"drifted": drifted, "features": results, "max_z": round(max_z, 3)}

    def compute_baseline_stats(self, training_df: pd.DataFrame) -> dict:
        """
        Compute mean and std for each numeric feature over the training period.
        Call this after training and store the result in the model registry.
        """
        if training_df is None or training_df.empty:
            return {}
        numeric = training_df.select_dtypes(include="number")
        stats = {}
        for col in numeric.columns:
            col_vals = numeric[col].dropna()
            if len(col_vals) > 10:
                stats[col] = {
                    "mean": round(float(col_vals.mean()), 6),
                    "std": round(float(col_vals.std()), 6),
                }
        return stats

    # Save the recommendation so its outcome can be checked once the horizon elapses
    def log_recommendation(
        self,
        ticker: str,
        action: str,
        predicted_return: float,
        date: str | None = None,
        horizon: int = PREDICTION_HORIZON,
        model_version=None,
    ):
        """Record a recommendation so its outcome can be evaluated once the model's
        HORIZON has elapsed. We store the prediction date, the horizon, the model
        version and (later) the evaluation date, so the outcome is measured over the
        intended holding period - not over 'whatever history exists now'."""
        entry = {
            "ticker": ticker,
            "action": action,
            "predicted_return": predicted_return,
            "date": date or datetime.now().strftime("%Y-%m-%d"),
            "horizon": int(horizon),
            "model_version": model_version,
            "evaluated": False,
            "evaluated_date": None,
            "actual_return": None,
            "correct": None,
        }
        outcomes = self._load_outcomes()
        outcomes.append(entry)
        self._save_outcomes(outcomes)

    # Count how many of the last N days of evaluated recommendations were correct
    def rolling_win_rate(self, lookback_days: int = ROLLING_WINDOW_DAYS) -> dict:
        """
        Compute win rate over the last N days of evaluated recommendations.
        Returns dict with win_rate, n_evaluated, and whether retraining is needed.
        """
        outcomes = self._load_outcomes()
        cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        # Filter on when the outcome was completed (evaluated_date), not when the
        # recommendation was made, so the rolling window contains completed outcomes.
        # Falls back to the rec date for records written before evaluated_date existed.
        recent = [
            o for o in outcomes if o.get("evaluated") and (o.get("evaluated_date") or o.get("date", "")) >= cutoff
        ]

        if len(recent) < 5:
            return {
                "win_rate": None,
                "n_evaluated": len(recent),
                "needs_retraining": False,
                "note": "Not enough evaluated recs yet",
            }

        wins = sum(1 for o in recent if o.get("correct"))
        win_rate = wins / len(recent)
        needs_retrain = win_rate < WIN_RATE_FLOOR

        if needs_retrain:
            logger.warning(f"Win rate {win_rate:.1%} below floor {WIN_RATE_FLOOR:.1%} - retraining recommended")

        return {
            "win_rate": round(win_rate, 3),
            "n_evaluated": len(recent),
            "n_wins": wins,
            "needs_retraining": needs_retrain,
            "lookback_days": lookback_days,
        }

    # Once the horizon (PREDICTION_HORIZON sessions) has elapsed, look up the actual return and mark correct/wrong
    def evaluate_past_recommendations(self, master_data: dict):
        """
        For recommendations made 30+ days ago, look up actual returns and mark correct/wrong.
        A BUY is 'correct' if the stock rose over the holding period.
        """
        outcomes = self._load_outcomes()
        changed = False

        for o in outcomes:
            if o.get("evaluated"):
                continue
            ticker = o.get("ticker")
            df = master_data.get(ticker)
            if df is None or df.empty:
                continue

            rec_date = pd.Timestamp(o["date"])
            H = int(o.get("horizon", PREDICTION_HORIZON))
            future_rows = df[df.index > rec_date]
            base = df.loc[df.index <= rec_date, "close"]
            # Only evaluate once the FULL horizon has elapsed (H trading sessions of
            # future data exist); otherwise the outcome isn't in yet - skip, don't
            # measure a partial period.
            if base.empty or len(future_rows) < H:
                continue

            entry_close = float(base.iloc[-1])
            horizon_close = float(future_rows["close"].iloc[H - 1])  # close H sessions later, NOT the latest
            actual_return = horizon_close / entry_close - 1.0
            is_correct = (o["action"] == "BUY" and actual_return > 0) or (o["action"] == "SELL" and actual_return < 0)

            o["actual_return"] = round(actual_return * 100, 3)
            o["correct"] = is_correct
            o["evaluated"] = True
            o["evaluated_date"] = datetime.now().strftime("%Y-%m-%d")
            changed = True

        if changed:
            self._save_outcomes(outcomes)

        return outcomes

    # Read the outcomes file or return an empty list if it does not exist yet
    def _load_outcomes(self) -> list:
        if OUTCOMES_PATH.exists():
            try:
                return json.loads(OUTCOMES_PATH.read_text())
            except Exception as e:
                # Preserve a corrupt outcomes log instead of silently discarding the
                # recommendation history that drift monitoring depends on.
                backup = OUTCOMES_PATH.with_suffix(".corrupt")
                try:
                    OUTCOMES_PATH.replace(backup)
                    logger.error("Outcomes log unreadable (%s); moved to %s", e, backup)
                except Exception:
                    logger.error("Outcomes log unreadable (%s) and could not be preserved", e)
                return []
        return []

    def _save_outcomes(self, outcomes: list):
        # Atomic write (temp + fsync + rename) so a crash mid-write cannot corrupt
        # the outcomes log.
        tmp = OUTCOMES_PATH.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(outcomes, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, OUTCOMES_PATH)

    # Append to the drift log, keeping only the last 100 entries to prevent unbounded growth
    def _append_log(self, entry: dict):
        log = []
        if DRIFT_LOG_PATH.exists():
            try:
                log = json.loads(DRIFT_LOG_PATH.read_text())
            except Exception:
                log = []
        log.append(entry)
        log = log[-100:]  # keep last 100 checks
        DRIFT_LOG_PATH.write_text(json.dumps(log, indent=2))
