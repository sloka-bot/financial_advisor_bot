"""Monitor feature-distribution shifts and recent recommendation outcomes."""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from backend.config.settings import PREDICTION_HORIZON

logger = logging.getLogger(__name__)

DRIFT_LOG_PATH = Path("data/drift_log.json")
OUTCOMES_PATH = Path("data/recommendation_outcomes.json")

# Thresholds
FEATURE_DRIFT_THRESHOLD = 2.0  # z-score of the mean shift against the training baseline
WIN_RATE_FLOOR = 0.45  # retrain below this win rate
ROLLING_WINDOW_DAYS = 30


class DriftMonitor:
    """Track recommendation outcomes and rolling predictive performance."""

    def __init__(self):
        DRIFT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    def check_feature_drift(self, current_df: pd.DataFrame, baseline_stats: dict) -> dict:
        """Measure feature mean shifts relative to each training standard deviation."""
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

        # Log the check.
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
        """Calculate training means and standard deviations for numeric features."""
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

    def log_recommendation(
        self,
        ticker: str,
        action: str,
        predicted_return: float,
        date: str | None = None,
        horizon: int = PREDICTION_HORIZON,
        model_version=None,
    ):
        """Record a recommendation, horizon and model version for later outcome evaluation."""
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

    def rolling_win_rate(self, lookback_days: int = ROLLING_WINDOW_DAYS) -> dict:
        """Summarise completed outcomes over a rolling window and assess retraining thresholds."""
        outcomes = self._load_outcomes()
        cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        # Select completed outcomes by evaluation date, or recommendation date for older records.
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

    def evaluate_past_recommendations(self, master_data: dict):
        """Score matured recommendations against returns over their stated horizon."""
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
            # Evaluate only after the full trading horizon is observable.
            if base.empty or len(future_rows) < H:
                continue

            entry_close = float(base.iloc[-1])
            horizon_close = float(future_rows["close"].iloc[H - 1])  # close H sessions after the recommendation
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

    # Read the outcomes file, or an empty list when absent.
    def _load_outcomes(self) -> list:
        if OUTCOMES_PATH.exists():
            try:
                return json.loads(OUTCOMES_PATH.read_text())
            except Exception as e:
                # Preserve corrupt outcome files instead of replacing history with an empty log.
                backup = OUTCOMES_PATH.with_suffix(".corrupt")
                try:
                    OUTCOMES_PATH.replace(backup)
                    logger.error("Outcomes log unreadable (%s); moved to %s", e, backup)
                except Exception:
                    logger.error("Outcomes log unreadable (%s) and could not be preserved", e)
                return []
        return []

    def _save_outcomes(self, outcomes: list):
        # Persist outcomes with an atomic file replacement.
        tmp = OUTCOMES_PATH.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(outcomes, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, OUTCOMES_PATH)

    # Append to the drift log, keeping the last 100 entries.
    def _append_log(self, entry: dict):
        log = []
        if DRIFT_LOG_PATH.exists():
            try:
                log = json.loads(DRIFT_LOG_PATH.read_text())
            except Exception:
                log = []
        log.append(entry)
        log = log[-100:]
        DRIFT_LOG_PATH.write_text(json.dumps(log, indent=2))
