"""HTTP routes for evaluation."""

import json
import logging
import sys
import threading
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException

from backend import runtime
from backend.infra.model_registry import ModelRegistry

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/api/model-registry")
def get_model_registry():
    """Return the full version history for both models, including
    walk-forward AUC and deployment decisions for each training run."""
    return ModelRegistry().list_all()


@router.get("/api/drift-status")
def get_drift_status():
    """Return the 30-day recommendation win rate and whether the
    system recommends triggering a retrain."""
    tickers = {r["ticker"] for r in runtime.drift_monitor._load_outcomes()}
    runtime.drift_monitor.evaluate_past_recommendations({t: runtime.fusion.load_master(t) for t in tickers})
    win_rate = runtime.drift_monitor.rolling_win_rate()
    return {
        "win_rate_stats": win_rate,
        "retraining_needed": win_rate.get("needs_retraining", False),
        "recommendation": (
            "Retraining recommended - win rate below threshold"
            if win_rate.get("needs_retraining")
            else (
                "Insufficient evaluated outcomes" if win_rate.get("win_rate") is None else "No performance-drift alert"
            )
        ),
    }


@router.get("/api/recommendation-outcomes")
def get_recommendation_outcomes():
    """Return all logged recommendations whose horizon (21 trading sessions) has
    elapsed, with their evaluated outcomes."""
    outcomes = runtime.drift_monitor._load_outcomes()
    evaluated = [o for o in outcomes if o.get("evaluated")]
    pending = [o for o in outcomes if not o.get("evaluated")]
    wins = sum(1 for o in evaluated if o.get("correct"))
    return {
        "total_logged": len(outcomes),
        "evaluated": len(evaluated),
        "pending": len(pending),
        "overall_wins": wins,
        "win_rate": round(wins / len(evaluated), 3) if evaluated else None,
        "recent": outcomes[-20:],
    }


@router.get("/api/experiment-results")
def experiment_results():
    """Serve the saved leakage-free experiment matrix (scripts/run_experiments.py)."""
    path = Path("data/experiments/experiment_results.json")
    if not path.exists():
        raise HTTPException(404, "No experiment results yet - run scripts/run_experiments.py")
    result = json.loads(path.read_text())
    if result.get("schema_version") != 3 or result.get("trading_protocol") != "next_close_v2":
        raise HTTPException(409, "Saved experiments predate the corrected methodology; regenerate them")
    return result


@router.get("/api/portfolio-comparison")
def portfolio_comparison_results():
    """Serve the saved portfolio comparison (scripts/train_ppo_oos.py)."""
    path = Path("data/experiments/portfolio_results.json")
    if not path.exists():
        raise HTTPException(404, "No portfolio comparison yet - run scripts/train_ppo_oos.py")
    result = json.loads(path.read_text())
    if result.get("schema_version") != 3 or result.get("trading_protocol") != "next_close_v2":
        raise HTTPException(409, "Saved comparison predates the corrected methodology; regenerate it")
    return result


_eval_job = {"state": "idle", "started_at": None, "finished_at": None, "returncode": None, "error": None}


_eval_lock = threading.Lock()


def _run_eval_job():
    import subprocess

    try:
        r = subprocess.run(
            [sys.executable, "scripts/run_experiments.py", "--sp500-only"], check=False, capture_output=True, text=True
        )
        with _eval_lock:
            _eval_job.update(
                state=("succeeded" if r.returncode == 0 else "failed"),
                finished_at=datetime.now().isoformat(),
                returncode=r.returncode,
                error=(r.stderr[-2000:] if r.returncode != 0 else None),
            )
    except Exception as e:  # noqa: BLE001
        with _eval_lock:
            _eval_job.update(state="failed", finished_at=datetime.now().isoformat(), error=str(e))


@router.post("/api/evaluate-all")
def run_evaluation(bg: BackgroundTasks):
    """Schedule one experiment job and prevent overlapping evaluations."""
    with _eval_lock:
        if _eval_job["state"] == "running":
            return {"status": "already_running", "started_at": _eval_job["started_at"]}
        _eval_job.update(
            state="running", started_at=datetime.now().isoformat(), finished_at=None, returncode=None, error=None
        )
    bg.add_task(_run_eval_job)
    return {"status": "running", "started_at": _eval_job["started_at"]}


@router.get("/api/evaluate-all/status")
def evaluation_status():
    """Report the tracked evaluation job's state, timestamps and exit code."""
    with _eval_lock:
        return dict(_eval_job)
