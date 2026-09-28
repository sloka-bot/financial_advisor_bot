"""HTTP routes for system."""

import logging

import requests
from fastapi import APIRouter
from fastapi.responses import RedirectResponse

from backend import runtime
from backend.api.schemas import (
    RunRequest,
)
from backend.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/")
def health():
    """Report the application health and API version."""
    return {"status": "ok", "version": "1.1.0"}


@router.get("/api/markets")
def get_markets():
    """Return the single supported market and index."""
    return {"universe": "S&P 500", "market": "United States"}


@router.get("/api/universe")
def get_universe(refresh: bool = False):
    """Return the supported universe and optionally refresh its constituents."""
    tickers = runtime.universe_builder.members(force_refresh=refresh)
    return {"universe": "S&P 500", "market": "United States", "count": len(tickers), "tickers": tickers}


@router.post("/api/run")
def run_pipeline(req: RunRequest):
    """Start a validated training request without blocking the API worker."""
    runtime.start_pipeline(req)
    return {"message": "Pipeline started", "poll": "/api/pipeline-status"}


@router.get("/api/pipeline-status")
def pipeline_status():
    """Return a consistent snapshot of the current pipeline progress."""
    with runtime.state_lock:
        return dict(runtime.pipeline_state)


@router.get("/api/regime")
def get_regime():
    """Return the current market regime detected from the processed tickers."""
    with runtime.state_lock:
        tickers = list(runtime.pipeline_state.get("processed_tickers", []))
    if not tickers:
        return {"regime": "unknown", "confidence": 0, "metrics": {}}
    master_data = {t: df for t in tickers[:20] if (df := runtime.fusion.load_master(t)) is not None and not df.empty}
    return runtime.regime_detector.detect(master_data)


@router.get("/api/model-status")
def model_status():
    """Report whether each deployed model artifact is present."""
    horizons = {}
    for h in settings.HORIZONS:
        xf, lf = runtime.horizon_models(h)
        horizons[h] = {"xgboost": xf.is_trained(), "lstm": lf.is_trained()}
    return {
        "xgboost_trained": runtime.xgb_model.is_trained(),
        "lstm_trained": runtime.lstm_model.is_trained(),
        "horizons": horizons,
    }


@router.get("/api/ollama-status")
def ollama_status():
    """Report local language-model availability and configured model status."""
    available = runtime.explainer.ollama_available()
    models = []
    if available:
        try:
            r = requests.get(f"{settings.OLLAMA_URL}/api/tags", timeout=2)
            models = [m["name"] for m in r.json().get("models", [])]
        except Exception:
            pass
    return {"ollama_available": available, "loaded_models": models}


@router.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Resolve the browser's default icon request to the application logo."""
    return RedirectResponse(url="/app/favicon.svg", status_code=307)
