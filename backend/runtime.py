"""Shared local service instances and pipeline state used by API routers."""

import logging
import threading
from pathlib import Path

from fastapi import HTTPException

from backend.api.schemas import (
    RunRequest,
)
from backend.config import settings
from backend.data.cleaner import DataCleaner
from backend.data.contracts import executable_price, finite_number, freshness
from backend.data.downloader import MarketDataDownloader
from backend.data.feature_engineer import FeatureEngineer
from backend.data.fusion import FeatureFusion
from backend.evaluation.backtester import Backtester
from backend.explain.explainer import Explainer
from backend.infra.drift_monitor import DriftMonitor
from backend.infra.model_registry import ModelRegistry
from backend.news.news_collector import NewsCollector
from backend.news.sentiment_analyzer import SentimentAnalyzer
from backend.portfolio.portfolio_manager import PortfolioManager
from backend.prediction.lstm_model import LSTMForecaster
from backend.prediction.ranker import StockRanker
from backend.prediction.recommender import RecommendationEngine
from backend.prediction.regime_detector import RegimeDetector
from backend.prediction.xgboost_model import XGBoostForecaster
from backend.services.pipeline import run_training_pipeline
from backend.services.portfolio import value_saved_portfolio
from backend.store.user_store import UserStore
from backend.universe.universe_builder import UniverseBuilder

logger = logging.getLogger(__name__)

# Retain shared services and model weights across API requests.
universe_builder = UniverseBuilder()
downloader = MarketDataDownloader()
cleaner = DataCleaner()
engineer = FeatureEngineer()
fusion = FeatureFusion()
user_store = UserStore()
news_collector = NewsCollector()
sentiment = SentimentAnalyzer()
xgb_model = XGBoostForecaster()
lstm_model = LSTMForecaster()

# Load short-horizon artifacts on demand; reuse primary-horizon models.
_horizon_models = {}


def horizon_models(horizon):
    """Return cached horizon forecasters, reusing deployed models for the primary horizon."""
    h = int(horizon)
    if h == settings.PRIMARY_HORIZON:
        return xgb_model, lstm_model
    pair = _horizon_models.get(h)
    if pair is None:
        pair = (
            XGBoostForecaster(models_dir=f"models/xgboost/h{h}", horizon=h),
            LSTMForecaster(models_dir=f"models/lstm/h{h}", horizon=h),
        )
        _horizon_models[h] = pair
    return pair


ranker = StockRanker()
recommender = RecommendationEngine()
explainer = Explainer()
portfolio_manager = PortfolioManager()
backtester = Backtester()
regime_detector = RegimeDetector()
model_registry = ModelRegistry()
drift_monitor = DriftMonitor()

# Protect pipeline progress updates with the shared state lock.
pipeline_state = {
    "step": "idle",
    "message": "",
    "progress": 0,
    "error": None,
    "processed_tickers": sorted(p.stem.removesuffix("_master") for p in Path("data/features").glob("*_master.csv")),
}
state_lock = threading.Lock()
model_lock = threading.RLock()  # serialises training and inference
_launch_lock = threading.Lock()  # one pipeline launch at a time

TRAINING_STEPS = {
    "starting",
    "fetching",
    "downloading",
    "cleaning",
    "features",
    "news",
    "sentiment",
    "fusion",
    "xgboost",
    "lstm",
    "recommendations",
}


# Exclude non-equity proxy instruments from stock recommendations.
NON_EQUITY = {
    "BND",
    "TLT",
    "IEF",
    "SHY",
    "AGG",
    "GOVT",
    "TIPS",
    "IAGG",
    "LQD",
    "HYG",
    "JNK",
    "GLD",
    "SLV",
    "USO",
    "DBO",
    "GSG",
    "PDBC",
    "VIX",
    "UVXY",
    "SVXY",
}


def _pipeline_active():
    with state_lock:
        return pipeline_state.get("step") in TRAINING_STEPS


def _reject_if_training():
    with state_lock:
        step = pipeline_state.get("step")
    if step in TRAINING_STEPS:
        raise HTTPException(409, "Model is training right now - try again in a moment")


def start_pipeline(req):
    """Start one background training thread while keeping request workers available."""
    with _launch_lock:
        if _pipeline_active():
            raise HTTPException(409, "A pipeline run is already in progress")
        update_pipeline("starting", "Starting pipeline", 1)
    threading.Thread(target=pipeline_worker, args=(req,), daemon=True).start()


def _stage_progress(step, message, lo, hi):
    """Map ticker progress into the pipeline stage's displayed percentage range."""

    def _cb(i, n):
        pct = lo + int((hi - lo) * i / max(1, n))
        update_pipeline(step, f"{message} ({i}/{n})", min(hi, pct))

    return _cb


def update_pipeline(step, message, progress=0, error=None, tickers=None):
    """Write the current pipeline step to the shared state dictionary."""
    with state_lock:
        pipeline_state.update({"step": step, "message": message, "progress": progress, "error": error})
        if tickers is not None:
            pipeline_state["processed_tickers"] = tickers


def pipeline_worker(req: RunRequest):
    """Run the training workflow with the shared services and progress callbacks."""
    return run_training_pipeline(
        req,
        universe_builder=universe_builder,
        downloader=downloader,
        cleaner=cleaner,
        engineer=engineer,
        news_collector=news_collector,
        sentiment=sentiment,
        fusion=fusion,
        xgb_model=xgb_model,
        lstm_model=lstm_model,
        model_lock=model_lock,
        update_pipeline=update_pipeline,
        _stage_progress=_stage_progress,
        user_store=user_store,
        queue_recommendations=queue_recommendations,
        drift_monitor=drift_monitor,
    )


def queue_recommendations(user_id, tickers, risk_profile, budget, *, pipeline_internal=False):
    """Store ranked, individually feasible signals for display three at a time."""
    # Restrict live recommendations to current constituents with usable prices.
    try:
        current_members = set(universe_builder.members())
        if current_members:
            tickers = [t for t in tickers if t in current_members]
    except Exception:
        pass
    predictions, master_data = collect_predictions(tickers, pipeline_internal=pipeline_internal)
    logger.info(
        "queue_recommendations[%s]: %d tickers in universe, %d predictions produced",
        user_id,
        len(tickers),
        len(predictions),
    )
    if not predictions:
        user_store.add_recommendations(user_id, [])
        logger.warning(
            "queue_recommendations[%s]: NO predictions - every ticker failed the "
            "freshness / execution-price / model checks",
            user_id,
        )
        return
    ranked = ranker.rank(predictions, master_data, risk_profile)
    result = recommender.recommend(ranked, risk_profile=risk_profile, top_n=3)
    all_signals = result.get("all_signals", [])

    profile = user_store.get(user_id) or {}
    _pf = profile.get("portfolio", {}) or {}
    current_holdings = {h["ticker"] for h in _pf.get("holdings", [])}
    universe_set = set(tickers)
    previous_visible = {r.get("ticker") for r in profile.get("pending_recommendations", [])[:3]}
    holdings = [dict(h) for h in _pf.get("holdings", [])]
    cash = float(_pf.get("cash", budget) or 0.0)
    for held in holdings:
        ticker = held["ticker"]
        if ticker not in master_data:
            frame = fusion.load_master(ticker)
            if frame is not None:
                master_data[ticker] = frame

    recs = []
    seen = set()
    for signal in all_signals:
        ticker = signal["ticker"]
        action = signal["signal"]
        if ticker in seen or ticker in previous_visible or ticker not in universe_set or ticker in NON_EQUITY:
            continue
        if action not in ("BUY", "SELL") or (action == "SELL" and ticker not in current_holdings):
            continue
        simulated = portfolio_manager.apply_recommendation(
            {"action": action, "ticker": ticker, "predicted_return": signal.get("predicted_return")},
            holdings,
            cash,
            master_data,
            risk_profile=risk_profile,
        )
        if simulated.get("error") or simulated.get("respects_profile") is False:
            continue
        seen.add(ticker)
        probability = finite_number(signal.get("prob_up_pct"))
        confidence = portfolio_manager.confidence_score(
            ticker,
            master_data.get(ticker),
            finite_number(signal.get("predicted_return"), 0) / 100,
            prob_up=probability / 100 if probability is not None else None,
        )
        reason = (
            portfolio_manager._buy_reason(signal, confidence)
            if action == "BUY"
            else (f"{ticker} meets the model's SELL thresholds. Review the proposed reduction before approval.")
        )
        recs.append(
            {
                "action": action,
                "ticker": ticker,
                "signal": action,
                "score": signal.get("composite_score"),
                "confidence": confidence["overall"],
                "factors": confidence["factors"],
                "reason": reason,
                "predicted_return": signal.get("predicted_return"),
            }
        )

    user_store.add_recommendations(user_id, recs)
    _sigs = result.get("all_signals", [])
    logger.info(
        "queue_recommendations[%s]: %d signals (%d BUY / %d SELL), %d recommendations queued: %s",
        user_id,
        len(_sigs),
        sum(1 for x in _sigs if x.get("signal") == "BUY"),
        sum(1 for x in _sigs if x.get("signal") == "SELL"),
        len(recs),
        [r["ticker"] for r in recs],
    )

    for r in recs:
        drift_monitor.log_recommendation(
            ticker=r["ticker"],
            action=r["action"],
            predicted_return=r.get("predicted_return"),
            date=next((p.get("as_of") for t, p in predictions.items() if t == r["ticker"]), None),
            model_version=(ModelRegistry().get_best("xgboost") or {}).get("version"),
        )


def collect_predictions(tickers, *, pipeline_internal=False):
    """Return separate horizon probabilities and return estimates with observation metadata."""
    if not pipeline_internal:
        _reject_if_training()
    horizon = settings.PRIMARY_HORIZON
    preds = {}
    master_data = {}
    for ticker in tickers:
        df = fusion.load_master(ticker)
        if df is None or df.empty or not freshness(df)["fresh"] or executable_price(df) is None:
            continue
        master_data[ticker] = df
        with model_lock:
            prob_up = xgb_model.predict_proba_up(df) if xgb_model.is_trained() else None
            exp_ret = lstm_model.predict_ticker(df) if lstm_model.is_trained() else None
        if prob_up is None and exp_ret is None:
            continue
        as_of = str(df.index[-1].date()) if hasattr(df.index[-1], "date") else str(df.index[-1])
        preds[ticker] = {
            "prob_up": float(prob_up) if prob_up is not None else None,
            "expected_return": float(exp_ret) if exp_ret is not None else None,
            "horizon": horizon,
            "as_of": as_of,
            "xgb_available": prob_up is not None,
            "lstm_available": exp_ret is not None,
        }
    return preds, master_data


def actual_portfolio(profile):
    """Value saved holdings using available execution prices and recorded cash."""
    return value_saved_portfolio(profile, fusion)
