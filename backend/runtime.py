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

# Single shared instances of every pipeline component.
# These are module-level so they persist across requests and
# the trained model weights stay in memory between API calls.
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

# Per-horizon "side" forecasters for the 1/5-day predictor. The primary horizon
# uses the deployed models above; short horizons lazily load their own artifacts
# from models/xgboost/h{H} and models/lstm/h{H} (trained by scripts/train_horizons.py).
_horizon_models = {}


def horizon_models(horizon):
    """Return cached (xgb, lstm) forecasters for a horizon; the primary horizon
    reuses the deployed models."""
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

# Shared dictionary written by the background pipeline thread and read
# by the polling endpoint. All writes go through state_lock to prevent
# race conditions between the worker thread and incoming requests.
pipeline_state = {
    "step": "idle",
    "message": "",
    "progress": 0,
    "error": None,
    "processed_tickers": sorted(p.stem.removesuffix("_master") for p in Path("data/features").glob("*_master.csv")),
}
state_lock = threading.Lock()
model_lock = threading.RLock()  # serialises model training vs inference
_launch_lock = threading.Lock()  # ensures only one pipeline launches at a time

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


# Non-equity instruments that should never appear as stock recommendations.
# These can end up in the universe if the data source includes ETF proxies.
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
    """Launch the pipeline in a dedicated daemon thread, ONE run at a time.

    Using a real thread (not Starlette BackgroundTasks) keeps the request
    threadpool free so /api/pipeline-status stays responsive during a long
    run, and the guard prevents two runs mutating the shared models at once.
    """
    with _launch_lock:
        if _pipeline_active():
            raise HTTPException(409, "A pipeline run is already in progress")
        update_pipeline("starting", "Starting pipeline", 1)
    threading.Thread(target=pipeline_worker, args=(req,), daemon=True).start()


def _stage_progress(step, message, lo, hi):
    """Build a progress callback for a per-ticker batch stage. Maps (i, n) into the
    stage's [lo, hi] band and pushes it to the pipeline state so the UI bar advances
    steadily through the stage instead of freezing at its start value."""

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
    """Score all processed tickers and write the top signals to the user's
    pending recommendation queue. SELL signals are only generated for stocks
    the user currently holds. Duplicates are prevented by tracking seen pairs."""
    predictions, master_data = collect_predictions(tickers, pipeline_internal=pipeline_internal)
    if not predictions:
        return
    ranked = ranker.rank(predictions, master_data, risk_profile)
    result = recommender.recommend(ranked, risk_profile=risk_profile, top_n=3)  # show 3 top picks at a time

    profile = user_store.get(user_id) or {}
    _pf = profile.get("portfolio", {}) or {}
    current_holdings = {h["ticker"] for h in _pf.get("holdings", [])}
    universe_set = set(tickers)

    # Simulated running book so we only surface BUYs the user can actually approve
    # under their risk caps (max position weight + cash floor). Each accepted BUY
    # advances the simulated cash/holdings, so the whole batch is jointly feasible.
    sim_holdings = [dict(h) for h in _pf.get("holdings", [])]
    sim_cash = float(_pf.get("cash", budget) or 0.0)

    recs = []
    seen = set()
    for signal in result.get("all_signals", []):
        ticker_sym = signal["ticker"]

        if ticker_sym not in universe_set or ticker_sym in NON_EQUITY:
            continue
        if signal["signal"] == "SELL" and ticker_sym not in current_holdings:
            continue
        if signal["signal"] not in ("BUY", "SELL"):
            continue

        key = (ticker_sym, signal["signal"])
        if key in seen:
            continue
        seen.add(key)

        # Feasibility gate: never queue a BUY the risk caps leave no room for, so the
        # user is not offered a recommendation that fails on approval.
        if signal["signal"] == "BUY":
            sim = portfolio_manager.apply_recommendation(
                {"action": "BUY", "ticker": ticker_sym, "predicted_return": signal.get("predicted_return", 0)},
                sim_holdings,
                sim_cash,
                master_data,
                risk_profile=risk_profile,
            )
            if sim.get("error") or sim.get("respects_profile") is False:
                continue  # no room under caps - skip this unactionable BUY
            sim_holdings = sim.get("holdings", sim_holdings)
            sim_cash = float(sim.get("cash", sim_cash))

        df = master_data.get(ticker_sym)
        _pp = finite_number(signal.get("prob_up_pct"))
        confidence = portfolio_manager.confidence_score(
            ticker_sym,
            df,
            (finite_number(signal.get("predicted_return"), 0)) / 100,
            prob_up=(_pp / 100 if _pp is not None else None),
        )
        reason = (
            portfolio_manager._buy_reason(signal, confidence)
            if signal["signal"] == "BUY"
            else f"{ticker_sym} rated SELL - composite score {signal.get('composite_score', 0):.0f}/100."
        )
        recs.append(
            {
                "action": signal["signal"],
                "ticker": ticker_sym,
                "signal": signal["signal"],
                "score": signal.get("composite_score", 50),
                "confidence": confidence["overall"],
                "factors": confidence["factors"],
                "reason": reason,
                "predicted_return": signal.get("predicted_return", 0),
            }
        )
        if len(recs) >= 3:  # show at most 3 actionable recommendations at a time
            break

    user_store.add_recommendations(user_id, recs)
    logger.info(f"Queued {len(recs)} recommendations for {user_id}")

    for r in recs:
        drift_monitor.log_recommendation(
            ticker=r["ticker"],
            action=r["action"],
            predicted_return=r.get("predicted_return"),
            date=next((p.get("as_of") for t, p in predictions.items() if t == r["ticker"]), None),
            model_version=(ModelRegistry().get_best("xgboost") or {}).get("version"),
        )


def collect_predictions(tickers, *, pipeline_internal=False):
    """Run both predictors over each ticker's master dataset and return their
    outputs as SEPARATE, named fields - never averaged. XGBoost is a direction
    classifier (probability of a rise); LSTM is a return regressor (estimated
    return in decimal). Each entry also carries the horizon, the observation date
    and which models were available, so consumers can never silently treat one
    number as the other.

    Returns (preds, master_data) where preds[ticker] = {
        'prob_up': float|None,          # P(rise) over the horizon, from XGBoost
        'expected_return': float|None,  # estimated horizon return (decimal), LSTM
        'horizon': int, 'as_of': 'YYYY-MM-DD',
        'xgb_available': bool, 'lstm_available': bool }.
    """
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
