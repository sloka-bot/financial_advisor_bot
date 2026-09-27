"""HTTP routes for portfolios."""

import logging

from fastapi import APIRouter, BackgroundTasks, HTTPException

from backend import runtime
from backend.api.schemas import (
    PortfolioRequest,
    RunRequest,
)
from backend.data.contracts import finite_number
from backend.portfolio.allocation import build_markowitz_portfolio
from backend.prediction.recommender import classify_signal

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/api/advisor/run/{user_id}")
def advisor_run(user_id: str, bg: BackgroundTasks):
    """Trigger a pipeline run using the user's stored risk profile."""
    profile = runtime.user_store.get(user_id)
    if not profile:
        raise HTTPException(404, "User profile not found - complete onboarding first")
    req = RunRequest(
        risk_profile=profile.get("risk_profile", "moderate"),
        budget=profile.get("budget", 10000),
        user_id=user_id,
    )
    runtime.start_pipeline(req)
    return {"message": "Analysis started using your stored profile"}


@router.post("/api/recommend")
def get_recommendations_api(req: PortfolioRequest):
    """Generate and explain risk-profile signals from available model outputs."""
    risk_profile = req.risk_profile
    top_n = req.top_n
    if not runtime.xgb_model.is_trained() and not runtime.lstm_model.is_trained():
        raise HTTPException(400, "Models not trained - run the pipeline first")
    with runtime.state_lock:
        processed = list(runtime.pipeline_state.get("processed_tickers", []))
    if not processed:
        raise HTTPException(422, "No processed tickers - run the pipeline first")
    predictions, master_data = runtime.collect_predictions(processed)
    if not predictions:
        raise HTTPException(422, "No predictions generated")
    ranked = runtime.ranker.rank(predictions, master_data, risk_profile)
    result = runtime.recommender.recommend(ranked, risk_profile=risk_profile, top_n=top_n)
    for signal in result.get("all_signals", []):
        df = master_data.get(signal["ticker"])
        _pp = finite_number(signal.get("prob_up_pct"))
        confidence = runtime.portfolio_manager.confidence_score(
            signal["ticker"], df, (finite_number(signal.get("predicted_return"), 0)) / 100,
            prob_up=(_pp / 100 if _pp is not None else None),
        )
        signal["confidence"] = confidence["overall"]
        signal["confidence_factors"] = confidence["factors"]
    runtime.explainer.explain_batch(result.get("all_signals", []), master_data)
    return {"universe": "S&P 500", **result}


@router.post("/api/portfolio")
def get_portfolio(req: PortfolioRequest):
    """Return the saved portfolio, proposed allocation and supporting signals."""
    risk_profile = req.risk_profile
    budget = req.budget
    top_n = req.top_n
    # For a user-specific request the saved profile is the source of truth, so a
    # conservative user's book is never analysed with an aggressive request profile.
    if req.user_id and (saved := runtime.user_store.get(req.user_id)):
        risk_profile = saved.get("risk_profile", risk_profile)
        budget = saved.get("budget", budget) or budget
    if not runtime.xgb_model.is_trained() and not runtime.lstm_model.is_trained():
        raise HTTPException(400, "Models not trained")
    with runtime.state_lock:
        processed = list(runtime.pipeline_state.get("processed_tickers", []))
    if not processed:
        raise HTTPException(422, "No processed tickers")
    predictions, master_data = runtime.collect_predictions(processed)
    if not predictions:
        raise HTTPException(422, "No predictions")
    ranked = runtime.ranker.rank(predictions, master_data, risk_profile)
    result = runtime.recommender.recommend(ranked, risk_profile=risk_profile, top_n=top_n)
    # Historical returns determine allocation; predictions are display metadata.
    ml_meta = (
        {
            r["ticker"]: {
                "predicted_return": r.get("predicted_return"),
                "sentiment": r.get("sentiment"),
                "composite_score": r.get("composite_score"),
                "signal": classify_signal(r.get("prob_up"), r.get("expected_return"), risk_profile),
            }  # the TRUE ML classification, kept separate from the allocation action
            for r in ranked.to_dict("records")
        }
        if hasattr(ranked, "to_dict")
        else {}
    )
    portfolio = build_markowitz_portfolio(master_data, processed, budget, risk_profile, ml_meta=ml_meta)
    proposed = portfolio["portfolio"]
    if req.user_id and (profile := runtime.user_store.get(req.user_id)):
        portfolio["portfolio"] = runtime.actual_portfolio(profile)
    holdings = portfolio.get("portfolio", {}).get("holdings", [])
    for h in holdings:
        df = master_data.get(h["ticker"])
        # expected_return_decimal is already a decimal (or None); confidence_score
        # expects a decimal, so it is passed through without dividing by 100.
        c = runtime.portfolio_manager.confidence_score(h["ticker"], df, h.get("expected_return_decimal") or 0.0)
        metadata = dict(ml_meta.get(h["ticker"], {}))
        if "predicted_return" in metadata:
            value = finite_number(metadata["predicted_return"])
            metadata["predicted_return"] = value * 100 if value is not None else None
        h.update(metadata)
        h["confidence"] = c["overall"]
        h["confidence_factors"] = c["factors"]
    # drift suggestions on the whole-portfolio value (incl. cash) and the SAME profile
    port_cash = portfolio.get("portfolio", {}).get("cash_remaining", 0.0)
    drift_suggestions = (
        runtime.portfolio_manager.suggest_rebalance(holdings, master_data, risk_profile=risk_profile, cash=port_cash)
        if holdings
        else []
    )
    runtime.explainer.explain_batch(result.get("all_signals", []), master_data)
    regime = runtime.regime_detector.detect(master_data)
    return {
        "universe": "S&P 500",
        "risk_profile": risk_profile,
        "budget": budget,
        **portfolio,
        "proposed_portfolio": proposed,
        "signals": result.get("summary", {}),
        "all_signals": result.get("all_signals", []),
        "top_picks": result.get("top_picks", []),
        "drift_suggestions": drift_suggestions,
        "market_regime": regime,
    }


@router.post("/api/risk-preview")
def risk_preview(req: PortfolioRequest):
    """Compare risk tiers using the same historical allocation as the portfolio view."""
    from backend.config.settings import RISK_CONSTRAINTS

    profile = req.risk_profile
    cons = RISK_CONSTRAINTS[profile]
    base = {
        "risk_profile": profile,
        "constraints": {
            "max_position_pct": cons["max_weight"] * 100,
            "min_cash_pct": cons["min_cash"] * 100,
            "risk_aversion": cons["risk_aversion"],
        },
    }
    runtime._reject_if_training()
    with runtime.state_lock:
        processed = list(runtime.pipeline_state.get("processed_tickers", []))
    if not processed:
        return {**base, "trained": False, "note": "Run analysis to prepare price history."}
    master_data = {ticker: runtime.fusion.load_master(ticker) for ticker in processed}
    portfolio = build_markowitz_portfolio(master_data, processed, req.budget, profile)["portfolio"]
    if not portfolio["holdings"]:
        return {**base, "trained": False, "note": "No holdings fit the available data and budget."}
    metrics = portfolio["risk_metrics"]
    return {
        **base,
        "trained": True,
        "expected_return_pct": portfolio["expected_portfolio_return"],
        "volatility_pct": round(metrics["portfolio_volatility"] * 100, 2),
        "sharpe": metrics["annualized_sharpe"],
        "var_95_1day_pct": round(metrics["var_95_1day"] * 100, 2),
        "cash_weight_pct": portfolio["cash_weight_pct"],
        "n_positions": portfolio["n_positions"],
        "holdings": portfolio["holdings"],
    }
