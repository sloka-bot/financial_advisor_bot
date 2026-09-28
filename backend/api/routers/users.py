"""HTTP routes for users."""

import logging

from fastapi import APIRouter, HTTPException

from backend import runtime
from backend.api.schemas import (
    ApprovalRequest,
    BuyRequest,
    ImportPortfolioRequest,
    ProfileRequest,
    ProfileUpdateRequest,
    SellRequest,
)
from backend.data.contracts import executable_price, freshness
from backend.portfolio.allocation import build_markowitz_portfolio

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/api/user/{user_id}")
def get_profile(user_id: str):
    """Return a saved user profile or a first-use state."""
    profile = runtime.user_store.get(user_id)
    if not profile:
        return {"exists": False, "is_new": True}
    return {"exists": True, "is_new": False, **profile}


@router.post("/api/user/{user_id}")
def create_profile(user_id: str, req: ProfileRequest):
    """Create a user profile from the validated request fields."""
    profile = runtime.user_store.create_profile(user_id, req.model_dump())
    return {"created": True, **profile}


@router.put("/api/user/{user_id}")
def update_profile(user_id: str, req: ProfileUpdateRequest):
    """Update supplied fields while preserving the remaining profile."""
    profile = runtime.user_store.update_profile(user_id, req.model_dump(exclude_unset=True, exclude_none=True))
    return {"updated": True, **profile}


@router.get("/api/user/{user_id}/correlation")
def holdings_correlation(user_id: str):
    """Calculate recent daily-return correlations for the holdings heatmap."""
    profile = runtime.user_store.get(user_id)
    if not profile:
        raise HTTPException(404, "User not found")
    tickers = [h["ticker"] for h in profile.get("portfolio", {}).get("holdings", [])]
    if len(tickers) < 2:
        return {"tickers": tickers, "matrix": []}
    import pandas as pd

    series = {}
    for t in tickers:
        df = runtime.fusion.load_master(t)
        if df is not None and not df.empty and "daily_return" in df.columns:
            series[t] = df["daily_return"].tail(252)
    valid = [t for t in tickers if t in series]
    if len(valid) < 2:
        return {"tickers": valid, "matrix": []}
    frame = pd.DataFrame({t: series[t] for t in valid}).dropna()
    if len(frame) < 20:
        return {"tickers": valid, "matrix": []}
    corr = frame.corr()
    matrix = [[round(float(corr.loc[a, b]), 2) if pd.notna(corr.loc[a, b]) else None for b in valid] for a in valid]
    return {"tickers": valid, "matrix": matrix}


@router.get("/api/user/{user_id}/portfolio")
def get_user_portfolio(user_id: str):
    """Return saved holdings valued from the available market data."""
    profile = runtime.user_store.get(user_id)
    if not profile:
        raise HTTPException(404, "User not found")
    return runtime.actual_portfolio(profile)


@router.post("/api/user/{user_id}/import-portfolio")
def import_user_portfolio(user_id: str, req: ImportPortfolioRequest):
    """Accept manually entered holdings from the onboarding import flow."""
    profile = runtime.user_store.get(user_id)
    if not profile:
        raise HTTPException(404, "Create a profile before importing holdings")
    holdings = [
        {
            "ticker": h.ticker.upper().strip(),
            "shares": h.shares,
            "price": h.price,
            "total_cost": round(h.shares * h.price, 2),
            # Imported holdings have no model signal until inference is available.
            "signal": "N/A",
            "confidence": None,
            "source": "manual_import",
        }
        for h in req.holdings
        if h.ticker and h.shares > 0 and h.price > 0
    ]
    if len({h["ticker"] for h in holdings}) != len(holdings):
        raise HTTPException(422, "Combine duplicate ticker rows before importing")
    total_cost = sum(h["total_cost"] for h in holdings)
    budget = float(profile.get("budget", 0) or 0)
    # Reject an import that costs more than the stated budget.
    if total_cost > budget + 1e-6:
        raise HTTPException(
            400,
            (
                f"Imported holdings cost ${total_cost:,.2f}, which exceeds your "
                f"stated budget of ${budget:,.2f}. Adjust the shares/prices or raise "
                "your budget before importing."
            ),
        )
    cash = round(budget - total_cost, 2)
    runtime.user_store.update_portfolio(user_id, holdings, cash)
    return {"imported": len(holdings), "holdings": holdings, "cash": cash}


@router.get("/api/user/{user_id}/recommendations")
def get_recommendations(user_id: str):
    """Return the saved recommendation queue for a user."""
    profile = runtime.user_store.get(user_id)
    if not profile:
        return {"pending": [], "history": []}
    return {
        "pending": profile.get("pending_recommendations", []),
        "history": profile.get("recommendation_history", [])[-20:],
    }


@router.post("/api/recommendations/approve")
def approve_recommendation(req: ApprovalRequest):
    """Validate and atomically apply an approved recommendation to holdings."""
    profile = runtime.user_store.get(req.user_id)
    if not profile:
        raise HTTPException(404, "User not found")
    # Read the proposal before execution; approval follows a successful trade.
    rec = runtime.user_store.get_pending_recommendation(req.user_id, req.rec_id)
    if not rec:
        raise HTTPException(404, "Recommendation not found")
    tickers = {rec["ticker"]} | {h["ticker"] for h in profile.get("portfolio", {}).get("holdings", [])}
    master_data = {t: df for t in tickers if (df := runtime.fusion.load_master(t)) is not None}
    if not freshness(master_data.get(rec["ticker"]))["fresh"]:
        raise HTTPException(409, "Refresh market data before approving this recommendation")
    result = runtime.user_store.execute_recommendation(
        req.user_id,
        req.rec_id,
        lambda pending, current: runtime.portfolio_manager.apply_recommendation(
            pending,
            current.get("portfolio", {}).get("holdings", []),
            current.get("portfolio", {}).get("cash", 0),
            master_data,
            risk_profile=current.get("risk_profile", "moderate"),
        ),
    )
    if result is None:
        raise HTTPException(409, "Recommendation was already processed")
    if result.get("error"):
        raise HTTPException(409, result["error"])
    return {"approved": True, "portfolio": runtime.actual_portfolio(runtime.user_store.get(req.user_id))}


@router.post("/api/recommendations/reject")
def reject_recommendation(req: ApprovalRequest):
    """Record rejection of a pending recommendation without trading."""
    rec = runtime.user_store.reject_recommendation(req.user_id, req.rec_id)
    if not rec:
        raise HTTPException(404, "Recommendation not found")
    return {"rejected": True}


@router.post("/api/recommendations/generate/{user_id}")
def generate_new_recommendations(user_id: str):
    """Refresh recommendations using existing models without retraining."""
    profile = runtime.user_store.get(user_id)
    if not profile:
        raise HTTPException(404, "User not found")
    if not runtime.xgb_model.is_trained():
        raise HTTPException(400, "Models not trained - run analysis first")
    with runtime.state_lock:
        tickers = list(runtime.pipeline_state.get("processed_tickers", []))
    if not tickers:
        raise HTTPException(422, "No processed tickers - run analysis first")
    runtime.queue_recommendations(
        user_id,
        tickers,
        profile.get("risk_profile", "moderate"),
        profile.get("budget", 10000),
    )
    pending = runtime.user_store.get(user_id).get("pending_recommendations", [])
    return {"generated": len(pending), "pending": pending}


def _current_market_data(profile, extra=()):
    """Load current execution prices for the whole saved book and requested names."""
    tickers = {h["ticker"] for h in profile.get("portfolio", {}).get("holdings", [])} | set(extra)
    data = {t: runtime.fusion.load_master(t) for t in tickers}
    missing = [t for t, frame in data.items() if not freshness(frame)["fresh"] or executable_price(frame) is None]
    if missing:
        raise HTTPException(409, "Refresh market data before trading: " + ", ".join(sorted(missing)))
    return data


def _manual_trade(user_id, ticker, action, shares=None):
    profile = runtime.user_store.get(user_id)
    if profile is None:
        raise HTTPException(404, "User not found")
    ticker = ticker.upper().strip()
    data = _current_market_data(profile, [ticker])

    def execute(current):
        port = current.get("portfolio", {})
        if ticker not in {h["ticker"] for h in port.get("holdings", [])}:
            return {"error": "This ticker is no longer in your holdings"}
        return runtime.portfolio_manager.apply_recommendation(
            {"ticker": ticker, "action": action, "shares": shares, "sell_all": action == "SELL"},
            port.get("holdings", []),
            port.get("cash", 0),
            data,
            risk_profile=current.get("risk_profile", "moderate"),
        )

    result = runtime.user_store.execute_portfolio_action(user_id, execute, action=action)
    if result is None or result.get("error") or result.get("respects_profile") is False:
        raise HTTPException(409, (result or {}).get("error", "Trade would breach the portfolio limits"))
    return result


@router.post("/api/user/{user_id}/sell")
def sell_user_holding(user_id: str, req: SellRequest):
    """Sell a held position in the local book using observed prices and trading fees."""
    return {"sold": True, **_manual_trade(user_id, req.ticker, "SELL")}


@router.post("/api/user/{user_id}/buy")
def buy_more_holding(user_id: str, req: BuyRequest):
    """Add a requested whole-share quantity to a saved position within risk limits."""
    return {"bought": True, **_manual_trade(user_id, req.ticker, "BUY", req.shares)}


@router.post("/api/user/{user_id}/build")
def build_and_merge_portfolio(user_id: str):
    """Add up to five historical allocations without replacing the existing book."""
    profile = runtime.user_store.get(user_id)
    if profile is None:
        raise HTTPException(404, "User not found")
    with runtime.state_lock:
        processed = list(runtime.pipeline_state.get("processed_tickers", []))
    if not processed:
        raise HTTPException(422, "No processed tickers - run analysis first")
    port = profile.get("portfolio", {})
    cash = float(port.get("cash", 0))
    if cash <= 0:
        raise HTTPException(409, "No available cash for additional holdings")
    _, master_data = runtime.collect_predictions(processed)
    master_data.update(_current_market_data(profile))
    built = build_markowitz_portfolio(
        master_data,
        processed,
        cash,
        profile.get("risk_profile", "moderate"),
        target_positions=5,
    )["portfolio"]
    if not built.get("available", True) or built.get("constraints_enforced") is False:
        raise HTTPException(409, "An allocation within the selected limits is unavailable")

    def execute(current):
        if current.get("portfolio") != port or current.get("risk_profile") != profile.get("risk_profile"):
            return {"error": "Your portfolio changed while building. Please try again."}
        holdings = [dict(h) for h in port.get("holdings", [])]
        balance = cash
        count = 0
        fees = 0.0
        for candidate in built.get("holdings", []):
            ticker = candidate["ticker"]
            if not freshness(master_data.get(ticker))["fresh"]:
                continue
            result = runtime.portfolio_manager.apply_recommendation(
                {"ticker": ticker, "action": "BUY", "max_shares": candidate["shares"]},
                holdings,
                balance,
                master_data,
                risk_profile=current.get("risk_profile", "moderate"),
            )
            if result.get("error") or result.get("respects_profile") is False:
                continue
            holdings, balance = result["holdings"], result["cash"]
            fees += result.get("fees", 0)
            count += 1
        return {"holdings": holdings, "cash": balance, "built": count, "fees": fees, "respects_profile": True}

    result = runtime.user_store.execute_portfolio_action(user_id, execute, action="BUILD")
    if result is None or result.get("error"):
        raise HTTPException(409, (result or {}).get("error", "Portfolio unavailable"))
    return result
