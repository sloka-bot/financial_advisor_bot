"""HTTP routes for users."""

import logging

from fastapi import APIRouter, HTTPException

from backend import runtime
from backend.api.schemas import (
    ApprovalRequest,
    ImportPortfolioRequest,
    ProfileRequest,
    ProfileUpdateRequest,
)
from backend.data.contracts import freshness

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
            # No model has classified an imported holding, so its signal/support is
            # unavailable rather than a fabricated HOLD/50.
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
    # Read the pending rec without changing its status: execute first, then mark
    # approved only on success, so a failed trade never looks approved.
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
    """Generate a fresh set of recommendations from the currently trained models
    without re-running the full data pipeline."""
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
