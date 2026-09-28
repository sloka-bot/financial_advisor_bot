"""Valuation of saved holdings, separate from proposed allocations."""

import pandas as pd

from backend.config import settings
from backend.data.contracts import executable_price, freshness


def value_saved_portfolio(profile, fusion):
    """Value the user's saved holdings; proposals are excluded."""
    import numpy as np

    from backend.portfolio.markowitz import MarkowitzOptimizer

    saved = profile.get("portfolio", {})
    cash = float(saved.get("cash", 0))
    holdings, returns = [], {}
    for original in saved.get("holdings", []):
        holding = dict(original)
        frame = fusion.load_master(holding["ticker"])
        current = executable_price(frame)
        holding["price"] = current or float(holding["price"])
        holding["current_value"] = holding["shares"] * holding["price"]
        holding["data_status"] = freshness(frame)
        holding["valuation_estimated"] = current is None or not holding["data_status"]["fresh"]
        holding["total_cost"] = original.get("total_cost", original["shares"] * original["price"])
        holdings.append(holding)
        if frame is not None and len(frame) >= 60:
            returns[holding["ticker"]] = frame["close"].pct_change().tail(252)
    invested = sum(h["current_value"] for h in holdings)
    total = invested + cash
    for holding in holdings:
        holding["weight_pct"] = round(100 * holding["current_value"] / total, 2) if total else 0
    metrics = {"annualized_sharpe": None, "var_95_1day": None, "cvar_95_1day": None}
    annual_return = None
    if holdings and len(returns) == len(holdings):
        matrix = pd.DataFrame(returns)
        weights = np.array([next(h["current_value"] / total for h in holdings if h["ticker"] == t) for t in matrix])
        mu = matrix.mean().to_numpy() * 252
        cov = MarkowitzOptimizer().covariance(matrix)
        annual_return = float(weights @ mu)
        vol = float(np.sqrt(max(0, weights @ cov @ weights)))
        daily_mean, daily_vol = annual_return / 252, vol / np.sqrt(252)
        metrics = {
            "annualized_sharpe": (annual_return - settings.RF_ANNUAL) / vol if vol > 1e-9 else None,
            "var_95_1day": max(0, 1.6448536 * daily_vol - daily_mean),
            "cvar_95_1day": max(0, 2.0627128 * daily_vol - daily_mean),
        }
    return {
        "holdings": holdings,
        "cash": cash,
        "cash_remaining": cash,
        "total_invested": invested,
        "total_value": total,
        "n_positions": len(holdings),
        "risk_profile": profile.get("risk_profile", "moderate"),
        "expected_portfolio_return": annual_return * 100 if annual_return is not None else None,
        "risk_metrics": metrics,
        "stats_basis": "saved_holdings_including_cash",
    }
