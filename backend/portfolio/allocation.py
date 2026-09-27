"""
allocation.py

The single shared portfolio-construction + execution layer (consolidation).

Both the live app and any experimental method build a book through here, so
transaction costs, share rounding, cash accounting and risk-constraint
enforcement are identical everywhere.

Firewall (per the project architecture): the allocation is computed ONLY from
historical market data (returns + covariance) and the risk profile. It does NOT
consume XGBoost / LSTM / HMM output. Model predictions and sentiment may be
attached to a holding as DISPLAY-ONLY metadata (`ml_meta`); they never influence
the weights.

Expected returns  : annualised historical mean (trailing `lookback` sessions).
Covariance        : Ledoit-Wolf shrinkage via MarkowitzOptimizer.covariance.
Optimiser         : constrained mean-variance (per-profile max_weight / min_cash
                    / risk-aversion); cost-aware when current holdings are given.
Execution         : whole-share rounding + fees that reconcile to the budget.
"""

import logging

import numpy as np
import pandas as pd

from backend.config.settings import RF_ANNUAL, TX_COST
from backend.data.contracts import executable_price
from backend.portfolio.markowitz import MarkowitzOptimizer, validate_allocation

logger = logging.getLogger(__name__)

DEFAULT_LOOKBACK = 252  # ~1 trading year for return/cov estimation
MAX_CANDIDATES = 40  # cap universe for a stable covariance matrix
MIN_HISTORY = 60  # a name needs at least this many closes to be eligible


def _returns_frame(master_data: dict, tickers: list, lookback: int) -> pd.DataFrame:
    """Daily-return matrix ALIGNED ON DATES (not row position).

    Covariance is only meaningful when it compares returns from the same dates,
    so each ticker's returns keep their DatetimeIndex and are joined on it; the
    common overlapping window is then taken, so returns are aligned by date rather
    than by position.
    """
    series = {}
    for t in tickers:
        df = master_data.get(t)
        if df is None or getattr(df, "empty", True) or "close" not in df.columns:
            continue
        s = pd.to_numeric(df["close"], errors="coerce")
        if not isinstance(s.index, pd.DatetimeIndex):
            try:
                s.index = pd.to_datetime(s.index)
            except Exception:
                continue
        r = s.sort_index().pct_change()
        if r.notna().sum() >= MIN_HISTORY:
            series[t] = r
    if not series:
        return pd.DataFrame()
    # outer-join on the date index, take the recent window, then keep only dates
    # where every retained name has a return (a genuine common history).
    frame = pd.DataFrame(series).sort_index().tail(lookback)
    frame = frame.dropna(axis=1, thresh=MIN_HISTORY)  # drop names too sparse in the window
    frame = frame.dropna(axis=0, how="any")  # common dates across all names
    return frame


def _liquidity_screen(master_data: dict, tickers: list, lookback: int, keep: int) -> list:
    """Non-predictive universe screen: keep the most liquid names (avg dollar
    volume). Purely mechanical - uses no model output - so the firewall holds.
    Falls back to history length when volume is unavailable."""
    scored = []
    for t in tickers:
        df = master_data.get(t)
        if df is None or getattr(df, "empty", True) or "close" not in df.columns:
            continue
        tail = df.tail(lookback)
        if len(tail) < MIN_HISTORY:
            continue
        if "volume" in tail.columns:
            score = float((tail["close"] * tail["volume"]).mean())
        else:
            score = float(len(tail))  # no volume -> prefer longer history
        scored.append((score, t))
    scored.sort(reverse=True)
    return [t for _, t in scored[:keep]]


def build_markowitz_portfolio(
    master_data: dict,
    candidate_tickers: list,
    budget: float,
    risk_profile: str = "moderate",
    ml_meta: dict = None,
    current_holdings: list = None,
    lookback: int = DEFAULT_LOOKBACK,
    max_candidates: int = MAX_CANDIDATES,
    tx_cost: float = TX_COST,
    strict: bool = False,
) -> dict:
    """Construct a portfolio via the constrained optimiser from HISTORICAL data.

    Returns a {'portfolio': {...}} dict consumed by the API and the frontend. `ml_meta[ticker]` (if
    given) supplies display-only predicted_return / sentiment / composite_score.
    `current_holdings` (list of {ticker, shares}) switches the optimiser into
    cost-aware mode so a rebalance only trades when it is worth the cost.
    """
    mkw = MarkowitzOptimizer(tx_cost=tx_cost)
    ml_meta = ml_meta or {}

    universe = _liquidity_screen(master_data, candidate_tickers, lookback, max_candidates)
    rets = _returns_frame(master_data, universe, lookback)
    tickers = list(rets.columns)
    if not tickers:
        return _empty(budget, risk_profile)

    mu = (rets.mean() * 252).reindex(tickers).values  # annualised historical mean
    cov = mkw.covariance(rets[tickers])
    prices = {t: executable_price(master_data[t]) or 0.0 for t in tickers}

    # cost-aware rebalance if we were told the current book
    current_weights = None
    if current_holdings:
        # Weight each held name over the WHOLE portfolio value (invested + cash) -
        # i.e. the `budget` passed in - so current weights sit on the SAME base as
        # the optimiser's target weights (fractions of the budget, which includes
        # the cash slot). Using invested-only as the base overstated every weight
        # and distorted the turnover/cost penalty. A held name outside the
        # optimiser's universe still consumes budget here, so it is not treated as
        # free cash to buy into.
        base = (
            float(budget)
            if (budget and float(budget) > 0)
            else sum(
                float(h.get("shares", 0)) * prices.get(h["ticker"], 0.0)
                for h in current_holdings
                if h.get("ticker") in prices
            )
        )
        if base > 0:
            current_weights = {
                h["ticker"]: float(h.get("shares", 0)) * prices.get(h["ticker"], 0.0) / base
                for h in current_holdings
                if h.get("ticker") in prices
            }

    res = mkw.optimize(tickers, mu, cov, risk_profile, current_weights=current_weights, strict=strict)
    # An optimiser failure must not be turned into an all-cash "target": that would
    # make suggest_rebalance read every held name as a sell. Surface it as unavailable.
    if not res.get("available", True):
        logger.warning("Portfolio optimisation unavailable: %s", res.get("error"))
        return _unavailable(budget, risk_profile, res.get("error"))
    alloc = mkw.to_shares(res["weights"], prices, budget, allow_fractional=False, fee_rate=tx_cost)

    holdings = []
    for h in alloc["holdings"]:
        t = h["ticker"]
        meta = ml_meta.get(t, {})
        # model expected return (LSTM, over the prediction horizon) is a DECIMAL or
        # None; expose it unambiguously in BOTH units and never mix it with the
        # annualised historical mean (a separate, clearly-named field).
        mdl = meta.get("predicted_return")
        holdings.append(
            {
                "ticker": t,
                # The optimiser CHOSE to allocate to this name from historical risk/return;
                # that is NOT an ML "BUY". Keep the two distinct: allocation_action is the
                # optimiser's decision, signal is the (firewalled) ML classification if any.
                "allocation_action": "ALLOCATE",
                "signal": (meta.get("signal") or "N/A"),
                "shares": h["shares"],
                "price": h["price"],
                "total_cost": h["cost"],
                "fee": h["fee"],
                "weight_pct": h["realised_weight_pct"],  # EXECUTED weight, not target
                # display-only metadata from the prediction leg (never feeds the optimiser)
                "expected_return_decimal": (round(float(mdl), 6) if mdl is not None else None),
                "expected_return_pct": (round(float(mdl) * 100, 4) if mdl is not None else None),
                "predicted_return": (round(float(mdl) * 100, 4) if mdl is not None else None),  # pct alias
                "sentiment": str(meta.get("sentiment", "n/a")),
                "composite_score": meta.get("composite_score"),
                "hist_mean_return_pct": round(float(mu[tickers.index(t)]) * 100, 2),
            }
        )

    # Portfolio statistics recomputed from the EXECUTED whole-share book (+ cash),
    # not the optimiser's pre-rounding target weights. With a small budget the
    # rounded holdings can differ materially from the target, so reporting the
    # target's return/vol/Sharpe would describe a portfolio the user does not hold.
    exec_tickers = [h["ticker"] for h in alloc["holdings"]]
    idx = [tickers.index(t) for t in exec_tickers]
    w_exec = np.array([h["realised_weight_pct"] / 100.0 for h in alloc["holdings"]], dtype=float)
    if len(w_exec):
        mu_e = mu[idx]
        cov_e = np.asarray(cov)[np.ix_(idx, idx)]
        exp_a = float(w_exec @ mu_e)  # annualised expected return
        vol_a = float(np.sqrt(max(float(w_exec @ cov_e @ w_exec), 0.0)))
        sharpe = round((exp_a - RF_ANNUAL) / vol_a, 3) if vol_a > 1e-9 else 0.0
        mu_d, vol_d = exp_a / 252.0, vol_a / np.sqrt(252.0)
        Z, PDF_Z = 1.6448536269514722, 0.10313564037537328  # 95% z and phi(z)
        var95 = round(float(max(0.0, Z * vol_d - mu_d)), 4)
        cvar95 = round(float(max(0.0, vol_d * PDF_Z / 0.05 - mu_d)), 4)
    else:
        exp_a = vol_a = 0.0
        sharpe = var95 = cvar95 = 0.0

    # Held names that fall OUTSIDE the optimiser universe (dropped by the
    # liquidity screen or absent from the returns frame) are not part of the new
    # target and would otherwise disappear silently. Surface them explicitly so the
    # caller can see they are being liquidated rather than quietly forgotten.
    excluded_holdings = []
    if current_holdings:
        uni = set(tickers)
        for h in current_holdings:
            tk = h.get("ticker")
            if tk and tk not in uni:
                dfx = master_data.get(tk)
                px = float(dfx["close"].iloc[-1]) if dfx is not None and not dfx.empty else float(h.get("price", 0.0))
                excluded_holdings.append(
                    {
                        "ticker": tk,
                        "shares": h.get("shares", 0),
                        "value": round(float(h.get("shares", 0)) * px, 2),
                        "reason": "outside optimiser universe (illiquid or insufficient history) - liquidate",
                    }
                )

    report = validate_allocation(alloc, risk_profile)
    return {
        "portfolio": {
            "holdings": holdings,
            "n_positions": len(holdings),
            "total_invested": alloc["invested"],
            "fees": alloc["fees"],
            "cash_remaining": alloc["cash_remaining"],  # real reconciled cash (never clamped)
            "cash_weight_pct": alloc["cash_weight_pct"],
            "risk_profile": risk_profile,
            # executed-book statistics (cash earns 0, so it dilutes the return)
            "expected_portfolio_return": round(exp_a * 100, 4),
            "stats_basis": "executed_holdings_incl_cash",
            "risk_metrics": {
                "annualized_sharpe": sharpe,
                "portfolio_volatility": round(vol_a, 6),
                "var_95_1day": var95,
                "cvar_95_1day": cvar95,
            },
            # the optimiser's pre-rounding target stats, kept for reference only
            "target_stats": {
                "expected_portfolio_return": round(res["expected_return"] * 100, 4),
                "annualized_sharpe": res["sharpe"],
                "portfolio_volatility": res["volatility"],
            },
            "constraints_enforced": report["ok"],
            "reconciles": alloc["reconciles"],
            "method": "markowitz_historical",
            "available": True,
            "covariance_method": mkw.last_covariance_method,
            "covariance_joint_rows": mkw.last_cov_joint_rows,
            "excluded_holdings": excluded_holdings,
        }
    }


def _unavailable(budget: float, risk_profile: str, error: str = None) -> dict:
    """Explicit unavailable result when optimisation fails. Distinct from _empty
    (a legitimate all-cash outcome): callers must NOT treat this as a target, so a
    solver failure never becomes a recommendation to liquidate the book."""
    return {
        "portfolio": {
            "available": False,
            "optimization_error": error or "optimisation failed",
            "holdings": [],
            "n_positions": 0,
            "cash_remaining": round(float(budget), 2),
            "risk_profile": risk_profile,
            "method": "markowitz_historical",
        }
    }


def _empty(budget: float, risk_profile: str) -> dict:
    return {
        "portfolio": {
            "holdings": [],
            "n_positions": 0,
            "total_invested": 0.0,
            "fees": 0.0,
            "cash_remaining": round(float(budget), 2),
            "cash_weight_pct": 100.0,
            "risk_profile": risk_profile,
            "expected_portfolio_return": 0.0,
            "risk_metrics": {
                "annualized_sharpe": 0.0,
                "portfolio_volatility": 0.0,
                "var_95_1day": 0.0,
                "cvar_95_1day": 0.0,
            },
            "constraints_enforced": True,
            "reconciles": True,
            "method": "markowitz_historical",
        }
    }
