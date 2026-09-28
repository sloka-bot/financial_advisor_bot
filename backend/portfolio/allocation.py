"""Allocate from historical returns and risk limits; attach forecasts as display metadata."""

import logging

import numpy as np
import pandas as pd

from backend.config.settings import RF_ANNUAL, TX_COST
from backend.data.contracts import executable_price
from backend.portfolio.markowitz import MarkowitzOptimizer, validate_allocation

logger = logging.getLogger(__name__)

DEFAULT_LOOKBACK = 252  # about one trading year for return and covariance estimates
MAX_CANDIDATES = 40  # universe cap for a stable covariance matrix
MIN_HISTORY = 60  # minimum closes for eligibility


def _returns_frame(master_data: dict, tickers: list, lookback: int) -> pd.DataFrame:
    """Build a daily-return matrix aligned on common observation dates."""
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
    # Keep recent dates with observed returns for every retained ticker.
    frame = pd.DataFrame(series).sort_index().tail(lookback)
    frame = frame.dropna(axis=1, thresh=MIN_HISTORY)  # drop sparse names
    frame = frame.dropna(axis=0, how="any")  # common dates across names
    return frame


def _liquidity_screen(master_data: dict, tickers: list, lookback: int, keep: int) -> list:
    """Select liquid stocks by dollar volume, falling back to available history length."""
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
            score = float(len(tail))  # without volume, prefer longer history
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
    target_positions: int | None = None,
) -> dict:
    """Build historical allocations with optional turnover costs and display-only forecasts."""
    mkw = MarkowitzOptimizer(tx_cost=tx_cost)
    ml_meta = ml_meta or {}

    eligible = list(dict.fromkeys(candidate_tickers))
    if target_positions is not None:
        eligible = [t for t in eligible if executable_price(master_data.get(t)) is not None]
    universe = _liquidity_screen(master_data, eligible, lookback, max_candidates)
    rets = _returns_frame(master_data, universe, lookback)
    tickers = list(rets.columns)
    if not tickers:
        return _empty(budget, risk_profile)

    mu = (rets.mean() * 252).reindex(tickers).values  # annualised historical mean
    cov = mkw.covariance(rets[tickers])
    prices = {t: executable_price(master_data[t]) or 0.0 for t in tickers}

    # Cost-aware rebalance when the current book is known.
    current_weights = None
    if current_holdings:
        # Measure existing weights against total portfolio value, including cash.
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
    # Return unavailable on solver failure instead of an all-cash target.
    if not res.get("available", True):
        logger.warning("Portfolio optimisation unavailable: %s", res.get("error"))
        return _unavailable(budget, risk_profile, res.get("error"))

    # Select unique candidates by historical weights and returns, then re-optimise the subset.
    if target_positions and len(tickers) > target_positions:
        w = res.get("weights", {})
        by_weight = sorted(tickers, key=lambda t: float(w.get(t, 0.0)), reverse=True)
        chosen = [t for t in by_weight if float(w.get(t, 0.0)) > 1e-6][:target_positions]
        if len(chosen) < target_positions:
            by_mu = sorted(tickers, key=lambda t: float(mu[tickers.index(t)]), reverse=True)
            for t in by_mu:
                if t not in chosen:
                    chosen.append(t)
                if len(chosen) >= target_positions:
                    break
        if chosen and set(chosen) != set(tickers):
            idx = [tickers.index(t) for t in chosen]
            sub_mu = np.array([mu[i] for i in idx])
            sub_cov = cov[np.ix_(idx, idx)]
            sub_res = mkw.optimize(
                chosen, sub_mu, sub_cov, risk_profile, current_weights=current_weights, strict=strict
            )
            if sub_res.get("available", True):
                res, tickers, mu, cov = sub_res, chosen, sub_mu, sub_cov
                prices = {t: prices[t] for t in chosen}
    alloc = mkw.to_shares(res["weights"], prices, budget, allow_fractional=False, fee_rate=tx_cost)

    holdings = []
    for h in alloc["holdings"]:
        t = h["ticker"]
        meta = ml_meta.get(t, {})
        # Expose horizon return forecasts in decimal and percentage units.
        mdl = meta.get("predicted_return")
        holdings.append(
            {
                "ticker": t,
                # Keep the historical allocation decision separate from the model's trading signal.
                "allocation_action": "ALLOCATE",
                "signal": (meta.get("signal") or "N/A"),
                "shares": h["shares"],
                "price": h["price"],
                "total_cost": h["cost"],
                "fee": h["fee"],
                "weight_pct": h["realised_weight_pct"],  # executed weight
                # Display-only forecast metadata.
                "expected_return_decimal": (round(float(mdl), 6) if mdl is not None else None),
                "expected_return_pct": (round(float(mdl) * 100, 4) if mdl is not None else None),
                "predicted_return": (round(float(mdl) * 100, 4) if mdl is not None else None),  # percentage alias
                "sentiment": str(meta.get("sentiment", "n/a")),
                "composite_score": meta.get("composite_score"),
                "hist_mean_return_pct": round(float(mu[tickers.index(t)]) * 100, 2),
            }
        )

    # Calculate portfolio statistics from executed shares and remaining cash.
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

    # Report existing holdings excluded from the proposed allocation universe.
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
            "cash_remaining": alloc["cash_remaining"],  # reconciled cash
            "cash_weight_pct": alloc["cash_weight_pct"],
            "risk_profile": risk_profile,
            # Executed-book statistics; cash earns zero.
            "expected_portfolio_return": round(exp_a * 100, 4),
            "stats_basis": "executed_holdings_incl_cash",
            "risk_metrics": {
                "annualized_sharpe": sharpe,
                "portfolio_volatility": round(vol_a, 6),
                "var_95_1day": var95,
                "cvar_95_1day": cvar95,
            },
            # Pre-rounding target statistics for reference.
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
    """Represent optimisation failure separately from a valid all-cash allocation."""
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
