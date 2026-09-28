"""Shared return, risk and drawdown metrics for portfolio evaluations."""

import numpy as np

from backend.config.settings import RF_ANNUAL, TRADING_DAYS
from backend.portfolio.transaction_costs import transaction_cost, turnover

__all__ = [
    "equity_curve",
    "total_return",
    "max_drawdown",
    "annualized_sharpe",
    "periods_per_year",
    "transaction_cost",
    "turnover",
    "RF_ANNUAL",
    "TRADING_DAYS",
]


def periods_per_year(horizon: int) -> float:
    """Number of holding periods per year for a horizon of `horizon` sessions."""
    return TRADING_DAYS / horizon


def equity_curve(returns, initial: float = 1.0) -> np.ndarray:
    """Compounded equity curve including the initial point, from period returns."""
    r = np.asarray(returns, dtype=float)
    return float(initial) * np.cumprod(np.concatenate([[1.0], 1.0 + r]))


def total_return(curve, capital: float = 1.0) -> float:
    """Total return of an equity curve relative to starting capital."""
    c = np.asarray(curve, dtype=float)
    return float(c[-1] / capital - 1.0)


def max_drawdown(curve, eps: float = 0.0) -> float:
    """Worst peak-to-trough drawdown of an equity curve, as a negative fraction."""
    c = np.asarray(curve, dtype=float)
    if len(c) == 0:
        return 0.0
    roll_max = np.maximum.accumulate(c)
    return float(((c - roll_max) / (roll_max + eps)).min())


def annualized_sharpe(
    returns, ppy: float, rf_annual: float = RF_ANNUAL, eps: float = 1e-9, min_periods: int = 2
) -> float:
    """Annualise excess-return Sharpe; return zero for fewer than min_periods observations."""
    r = np.asarray(returns, dtype=float)
    if len(r) < min_periods:
        return 0.0
    return float((r.mean() - rf_annual / ppy) / (r.std() + eps) * np.sqrt(ppy))
