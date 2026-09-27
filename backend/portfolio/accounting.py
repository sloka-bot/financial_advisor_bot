"""
accounting.py

One definition of the portfolio value identity, shared by the allocation and
backtest paths so the same reconciliation rule is applied everywhere:

    budget = invested_notional + fees + cash_remaining

A ledger that does not satisfy this within CASH_TOL is a real accounting fault,
not a rounding artefact, and callers can choose to fail loudly (strict) rather
than continue on numbers that do not add up.
"""

from backend.config.settings import CASH_TOL


class AccountingError(RuntimeError):
    """Raised when a portfolio ledger fails to reconcile in strict mode."""


def reconcile_allocation(notional, fees, cash_remaining, budget, weight_sum_pct, cash_tol=CASH_TOL):
    """Return (ok, detail) for the budget = notional + fees + cash identity.

    ok is True when cash is non-negative, the identity holds within cash_tol and
    invested weights do not exceed 100%. detail carries the residual and parts for
    logging or saving on failure.
    """
    residual = notional + fees + cash_remaining - budget
    ok = (
        cash_remaining >= -cash_tol
        and abs(residual) < cash_tol
        and weight_sum_pct <= 100.0 + cash_tol
    )
    detail = {
        "identity_residual": round(residual, 6),
        "cash_remaining": round(cash_remaining, 6),
        "invested_notional": round(notional, 6),
        "fees": round(fees, 6),
        "weight_sum_pct": weight_sum_pct,
        "budget": round(budget, 6),
    }
    return bool(ok), detail
