"""Reconcile portfolio budget against invested notional, fees and remaining cash."""

from backend.config.settings import CASH_TOL


class AccountingError(RuntimeError):
    """Raised when a portfolio ledger fails to reconcile in strict mode."""


def reconcile_allocation(notional, fees, cash_remaining, budget, weight_sum_pct, cash_tol=CASH_TOL):
    """Check budget reconciliation, non-negative cash and total invested weight."""
    residual = notional + fees + cash_remaining - budget
    ok = cash_remaining >= -cash_tol and abs(residual) < cash_tol and weight_sum_pct <= 100.0 + cash_tol
    detail = {
        "identity_residual": round(residual, 6),
        "cash_remaining": round(cash_remaining, 6),
        "invested_notional": round(notional, 6),
        "fees": round(fees, 6),
        "weight_sum_pct": weight_sum_pct,
        "budget": round(budget, 6),
    }
    return bool(ok), detail
