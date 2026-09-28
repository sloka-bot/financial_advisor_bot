"""Calculate proportional trading costs from absolute portfolio-weight changes."""

from collections.abc import Mapping

import numpy as np

from backend.config.settings import TX_COST


def turnover(prev_weights, new_weights) -> float:
    """Sum absolute weight changes across aligned arrays or the union of asset keys."""
    if isinstance(prev_weights, Mapping) or isinstance(new_weights, Mapping):
        prev = dict(prev_weights) if isinstance(prev_weights, Mapping) else {}
        new = dict(new_weights) if isinstance(new_weights, Mapping) else {}
        keys = set(prev) | set(new)
        return float(sum(abs(float(new.get(k, 0.0)) - float(prev.get(k, 0.0))) for k in keys))
    prev = np.asarray(list(prev_weights), dtype=float)
    new = np.asarray(list(new_weights), dtype=float)
    return float(np.abs(new - prev).sum())


def transaction_cost(turnover_value: float, rate: float = TX_COST) -> float:
    """Proportional cost charged on a given turnover: ``rate * turnover``."""
    return float(rate) * float(turnover_value)


def trade_cost(prev_weights, new_weights, rate: float = TX_COST) -> float:
    """Transaction cost of moving from ``prev_weights`` to ``new_weights``."""
    return transaction_cost(turnover(prev_weights, new_weights), rate)
