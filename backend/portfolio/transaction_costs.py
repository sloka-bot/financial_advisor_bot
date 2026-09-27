"""
transaction_costs.py

Single source of truth for how a proportional transaction cost is charged
across the system. The Markowitz cost-aware objective, the PPO trading
environment, the walk-forward trading backtest and the single-name backtester
all compute turnover the same way and charge the same rate here, so a
strategy's reported net return rests on one consistent cost model instead of
several slightly different ones.

Cost model: a proportional cost of ``rate`` is charged on turnover, where
turnover is the sum of absolute changes in portfolio weight,

    turnover = sum_i |w_i - w_prev_i|

and the charge is ``rate * turnover``. ``rate`` defaults to ``TX_COST`` (10 bps
per unit of weight turnover).
"""

from collections.abc import Mapping

import numpy as np

from backend.config.settings import TX_COST


def turnover(prev_weights, new_weights) -> float:
    """Return sum_i |w_i - w_prev_i| between two weight books.

    Accepts either aligned numeric sequences/arrays or dicts keyed by asset.
    Dicts are compared over the union of their keys (a missing key counts as 0).
    """
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
