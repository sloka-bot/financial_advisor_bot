"""Shared data validation for training, inference, and execution."""

import math
from datetime import date

import pandas as pd

MAX_PRICE_AGE_DAYS = 5


def finite_number(value, default=None):
    """Return a finite float or the supplied default for invalid input."""
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def execution_status(frame):
    """Explain whether the latest row provides a genuine, tradable execution price.

    A forward-filled (imputed) or suspension-edge latest row does not: its price
    was carried, not observed, so it must never be treated as a tradable print.
    Dataset-level staleness (age) is reported separately by ``freshness``.
    """
    if frame is None or frame.empty:
        return {"available": False, "reason": "No price observations", "price": None}
    row = frame.iloc[-1]
    if int(finite_number(row.get("imputed_flag"), 0) or 0):
        return {"available": False, "reason": "Latest price is forward-filled, not an observed print", "price": None}
    if int(finite_number(row.get("suspension_flag"), 0) or 0):
        return {"available": False, "reason": "Latest row sits at a trading-suspension gap", "price": None}
    for column in ("exec_close", "close_unadj"):
        value = finite_number(row.get(column))
        if value is not None and value > 0:
            return {"available": True, "reason": None, "price": value}
    return {"available": False, "reason": "No positive raw execution price", "price": None}


def executable_price(frame):
    """Return a positive raw execution price, or None. Never substitutes the
    adjusted close, and never returns a forward-filled or suspension-edge price."""
    status = execution_status(frame)
    return status["price"] if status.get("available") else None


def freshness(frame, today=None):
    """Describe whether the latest observation meets the age threshold."""
    if frame is None or frame.empty:
        return {"fresh": False, "as_of": None, "reason": "No price observations"}
    last = pd.Timestamp(frame.index.max()).tz_localize(None).normalize()
    now = pd.Timestamp(today or date.today()).tz_localize(None).normalize()
    age = (now - last).days
    return {
        "fresh": 0 <= age <= MAX_PRICE_AGE_DAYS,
        "as_of": str(last.date()),
        "age_days": age,
        "reason": None if 0 <= age <= MAX_PRICE_AGE_DAYS else "Price data is stale or future-dated",
    }


def json_safe(value):
    """Convert non-finite numeric outputs to unavailable values at the API boundary."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value
