"""Resolve exchange sessions and closing timestamps through a shared calendar."""

import logging
from functools import lru_cache

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_EXCHANGE = "XNYS"


@lru_cache(maxsize=64)
def sessions(exchange, start, end):
    """Return normalised exchange-session dates, or None if unavailable."""
    try:
        import pandas_market_calendars as mcal

        cal = mcal.get_calendar(exchange)
        sched = cal.schedule(start_date=pd.Timestamp(start).date(), end_date=pd.Timestamp(end).date())
        idx = sched.index
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        return pd.DatetimeIndex(idx).normalize()
    except ImportError:
        return None
    except Exception as e:
        logger.warning("Market calendar sessions unavailable (%s)", e)
        return None


@lru_cache(maxsize=64)
def session_closes(year, exchange="NYSE"):
    """Return session closing timestamps for a year and the following January."""
    try:
        import pandas_market_calendars as mcal

        sched = mcal.get_calendar(exchange).schedule(start_date=f"{year}-01-01", end_date=f"{year + 1}-01-10")
        return sched["market_close"]
    except ImportError:
        return None
    except Exception as e:
        logger.warning("Market calendar closes unavailable (%s)", e)
        return None
