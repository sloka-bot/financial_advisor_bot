"""Resolve current S&P 500 constituents from historical snapshots, cache or an offline fallback."""

import json
import logging
from pathlib import Path

import pandas as pd

from backend.universe.sp500_membership import SP500Membership, normalize

logger = logging.getLogger(__name__)

UNIVERSE_NAME = "S&P 500"
UNIVERSE_MARKET = "United States"

# Offline fallback list of large, liquid names.
_STATIC_FALLBACK = [
    "AAPL",
    "MSFT",
    "AMZN",
    "NVDA",
    "GOOGL",
    "META",
    "BRK-B",
    "LLY",
    "AVGO",
    "JPM",
    "XOM",
    "UNH",
    "V",
    "PG",
    "MA",
    "HD",
    "COST",
    "JNJ",
    "MRK",
    "ABBV",
    "CVX",
    "PEP",
    "KO",
    "ADBE",
    "WMT",
    "CRM",
    "BAC",
    "MCD",
    "ACN",
    "AMD",
]


class UniverseBuilder:
    """Resolve the supported stock universe and point-in-time eligibility."""

    def __init__(self, data_dir="data/universe"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._membership = SP500Membership(data_dir)
        self._cache = self.data_dir / "United_States__S_P_500.json"

    # Public API.
    def members(self, force_refresh=False, as_of=None):
        """Current S&P 500 constituents (normalized to Yahoo tickers)."""
        # Point-in-time membership as of today.
        try:
            m = self._membership.members_on(
                as_of or pd.Timestamp.today(), allow_fetch=force_refresh or not self._cache.exists()
            )
            if m:
                tickers = sorted(normalize(t) for t in m)
                self._save_cache(tickers)
                return tickers
        except Exception as e:
            logger.warning(f"Point-in-time S&P 500 membership unavailable ({e}); using cache")

        # Cached list on disk.
        if self._cache.exists():
            try:
                return json.loads(self._cache.read_text())["tickers"]
            except Exception:
                pass

        # Static fallback.
        logger.warning("Using static S&P 500 fallback list (30 names)")
        return list(_STATIC_FALLBACK)

    def eligible_between(self, start, end):
        """Point-in-time membership across a window (for historical studies)."""
        return self._membership.eligible_between(start, end)

    def filter_eligible_rows(self, panel, ticker_col="ticker", strict=False):
        """Filter ticker-date rows by historical membership; strict mode rejects unavailable history."""
        import numpy as np

        try:
            snap = self._membership.snapshots(allow_fetch=False)
        except Exception as e:  # loading the history can fail
            snap = None
            logger.warning(f"membership snapshot load failed ({e})")
        if snap is None or snap.empty or ticker_col not in getattr(panel, "columns", []):
            msg = "point-in-time membership history unavailable"
            if strict:
                raise ValueError(f"{msg}; strict membership filtering requires it")
            logger.warning(f"{msg}; using unfiltered panel")
            return panel
        idx = pd.DatetimeIndex(panel.index)
        snap_dates = np.asarray(snap.index.values)
        members = list(snap["members"].values)
        # Each row uses the latest snapshot on or before its date.
        pos = np.searchsorted(snap_dates, idx.values, side="right") - 1
        tick = panel[ticker_col].astype(str).map(normalize).values
        keep = np.fromiter(
            ((p >= 0) and (tick[i] in members[p]) for i, p in enumerate(pos)), dtype=bool, count=len(panel)
        )
        n_keep = int(keep.sum())
        if n_keep < 0.2 * len(panel):
            msg = (
                f"point-in-time row filter would keep only {n_keep}/{len(panel)} rows "
                "(>80% dropped) - membership data likely mismatched"
            )
            if strict:
                raise ValueError(msg + "; stopping in strict mode")
            logger.warning(msg + "; using unfiltered panel")
            return panel
        logger.info(
            f"Point-in-time eligibility: kept {n_keep}/{len(panel)} rows "
            f"(dropped {len(panel) - n_keep} out-of-membership rows)"
        )
        return panel[keep]

    def _save_cache(self, tickers):
        try:
            self._cache.write_text(
                json.dumps(
                    {"market": UNIVERSE_MARKET, "index": UNIVERSE_NAME, "count": len(tickers), "tickers": tickers},
                    indent=2,
                )
            )
        except Exception:
            pass

    # Market and index helpers for the API layer.

    def list_markets(self):
        """Return the single supported market and index for the interface."""
        return {UNIVERSE_MARKET: [UNIVERSE_NAME]}
