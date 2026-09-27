"""
universe_builder.py

The single source of the tradable universe: the S&P 500 (US), and nothing else.
This project is S&P-500-only by design, so there is no market/index selection -
`members()` always returns the S&P 500 constituents.

Resolution order for the live constituent list:
  1. point-in-time membership as of today (SP500Membership);
  2. the cached S&P 500 list on disk (data/universe/United_States__S_P_500.json);
  3. a small static fallback so the app still starts offline.
"""

import json
import logging
from pathlib import Path

import pandas as pd

from backend.universe.sp500_membership import SP500Membership, normalize

logger = logging.getLogger(__name__)

UNIVERSE_NAME = "S&P 500"
UNIVERSE_MARKET = "United States"

# Minimal offline fallback (large, liquid names) so the UI is never empty.
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

    # -- public API (S&P 500 only) --
    def members(self, force_refresh=False, as_of=None):
        """Current S&P 500 constituents (normalized to Yahoo tickers)."""
        # 1) point-in-time membership as of today
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

        # 2) cached list on disk
        if self._cache.exists():
            try:
                return json.loads(self._cache.read_text())["tickers"]
            except Exception:
                pass

        # 3) static fallback
        logger.warning("Using static S&P 500 fallback list (30 names)")
        return list(_STATIC_FALLBACK)

    def eligible_between(self, start, end):
        """Point-in-time membership across a window (for historical studies)."""
        return self._membership.eligible_between(start, end)

    def filter_eligible_rows(self, panel, ticker_col="ticker", strict=False):
        """Keep only (ticker, date) rows where that ticker was an S&P 500 member ON
        that date (point-in-time), not merely a member at some point in the window.
        eligible_between() gives the right UNIVERSE; a stock should only contribute
        rows for the dates it was actually in the index.

        `strict` controls what happens when the filter cannot be applied faithfully
        (no membership history, or it would drop >80% of the panel):
          * strict=False (live app): keep the unfiltered panel so the product keeps
            working, with a warning.
          * strict=True (research/evaluation): raise, so an experiment stops rather
            than silently producing survivorship-biased results.
        """
        import numpy as np

        try:
            snap = self._membership.snapshots(allow_fetch=False)
        except Exception as e:  # loading the history itself can fail
            snap = None
            logger.warning(f"membership snapshot load failed ({e})")
        if snap is None or snap.empty or ticker_col not in getattr(panel, "columns", []):
            msg = "point-in-time membership history unavailable"
            if strict:
                raise ValueError(f"{msg}; refusing to run a survivorship-biased evaluation")
            logger.warning(f"{msg}; using unfiltered panel")
            return panel
        idx = pd.DatetimeIndex(panel.index)
        snap_dates = np.asarray(snap.index.values)
        members = list(snap["members"].values)
        # for each row date, the active snapshot is the most recent one at/before it
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
                raise ValueError(msg + "; refusing to continue")
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

    # -- market and index helpers used by the API layer --

    def list_markets(self):
        """Return the single supported market and index for the interface."""
        return {UNIVERSE_MARKET: [UNIVERSE_NAME]}
