"""
sp500_membership.py

Point-in-time S&P 500 membership: using today's constituents throughout history
would introduce survivorship bias, so membership is resolved per date.

The class answers one question: *which tickers were in the S&P 500 on a given
date?* It reads a snapshot history in the format published by the widely used
fja05680/sp500 dataset:

    date,tickers
    2010-01-04,"A,AA,AAPL,ABC,..."
    2010-01-29,"A,AAPL,ABC,..."
    ...

Each row is the full constituent list as of that date; membership on any query
date is the most recent snapshot at or before it. From this we derive, for each
ticker, its eligibility window (first_seen / last_seen), which is written to an
audit file so every included security has a documented identity and eligibility
date.

Data source resolution order:
  1. a local CSV at `data/universe/sp500_history.csv` (preferred, reproducible);
  2. otherwise fetch from `SP500_HISTORY_URL` (env var) or the bundled default
     and cache it locally;
  3. if neither works, `degraded_current()` returns today's constituents with a
     loud survivorship-bias warning recorded in the audit - never silently.
"""

import logging
import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# fja05680/sp500 maintains a point-in-time components history (MIT licensed).
# Its filename carries a date stamp that changes over time, so rather than pin a
# brittle URL we resolve the current file via the GitHub contents API. An
# explicit SP500_HISTORY_URL env var overrides this.
ENV_HISTORY_URL = os.environ.get("SP500_HISTORY_URL")
GH_CONTENTS_API = "https://api.github.com/repos/fja05680/sp500/contents/"
FILE_PATTERN = re.compile(r"historical components.*changes.*\.csv$", re.I)


def _resolve_history_url():
    """Return a downloadable URL for the components-history CSV, or None."""
    if ENV_HISTORY_URL:
        return ENV_HISTORY_URL
    try:
        import requests

        r = requests.get(GH_CONTENTS_API, timeout=20, headers={"Accept": "application/vnd.github+json"})
        r.raise_for_status()
        candidates = [f for f in r.json() if FILE_PATTERN.search(f.get("name", ""))]
        if not candidates:
            return None
        # newest by name (the date stamp sorts chronologically enough for latest)
        candidates.sort(key=lambda f: ("updated" in f["name"].lower(), f["name"]))
        return candidates[-1].get("download_url")
    except Exception as e:
        logger.warning(f"Could not resolve S&P 500 history URL: {e}")
        return None


def normalize(ticker: str) -> str:
    """Match Yahoo Finance conventions: class shares use '-' not '.' (BRK.B -> BRK-B)."""
    return str(ticker).strip().upper().replace(".", "-")


class SP500Membership:
    """Load historical constituent snapshots and resolve date-specific membership."""

    def __init__(self, data_dir="data/universe"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.local_path = self.data_dir / "sp500_history.csv"
        self._snap = None  # cached snapshot frame: index=date, col 'members' (set)

    # ------------------------------------------------------------------ #
    def snapshots(self, allow_fetch=True) -> pd.DataFrame:
        """Return a DataFrame indexed by snapshot date with a 'members' column
        holding the frozenset of normalized tickers as of that date."""
        if self._snap is not None:
            return self._snap

        raw = None
        # a cached local file must be non-empty to count as usable
        if self.local_path.exists() and self.local_path.stat().st_size > 0:
            logger.info(f"Loading S&P 500 history from {self.local_path.name}")
            raw = pd.read_csv(self.local_path)
        elif allow_fetch:
            url = _resolve_history_url()
            if not url:
                raise RuntimeError(
                    "Could not resolve an S&P 500 membership history source. "
                    f"Place a snapshot CSV at {self.local_path} (columns: date,tickers), "
                    "or set SP500_HISTORY_URL."
                )
            try:
                logger.info(f"Fetching S&P 500 history from {url}")
                raw = pd.read_csv(url)
                raw.to_csv(self.local_path, index=False)  # cache for reproducibility
                logger.info(f"Cached to {self.local_path} ({len(raw)} snapshots)")
            except Exception as e:
                raise RuntimeError(
                    f"Could not load S&P 500 membership history ({e}). "
                    f"Place a snapshot CSV at {self.local_path} (columns: date,tickers)."
                )
        else:
            raise RuntimeError(f"No S&P 500 history at {self.local_path} and fetch disabled.")

        cols = {c.lower(): c for c in raw.columns}
        date_col = cols.get("date")
        tick_col = cols.get("tickers") or cols.get("ticker")
        if not date_col or not tick_col:
            raise ValueError(f"Unexpected columns {list(raw.columns)}; need date,tickers")

        raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
        raw = raw.dropna(subset=[date_col]).sort_values(date_col)
        members = raw[tick_col].apply(lambda s: frozenset(normalize(t) for t in str(s).split(",") if t.strip()))
        self._snap = pd.DataFrame({"members": members.values}, index=raw[date_col].values)
        self._snap.index = pd.DatetimeIndex(self._snap.index)
        return self._snap

    def members_on(self, date, allow_fetch=True) -> frozenset:
        """The constituent set as of `date` (most recent snapshot at or before)."""
        snap = self.snapshots(allow_fetch=allow_fetch)
        d = pd.Timestamp(date)
        if d > snap.index.max() + pd.Timedelta(days=90):
            raise ValueError(f"Membership history ends {snap.index.max().date()}; refresh it before using {d.date()}")
        prior = snap.index[snap.index <= d]
        if len(prior) == 0:
            return frozenset()
        return snap.loc[prior.max(), "members"]

    def was_member(self, ticker, date, allow_fetch=True) -> bool:
        """Check whether a normalised ticker belonged to the index on a date."""
        return normalize(ticker) in self.members_on(date, allow_fetch=allow_fetch)

    def eligibility_table(self, allow_fetch=True) -> pd.DataFrame:
        """Per-ticker first_seen / last_seen / still_member from the snapshots.

        Written to `sp500_membership_audit.csv` by `write_audit`.
        """
        snap = self.snapshots(allow_fetch=allow_fetch)
        first, last = {}, {}
        for d, row in snap.iterrows():
            for t in row["members"]:
                first.setdefault(t, d)
                last[t] = d
        last_snap_date = snap.index.max()
        rows = [
            {
                "ticker": t,
                "first_seen": str(pd.Timestamp(first[t]).date()),
                "last_seen": str(pd.Timestamp(last[t]).date()),
                "still_member": bool(last[t] == last_snap_date),
            }
            for t in sorted(first)
        ]
        return pd.DataFrame(rows)

    def write_audit(self, allow_fetch=True) -> Path:
        """Persist the constituent eligibility table used to constrain experiments."""
        tbl = self.eligibility_table(allow_fetch=allow_fetch)
        out = self.data_dir / "sp500_membership_audit.csv"
        tbl.to_csv(out, index=False)
        logger.info(f"S&P 500 membership audit written to {out} ({len(tbl)} securities)")
        return out

    def eligible_between(self, start, end, allow_fetch=True) -> list:
        """Every ticker that was a member at any point within [start, end].

        This is the correct universe for a historical backtest window: a stock
        that left the index mid-window still belongs in the study for the period
        it was a member, and one that joined mid-window is included from then.
        """
        snap = self.snapshots(allow_fetch=allow_fetch)
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        if e > snap.index.max() + pd.Timedelta(days=90):
            raise ValueError(f"Membership history ends {snap.index.max().date()}; requested end is {e.date()}")
        window = snap[(snap.index >= s) & (snap.index <= e)]
        # include the snapshot in force at `start` too
        prior = snap.index[snap.index <= s]
        seen = set()
        if len(prior):
            seen |= set(snap.loc[prior.max(), "members"])
        for _, row in window.iterrows():
            seen |= set(row["members"])
        return sorted(seen)

    # ------------------------------------------------------------------ #
    def degraded_current(self, tickers) -> dict:
        """Fallback: treat a supplied current constituent list as the universe.

        Returns an audit dict that RECORDS the survivorship bias rather than
        hiding it, so a run that had to fall back is never mistaken for a
        point-in-time study.
        """
        logger.warning("Using CURRENT constituents - results carry survivorship bias.")
        return {
            "mode": "degraded_current_constituents",
            "survivorship_bias": True,
            "warning": "No point-in-time history available; today's members applied "
            "to all history. Prefer sp500_history.csv for a clean study.",
            "n_tickers": len(tickers),
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }
