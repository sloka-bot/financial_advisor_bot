"""Resolve historical S&P 500 membership from dated constituent snapshots."""

import logging
import os
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Resolve the dated history file from fja05680/sp500 (MIT), unless a URL is configured.
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
        # newest by name; the date stamp sorts chronologically
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
        self._snap = None  # cached snapshot frame indexed by date

    def snapshots(self, allow_fetch=True) -> pd.DataFrame:
        """Return date-indexed constituent snapshots containing normalised ticker sets."""
        if self._snap is not None:
            return self._snap

        raw = None
        # Use a cached local file only when non-empty.
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
        """Return each ticker's first and last observed membership dates and current status."""
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
        """Return all tickers with membership during the requested interval."""
        snap = self.snapshots(allow_fetch=allow_fetch)
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        if e > snap.index.max() + pd.Timedelta(days=90):
            raise ValueError(f"Membership history ends {snap.index.max().date()}; requested end is {e.date()}")
        window = snap[(snap.index >= s) & (snap.index <= e)]
        # Include the snapshot in force at start.
        prior = snap.index[snap.index <= s]
        seen = set()
        if len(prior):
            seen |= set(snap.loc[prior.max(), "members"])
        for _, row in window.iterrows():
            seen |= set(row["members"])
        return sorted(seen)

    def degraded_current(self, tickers) -> dict:
        """Record a current-member fallback and its survivorship-bias limitation."""
        logger.warning("Using CURRENT constituents - results carry survivorship bias.")
        return {
            "mode": "degraded_current_constituents",
            "survivorship_bias": True,
            "warning": "No point-in-time history available; today's members applied "
            "to all history. Prefer sp500_history.csv for a clean study.",
            "n_tickers": len(tickers),
            "generated_at": datetime.utcnow().isoformat() + "Z",
        }
