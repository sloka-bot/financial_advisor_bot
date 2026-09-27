"""
downloader.py

Downloads and caches historical OHLCV price data from Yahoo Finance.

Unlike a plain `auto_adjust=True` pull, this keeps both price views:

  * open / high / low / close  -> split & dividend ADJUSTED (analytical) prices,
    used to compute indicators so a split does not look like a 50% crash;
  * close_unadj                -> the raw, UNADJUSTED close, i.e. the price an
    order would actually have executed at, and the series used to detect splits
    that were never applied.

Full history is fetched on the first run; subsequent runs fetch only the new
rows and merge them in.
"""

import logging
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from backend.config.settings import HISTORY_START

logger = logging.getLogger(__name__)

FULL_HISTORY_START = HISTORY_START
OUTPUT_COLS = ["open", "high", "low", "close", "volume", "close_unadj"]


class MarketDataDownloader:
    """Download and cache adjusted analysis data and raw execution prices."""

    def __init__(self, raw_dir="data/raw"):
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    def download_ticker(self, ticker, force=False):
        """Download full history from HISTORY_START. Skip if already cached."""
        path = self._path(ticker)
        if path.exists() and not force:
            logger.debug(f"{ticker}: already cached; use incremental to update")
            return pd.read_csv(path, index_col=0, parse_dates=True)
        return self._fetch_range(ticker, FULL_HISTORY_START, datetime.now().strftime("%Y-%m-%d"), path)

    def download_ticker_incremental(self, ticker):
        """Fetch only the sessions we do not have yet; full pull if no cache."""
        path = self._path(ticker)
        existing = self._load_raw(path)

        if existing is None or existing.empty or "close_unadj" not in existing.columns:
            logger.info(f"{ticker}: no existing data - downloading full history")
            return self._fetch_range(ticker, FULL_HISTORY_START, datetime.now().strftime("%Y-%m-%d"), path)

        last_date = existing.index.max()
        today = pd.Timestamp.now(tz=None).normalize()
        if last_date >= today - pd.Timedelta(days=1):
            logger.debug(f"{ticker}: up to date (last {last_date.date()})")
            return existing

        # Re-fetch a small OVERLAP so we can detect a corporate action that
        # re-scaled the whole adjusted history since we cached it. Adjusted OHLC is
        # relative to the latest split/dividend, so appending freshly-adjusted rows
        # onto stale-adjusted ones would leave a discontinuity at the seam.
        overlap_start = (last_date - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
        end = datetime.now().strftime("%Y-%m-%d")
        new = self._fetch_range(ticker, overlap_start, end)
        if new is None or new.empty:
            return existing

        common = existing.index.intersection(new.index)
        if len(common):
            a = pd.to_numeric(existing.loc[common, "close"], errors="coerce")
            b = pd.to_numeric(new.loc[common, "close"], errors="coerce")
            rel = ((a - b).abs() / b.replace(0, np.nan)).max()
            if pd.notna(rel) and rel > 1e-5:
                logger.info(
                    f"{ticker}: adjustment factor changed (likely split/dividend) - "
                    "re-downloading full history so the series stays consistent"
                )
                return self._fetch_range(ticker, FULL_HISTORY_START, end, path)

        merged = pd.concat([existing, new]).sort_index()
        merged = merged[~merged.index.duplicated(keep="last")]  # re-fetched rows win over cached
        merged.to_csv(path)
        added = max(0, len(merged) - len(existing))
        logger.info(f"{ticker}: refreshed overlap + added {added} rows (total {len(merged)})")
        return merged

    def download_universe(self, tickers, delay=0.3, progress_cb=None):
        """Download each ticker and report successes and failures."""
        results = {"success": [], "failed": []}
        n = len(tickers)
        for i, ticker in enumerate(tickers, 1):
            time.sleep(delay)
            df = self.download_ticker(ticker)
            (results["success"] if df is not None and not df.empty else results["failed"]).append(ticker)
            if progress_cb is not None:
                progress_cb(i, n)
        logger.info(f"Download complete: {len(results['success'])} ok, {len(results['failed'])} failed")
        return results

    def download_universe_incremental(self, tickers, delay=0.3, progress_cb=None):
        """Update cached ticker histories and report per-ticker outcomes."""
        results = {"success": [], "failed": []}
        n = len(tickers)
        for i, ticker in enumerate(tickers, 1):
            time.sleep(delay)
            df = self.download_ticker_incremental(ticker)
            (results["success"] if df is not None and not df.empty else results["failed"]).append(ticker)
            if progress_cb is not None:
                progress_cb(i, n)
        return results

    # ------------------------------------------------------------------ #
    def _fetch_range(self, ticker, start, end, save_path=None):
        """Download with retries and derive adjusted + unadjusted price views."""
        for attempt in range(1, 4):
            try:
                raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False, threads=False)
                if raw is None or raw.empty:
                    logger.warning(f"{ticker}: empty response (attempt {attempt}/3)")
                    time.sleep(attempt)
                    continue

                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                raw.columns = [str(c).lower() for c in raw.columns]
                raw.index.name = "date"

                df = self._to_adjusted(raw)
                if df is None:
                    continue

                if save_path:
                    df.to_csv(save_path)
                    logger.info(f"{ticker}: {len(df)} rows saved (adjusted + executable close)")
                return df
            except Exception as e:
                logger.warning(f"{ticker}: attempt {attempt}/3 failed - {e}")
                if attempt < 3:
                    time.sleep((2**attempt) + random.uniform(0, 1))

        logger.error(f"{ticker}: all 3 download attempts failed")
        return None

    @staticmethod
    def _to_adjusted(raw: pd.DataFrame):
        """Return open/high/low/close (adjusted) + volume + close_unadj (raw)."""
        need = {"open", "high", "low", "close", "volume"}
        if not need.issubset(raw.columns):
            logger.error(f"missing columns from Yahoo: {need - set(raw.columns)}")
            return None

        raw_close = pd.to_numeric(raw["close"], errors="coerce")
        adj_close = pd.to_numeric(raw.get("adj close", raw["close"]), errors="coerce")
        # per-day cumulative adjustment factor (splits + dividends)
        factor = (adj_close / raw_close.replace(0, np.nan)).fillna(1.0)

        out = pd.DataFrame(index=raw.index)
        out["open"] = pd.to_numeric(raw["open"], errors="coerce") * factor
        out["high"] = pd.to_numeric(raw["high"], errors="coerce") * factor
        out["low"] = pd.to_numeric(raw["low"], errors="coerce") * factor
        out["close"] = adj_close  # analytical (adjusted) close
        out["volume"] = pd.to_numeric(raw["volume"], errors="coerce")
        out["close_unadj"] = raw_close  # executable (raw) close
        return out[OUTPUT_COLS]

    def _path(self, ticker):
        return self.raw_dir / f"{ticker.replace('/', '_')}.csv"

    def _load_raw(self, path):
        if not path.exists():
            return None
        try:
            return pd.read_csv(path, index_col=0, parse_dates=True)
        except Exception:
            return None
