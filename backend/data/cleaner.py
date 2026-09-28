"""Clean analytical prices, preserve execution prices and record quality findings."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from backend.config.settings import CLEANING
from backend.data import market_calendar

logger = logging.getLogger(__name__)

REQUIRED = ("open", "high", "low", "close", "volume")

# Keep adjusted analytical prices separate from raw execution prices.
EXEC_CLOSE_SRC = "close_unadj"

FLAG_COLS = [
    "extreme_move_flag",  # |daily return| beyond CLEANING["extreme_move"]
    "outlier_flag",  # rolling-MAD outlier, flagged only
    "potential_split_flag",  # unadjusted jump near a split ratio, adj continuous
    "corp_action_flag",  # any suspected corporate action on this row
    "ohlc_breach_flag",  # OHLC ordering violated (repaired or dropped)
    "penny_stock_flag",  # close below CLEANING["penny_price"]
    "imputed_flag",  # value on this row was forward-filled
    "suspension_flag",  # row sits at the edge of an excluded long gap
]


class DataCleaner:
    """Clean market histories and persist data-quality audit findings."""

    def __init__(self, raw_dir="data/raw", processed_dir="data/processed", audit_dir="data/audit"):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.audit_dir = Path(audit_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.audit_dir.mkdir(parents=True, exist_ok=True)

    # Public cleaning methods.
    def clean(self, ticker, df=None, save=True):
        """Clean one ticker and expose its quality findings through last_audit."""
        if df is None:
            df = self._load_raw(ticker)
        if df is None or df.empty:
            self.last_audit = {"ticker": ticker, "status": "empty_input"}
            return None

        audit = {"ticker": ticker, "n_input": int(len(df))}

        df = df.copy()
        df.columns = [str(c).lower() for c in df.columns]

        # De-duplicate and sort the calendar.
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df[~df.index.isna()]
        df = df.sort_index()
        audit["dupe_rows"] = int(df.index.duplicated(keep="first").sum())
        df = df[~df.index.duplicated(keep="first")]

        # Drop weekend rows.
        audit["weekend_rows"] = int((df.index.dayofweek >= 5).sum())
        df = df[df.index.dayofweek < 5]

        if not set(REQUIRED).issubset(df.columns):
            missing = set(REQUIRED) - set(df.columns)
            logger.error(f"{ticker}: missing columns {missing}")
            self.last_audit = {"ticker": ticker, "status": f"missing_columns:{sorted(missing)}"}
            return None

        # Keep the unadjusted close when the download provides one.
        has_exec = EXEC_CLOSE_SRC in df.columns
        if has_exec:
            df["exec_close"] = pd.to_numeric(df[EXEC_CLOSE_SRC], errors="coerce")
        audit["has_executable_prices"] = bool(has_exec)

        for col in REQUIRED:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Invalid observations: positive prices, non-negative volume and OHLC order.
        price_cols = ["open", "high", "low", "close"]
        nonpos = (df[price_cols] <= 0).any(axis=1)
        audit["nonpositive_price_rows"] = int(nonpos.sum())
        df = df[~nonpos]

        neg_vol = df["volume"] < 0
        audit["negative_volume_rows"] = int(neg_vol.sum())
        df = df[~neg_vol]

        # High is the maximum and low the minimum of OHLC.
        row_max = df[price_cols].max(axis=1)
        row_min = df[price_cols].min(axis=1)
        breach = (df["high"] < row_max - 1e-9) | (df["low"] > row_min + 1e-9)
        df["ohlc_breach_flag"] = breach.astype(int)
        audit["ohlc_breach_rows"] = int(breach.sum())
        # Repair recoverable OHLC bounds and reject rows where high is below low.
        df.loc[breach, "high"] = df.loc[breach, price_cols].max(axis=1)
        df.loc[breach, "low"] = df.loc[breach, price_cols].min(axis=1)
        impossible = df["high"] < df["low"]
        audit["impossible_rows_dropped"] = int(impossible.sum())
        df = df[~impossible]

        if df.empty:
            self.last_audit = {**audit, "status": "empty_after_validity"}
            return None

        # Missing data: short forward-fill, long gaps excluded.
        df, gap_audit = self._handle_gaps(df, ticker)
        audit.update(gap_audit)

        # Corporate actions and splits.
        df, ca_audit = self._flag_corporate_actions(df, has_exec)
        audit.update(ca_audit)

        # Returns, extreme moves and robust outliers are flagged only.
        df["daily_return"] = df["close"].pct_change()
        df["extreme_move_flag"] = (df["daily_return"].abs() > CLEANING["extreme_move"]).astype(int)
        df["outlier_flag"] = self._robust_outliers(df["daily_return"])
        audit["extreme_move_rows"] = int(df["extreme_move_flag"].sum())
        audit["outlier_rows"] = int(df["outlier_flag"].sum())

        # Penny stock flag.
        df["penny_stock_flag"] = (df["close"] < CLEANING["penny_price"]).astype(int)
        audit["penny_rows"] = int(df["penny_stock_flag"].sum())

        # Add any absent flag columns as zero.
        for c in FLAG_COLS:
            if c not in df.columns:
                df[c] = 0
            df[c] = df[c].fillna(0).astype(int)

        df = df.dropna(subset=list(REQUIRED))

        audit["n_output"] = int(len(df))
        audit["net_row_change"] = int(len(df) - audit["n_input"])
        audit["date_start"] = str(df.index.min().date()) if len(df) else None
        audit["date_end"] = str(df.index.max().date()) if len(df) else None
        audit["status"] = "ok"
        self.last_audit = audit

        logger.info(
            f"  {ticker}: {audit['n_input']} in -> {len(df)} out "
            f"(imputed {audit.get('rows_imputed', 0)}, "
            f"excluded-longgap {audit.get('rows_excluded_longgap', 0)}, "
            f"splits~{audit.get('suspected_splits', 0)}, "
            f"extreme {audit['extreme_move_rows']})"
        )

        if save:
            df.to_csv(self.processed_dir / f"{ticker}.csv")

        return df

    def clean_universe(self, tickers, progress_cb=None):
        """Clean every ticker and write a consolidated audit trail to disk."""
        results = {"success": [], "failed": []}
        audits = []
        n = len(tickers)
        logger.info(f"Cleaning {n} tickers...")

        for i, ticker in enumerate(tickers, 1):
            df = self.clean(ticker)
            audits.append(getattr(self, "last_audit", {"ticker": ticker, "status": "unknown"}))
            if df is not None and not df.empty:
                results["success"].append(ticker)
            else:
                results["failed"].append(ticker)
            if i % 25 == 0 or i == n:
                logger.info(f"  [{i}/{n}]  ok={len(results['success'])}  failed={len(results['failed'])}")
            if progress_cb is not None:
                progress_cb(i, n)

        # Write the audit trail.
        audit_df = pd.DataFrame(audits)
        audit_df.to_csv(self.audit_dir / "cleaning_audit.csv", index=False)

        def _sum(col):
            return int(audit_df.get(col, pd.Series(dtype=float)).fillna(0).sum())

        summary = {
            "tickers_total": len(tickers),
            "tickers_ok": len(results["success"]),
            "tickers_failed": len(results["failed"]),
            "rows_imputed_total": _sum("rows_imputed"),
            "rows_excluded_longgap_total": _sum("rows_excluded_longgap"),
            "ohlc_breach_total": _sum("ohlc_breach_rows"),
            "impossible_dropped_total": _sum("impossible_rows_dropped"),
            "nonpositive_price_total": _sum("nonpositive_price_rows"),
            "negative_volume_total": _sum("negative_volume_rows"),
            "suspected_splits_total": _sum("suspected_splits"),
            "tickers_without_executable_prices": int(
                (~audit_df.get("has_executable_prices", pd.Series([False] * len(audit_df))).fillna(False)).sum()
            ),
            "failed_tickers": results["failed"],
        }
        (self.audit_dir / "cleaning_summary.json").write_text(json.dumps(summary, indent=2))
        logger.info(
            f"Audit written to {self.audit_dir}/cleaning_audit.csv "
            f"({summary['rows_excluded_longgap_total']} long-gap rows excluded, "
            f"{summary['rows_imputed_total']} imputed across universe)"
        )
        return results

    def load(self, ticker):
        """Read cached cleaned prices, returning None when the file is absent."""
        path = self.processed_dir / f"{ticker}.csv"
        if not path.exists():
            return None
        return pd.read_csv(path, index_col=0, parse_dates=True)

    # Cleaning helpers.
    def _handle_gaps(self, df, ticker):
        """Align exchange sessions, fill short analytical gaps and exclude longer gaps."""
        limit = int(CLEANING["max_ffill_days"])
        # Leave missing execution prices unfilled; only analytical columns may carry forward.
        unfilled_cols = {"exec_close", "close_unadj"}
        price_like = [c for c in df.columns if c != "volume" and c not in unfilled_cols]
        sessions = self._sessions(ticker, df.index.min(), df.index.max())

        if sessions is None:
            miss = df["close"].isna()
            df[price_like] = df[price_like].ffill(limit=limit)
            df["volume"] = df["volume"].fillna(0)
            df["imputed_flag"] = (miss & df["close"].notna()).astype(int)
            df["suspension_flag"] = 0
            df = df.dropna(subset=["close"])
            return df, {
                "calendar_alignment": "skipped_no_calendar",
                "missing_sessions": None,
                "rows_imputed": int(df["imputed_flag"].sum()),
                "rows_excluded_longgap": 0,
                "suspension_edges": 0,
            }

        # Align observations to exchange sessions and discard non-session rows.
        non_session_rows = int(df.index.difference(sessions).size)
        target = sessions
        df = df.reindex(target).sort_index()
        is_session = df.index.isin(sessions)
        missing_session = is_session & df["close"].isna()
        n_missing_sessions = int(missing_session.sum())

        df[price_like] = df[price_like].ffill(limit=limit)
        df["volume"] = df["volume"].fillna(0)

        df["imputed_flag"] = (missing_session & df["close"].notna()).astype(int)

        still_missing = df["close"].isna()  # long gaps beyond the fill limit
        df["suspension_flag"] = 0
        edge = still_missing.shift(1, fill_value=False) & df["close"].notna()
        df.loc[edge, "suspension_flag"] = 1
        df = df[~still_missing]

        return df, {
            "calendar_alignment": self._exchange_for(ticker),
            "missing_sessions": n_missing_sessions,
            "rows_imputed": int(df["imputed_flag"].sum()),
            "rows_excluded_longgap": int(n_missing_sessions - df["imputed_flag"].sum()),
            "suspension_edges": int(df["suspension_flag"].sum()),
            "rows_dropped_non_session": non_session_rows,
        }

    @staticmethod
    def _exchange_for(ticker: str) -> str:
        """Map a ticker suffix to its exchange calendar name."""
        t = str(ticker).upper()
        if t.endswith(".SI"):
            return "XSES"  # Singapore Exchange
        if t.endswith(".L"):
            return "XLON"  # London Stock Exchange
        return "XNYS"  # NYSE / US default

    def _sessions(self, ticker, start, end):
        """Return exchange sessions, or None when the calendar is unavailable."""
        return market_calendar.sessions(
            self._exchange_for(ticker), pd.Timestamp(start).date(), pd.Timestamp(end).date()
        )

    def _flag_corporate_actions(self, df, has_exec):
        """Flag suspected splits from raw-price jumps absent from adjusted prices."""
        df["potential_split_flag"] = 0
        df["corp_action_flag"] = 0
        if not has_exec or "exec_close" not in df.columns:
            return df, {"suspected_splits": 0, "split_check": "unavailable_no_executable_prices"}

        # Raw close jumps on an unapplied split
        exec_ret = (df["exec_close"] / df["exec_close"].shift(1)).replace([np.inf, -np.inf], np.nan)
        # while the adjusted close stays continuous.
        adj_ret = (df["close"] / df["close"].shift(1)).replace([np.inf, -np.inf], np.nan)

        tol = CLEANING["split_tol"]
        suspected = pd.Series(False, index=df.index)
        for r in CLEANING["split_ratios"]:
            for ratio in (r, 1.0 / r):
                near = (exec_ret - ratio).abs() < tol
                continuous_adj = (adj_ret - 1.0).abs() < 0.20
                suspected = suspected | (near & continuous_adj)
        df.loc[suspected, "potential_split_flag"] = 1
        df.loc[suspected, "corp_action_flag"] = 1
        return df, {"suspected_splits": int(suspected.sum()), "split_check": "ok"}

    @staticmethod
    def _robust_outliers(returns: pd.Series) -> pd.Series:
        """Flag return deviations using a rolling median and median absolute deviation."""
        w = int(CLEANING["outlier_window"])
        k = float(CLEANING["outlier_mad_k"])
        med = returns.rolling(w, min_periods=w // 2).median()
        mad = (returns - med).abs().rolling(w, min_periods=w // 2).median()
        # 1.4826 scales MAD to a standard deviation
        scaled = 1.4826 * mad
        z = (returns - med).abs() / scaled.replace(0, np.nan)
        return (z > k).fillna(False).astype(int)

    def _load_raw(self, ticker):
        path = self.raw_dir / f"{ticker}.csv"
        if not path.exists():
            logger.warning(f"{ticker}: raw file not found")
            return None
        return pd.read_csv(path, index_col=0, parse_dates=True)
