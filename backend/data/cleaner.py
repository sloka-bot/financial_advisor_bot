import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

MAX_FFILL = 5         # max consecutive missing days to forward-fill
EXTREME_MOVE = 0.40   # flag moves >= 40% but never remove them


class DataCleaner:

    def __init__(self, raw_dir="data/raw", processed_dir="data/processed"):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def clean(self, ticker, df=None, save=True):
        if df is None:
            df = self._load_raw(ticker)
        if df is None or df.empty:
            return None

        n_original = len(df)

        # sort by date and remove any duplicate rows Yahoo sends sometimes
        df = df.sort_index()
        dupes = df.index.duplicated(keep="first").sum()
        if dupes:
            df = df[~df.index.duplicated(keep="first")]

        df.columns = [c.lower() for c in df.columns]
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            logger.error(f"{ticker}: missing columns {required - set(df.columns)}")
            return None

        for col in required:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # zero or negative prices are data errors, not crashes
        bad_prices = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
        df = df[~bad_prices]

        # High < Low is physically impossible — corrupt row
        df = df[~(df["high"] < df["low"])]

        # zero volume = non-trading day that slipped through
        df = df[df["volume"] > 0]

        # fill small gaps (weekends, bank holidays)
        n_missing = df.isnull().sum().sum()
        if n_missing:
            df = df.ffill(limit=MAX_FFILL).bfill(limit=1)
        df = df.dropna()

        # flag extreme moves but keep them — crashes and earnings surprises are real signals
        df["daily_return"] = df["close"].pct_change()
        df["extreme_move_flag"] = (df["daily_return"].abs() > EXTREME_MOVE).astype(int)

        dropped = n_original - len(df)
        logger.info(f"  {ticker}: {n_original} -> {len(df)} rows (dropped {dropped})")

        if save:
            df.to_csv(self.processed_dir / f"{ticker}.csv")

        return df

    def clean_universe(self, tickers):
        results = {"success": [], "failed": []}
        logger.info(f"Cleaning {len(tickers)} tickers...")

        for i, ticker in enumerate(tickers, 1):
            df = self.clean(ticker)
            if df is not None and not df.empty:
                results["success"].append(ticker)
            else:
                results["failed"].append(ticker)

            if i % 10 == 0 or i == len(tickers):
                logger.info(f"  [{i}/{len(tickers)}]  ok={len(results['success'])}  failed={len(results['failed'])}")

        return results

    def load(self, ticker):
        path = self.processed_dir / f"{ticker}.csv"
        if not path.exists():
            return None
        return pd.read_csv(path, index_col=0, parse_dates=True)

    def _load_raw(self, ticker):
        path = self.raw_dir / f"{ticker}.csv"
        if not path.exists():
            logger.warning(f"{ticker}: raw file not found")
            return None
        return pd.read_csv(path, index_col=0, parse_dates=True)
