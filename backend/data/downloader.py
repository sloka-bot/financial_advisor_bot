import logging
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class MarketDataDownloader:

    def __init__(self, raw_dir="data/raw", start_date="2018-01-01"):
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.start_date = start_date
        self.end_date = datetime.now().strftime("%Y-%m-%d")

    def download_ticker(self, ticker, force=False):
        path = self._path(ticker)

        if path.exists() and not force:
            logger.debug(f"{ticker}: already cached")
            return pd.read_csv(path, index_col=0, parse_dates=True)

        try:
            df = yf.download(
                ticker,
                start=self.start_date,
                end=self.end_date,
                progress=False,
                auto_adjust=True,
                threads=False,
            )

            if df.empty:
                logger.warning(f"{ticker}: no data returned (possibly delisted)")
                return None

            # yfinance sometimes returns MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.columns = [c.lower() for c in df.columns]
            df.index.name = "date"
            df.to_csv(path)

            logger.info(f"  {ticker}: {len(df)} rows  ({df.index[0].date()} to {df.index[-1].date()})")
            return df

        except Exception as e:
            logger.error(f"  {ticker}: {e}")
            return None

    def download_universe(self, tickers, delay=0.25, force=False):
        results = {"success": [], "skipped": [], "failed": []}
        total = len(tickers)

        logger.info(f"Downloading {total} tickers (start={self.start_date})")

        for i, ticker in enumerate(tickers, 1):
            if self._path(ticker).exists() and not force:
                results["skipped"].append(ticker)
            else:
                df = self.download_ticker(ticker, force=force)
                if df is not None and not df.empty:
                    results["success"].append(ticker)
                else:
                    results["failed"].append(ticker)
                time.sleep(delay)

            if i % 10 == 0 or i == total:
                logger.info(f"  [{i}/{total}]  downloaded={len(results['success'])}  skipped={len(results['skipped'])}  failed={len(results['failed'])}")

        return results

    def load(self, ticker):
        path = self._path(ticker)
        if not path.exists():
            return None
        return pd.read_csv(path, index_col=0, parse_dates=True)

    def _path(self, ticker):
        return self.raw_dir / f"{ticker.replace('/', '_').replace(':', '_')}.csv"
