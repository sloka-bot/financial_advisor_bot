"""Rebuild price-derived artifacts with a per-ticker audit and resumable raw cache."""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config.settings import HISTORY_START
from backend.data.cleaner import DataCleaner
from backend.data.downloader import MarketDataDownloader
from backend.data.feature_engineer import FeatureEngineer
from backend.data.fusion import FeatureFusion
from backend.universe.sp500_membership import SP500Membership


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="Smoke-test subset only; omit for the historical universe")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    end = str(datetime.now().date())
    tickers = SP500Membership().eligible_between(HISTORY_START, end)
    if args.limit:
        tickers = tickers[: args.limit]
    audit = {
        "started_at": datetime.now().isoformat(),
        "start": HISTORY_START,
        "end": end,
        "requested": tickers,
        "smoke_test": args.limit is not None,
        "success": {},
        "failed": {},
    }
    output = Path("data/audit/rebuild.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    downloader, cleaner, engineer, fusion = MarketDataDownloader(), DataCleaner(), FeatureEngineer(), FeatureFusion()
    start = time.monotonic()
    for i, ticker in enumerate(tickers, 1):
        try:
            raw = downloader.download_ticker_incremental(ticker)
            if raw is None or raw.empty:
                raise ValueError("No downloadable history; may be delisted or unavailable from provider")
            cleaned = cleaner.clean(ticker, raw)
            featured = engineer.generate(ticker, cleaned)
            if featured is None or featured.empty:
                raise ValueError("No usable technical history")
            master = fusion.fuse_ticker(ticker)
            audit["success"][ticker] = {
                "rows": len(master),
                "first": str(master.index.min().date()),
                "last": str(master.index.max().date()),
                "execution_prices": int(master["exec_close"].notna().sum()),
                "news_sessions": int((master["sent_news_count"] > 0).sum()),
            }
        except Exception as exc:
            logging.exception("%s failed", ticker)
            audit["failed"][ticker] = str(exc)
        audit["elapsed_seconds"] = round(time.monotonic() - start, 2)
        output.write_text(json.dumps(audit, indent=2))
        logging.info(
            "Rebuilt %d/%d; successful=%d; failed=%d", i, len(tickers), len(audit["success"]), len(audit["failed"])
        )
    audit["completed_at"] = datetime.now().isoformat()
    output.write_text(json.dumps(audit, indent=2))
    print(json.dumps({"success": len(audit["success"]), "failed": len(audit["failed"]), "audit": str(output)}))


if __name__ == "__main__":
    main()
