import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import yfinance as yf

logger = logging.getLogger(__name__)


class NewsCollector:

    def __init__(self, news_dir="data/news"):
        self.news_dir = Path(news_dir)
        self.news_dir.mkdir(parents=True, exist_ok=True)

    def get_ticker_news(self, ticker, force=False):
        path = self._path(ticker)

        if path.exists() and not force:
            return json.loads(path.read_text()).get("articles", [])

        try:
            raw = yf.Ticker(ticker).news or []
            articles = []

            for item in raw:
                ts = item.get("providerPublishTime") or item.get("published")
                articles.append({
                    "uuid":         item.get("uuid", ""),
                    "title":        item.get("title", ""),
                    "publisher":    item.get("publisher", ""),
                    "link":         item.get("link", ""),
                    "published_at": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else None,
                    "content":      item.get("summary") or item.get("title", ""),
                })

            path.write_text(json.dumps({
                "ticker":       ticker,
                "count":        len(articles),
                "collected_at": datetime.utcnow().isoformat() + "Z",
                "articles":     articles,
            }, indent=2))

            logger.info(f"  {ticker}: {len(articles)} articles")
            return articles

        except Exception as e:
            logger.warning(f"  {ticker}: {e}")
            return []

    def get_universe_news(self, tickers, delay=0.5, force=False):
        results = {"success": [], "empty": [], "failed": []}
        logger.info(f"Collecting news for {len(tickers)} tickers...")

        for i, ticker in enumerate(tickers, 1):
            try:
                articles = self.get_ticker_news(ticker, force=force)
                if articles:
                    results["success"].append(ticker)
                else:
                    results["empty"].append(ticker)
            except Exception as e:
                logger.error(f"  {ticker}: {e}")
                results["failed"].append(ticker)

            if i % 10 == 0 or i == len(tickers):
                logger.info(f"  [{i}/{len(tickers)}]  with_news={len(results['success'])}")

            time.sleep(delay)

        return results

    def load(self, ticker):
        path = self._path(ticker)
        if not path.exists():
            return []
        return json.loads(path.read_text()).get("articles", [])

    def _path(self, ticker):
        return self.news_dir / f"{ticker.replace('/', '_').replace(':', '_')}.json"
