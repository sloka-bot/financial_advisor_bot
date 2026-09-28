"""Cache ticker news and distinguish collection failures from empty article lists."""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import yfinance as yf

logger = logging.getLogger(__name__)

# Refresh news caches older than the configured interval.
DEFAULT_MAX_AGE_HOURS = 12


class NewsCollector:
    """Fetch and cache current ticker news for live sentiment processing."""

    def __init__(self, news_dir="data/news/live"):
        # Create the news data directory.
        self.news_dir = Path(news_dir)
        self.news_dir.mkdir(parents=True, exist_ok=True)

    def get_ticker_news(self, ticker, force=False, max_age_hours=DEFAULT_MAX_AGE_HOURS):
        """Return the ticker's articles without collection-status metadata."""
        return self.fetch_ticker_news(ticker, force=force, max_age_hours=max_age_hours)[1]

    def fetch_ticker_news(self, ticker, force=False, max_age_hours=DEFAULT_MAX_AGE_HOURS):
        """Return collection status and articles, retaining cached articles on fetch failure."""
        path = self._path(ticker)

        # Serve from cache while fresh unless a refresh is forced.
        if path.exists() and not force:
            cached = self._read_cache(path)
            if cached is not None and not self._is_stale(cached, max_age_hours):
                return "cache_fresh", cached.get("articles", [])

        try:
            raw = yf.Ticker(ticker).news or []
            articles = []

            for item in raw:
                content = item.get("content") or item
                ts = content.get("pubDate") or content.get("providerPublishTime") or content.get("published")
                title = content.get("title", "").strip()
                if not title or not ts:
                    continue
                published = (
                    datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if isinstance(ts, (int, float)) else str(ts)
                )
                provider = content.get("provider") or {}
                link = content.get("canonicalUrl") or {}
                articles.append(
                    {
                        "uuid": item.get("id") or item.get("uuid", ""),
                        "title": title,
                        "publisher": provider.get("displayName", content.get("publisher", "")),
                        "link": link.get("url", "") if isinstance(link, dict) else link,
                        "published_at": published,
                        "content": content.get("summary") or title,
                    }
                )
            if raw and not articles:
                raise ValueError("News provider returned no usable titles/timestamps")

            # Save with a fresh timestamp.
            path.write_text(
                json.dumps(
                    {
                        "ticker": ticker,
                        "count": len(articles),
                        "collected_at": datetime.now(timezone.utc).isoformat(),
                        "articles": articles,
                    },
                    indent=2,
                )
            )

            logger.info(f"  {ticker}: {len(articles)} articles")
            return ("fetched" if articles else "no_articles"), articles

        except Exception as e:
            # Report fetch failures separately and retain previously cached articles.
            logger.warning(f"  {ticker}: news fetch failed - {e}")
            cached = self._read_cache(path) if path.exists() else None
            return "fetch_failed", (cached.get("articles", []) if cached else [])

    def get_universe_news(self, tickers, delay=0.5, force=False, max_age_hours=DEFAULT_MAX_AGE_HOURS, progress_cb=None):
        """Fetch universe news while distinguishing failed requests from empty responses."""
        results = {"success": [], "empty": [], "failed": []}
        n = len(tickers)
        for i, ticker in enumerate(tickers, 1):
            time.sleep(delay)
            status, articles = self.fetch_ticker_news(ticker, force=force, max_age_hours=max_age_hours)
            if status == "fetch_failed":
                results["failed"].append(ticker)
            elif articles:
                results["success"].append(ticker)
            else:
                results["empty"].append(ticker)
            if progress_cb is not None:
                progress_cb(i, n)
        logger.info(
            f"News: {len(results['success'])} with articles, "
            f"{len(results['empty'])} empty, {len(results['failed'])} failed"
        )
        return results

    @staticmethod
    def _read_cache(path):
        """Load a cached news payload, or None if it is missing/corrupt."""
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    @staticmethod
    def _is_stale(cached, max_age_hours):
        """Check cache age; missing dates are stale unless freshness checks are disabled."""
        if max_age_hours is None:
            return False
        ts = cached.get("collected_at")
        if not ts:
            return True
        try:
            collected = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except ValueError:
            return True
        if collected.tzinfo is None:
            collected = collected.replace(tzinfo=timezone.utc)
        age_hours = (datetime.now(timezone.utc) - collected).total_seconds() / 3600.0
        return age_hours > max_age_hours

    def _path(self, ticker):
        return self.news_dir / f"{ticker.replace('/', '_')}.json"
