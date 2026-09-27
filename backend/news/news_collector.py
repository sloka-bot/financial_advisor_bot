"""
news_collector.py

Collects recent news headlines for each ticker from Yahoo Finance and
saves them as per-ticker JSON files. The sentiment analyser reads these
files in the next pipeline step.

Cached files carry a ``collected_at`` timestamp so a stale cache is
refreshed automatically instead of being served forever (see
``max_age_hours``). A fetch that raises is reported as ``fetch_failed`` and
kept distinct from ``no_articles`` (the fetch succeeded but the source
returned nothing): downstream, "we could not reach the source" must not be
treated the same as "there genuinely was no news".
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import yfinance as yf

logger = logging.getLogger(__name__)

# A cached news file older than this is refetched on the next request rather
# than served indefinitely. Kept modest so a daily pipeline run picks up news
# published since the previous collection.
DEFAULT_MAX_AGE_HOURS = 12


class NewsCollector:
    """Fetch and cache current ticker news for live sentiment processing."""

    def __init__(self, news_dir="data/news/live"):
        # Create the news data directory on startup
        self.news_dir = Path(news_dir)
        self.news_dir.mkdir(parents=True, exist_ok=True)

    def get_ticker_news(self, ticker, force=False, max_age_hours=DEFAULT_MAX_AGE_HOURS):
        """Return the ticker's article list (possibly empty).

        Backward-compatible wrapper around ``fetch_ticker_news`` for callers
        that only need the articles and do not care why the list is empty.
        """
        return self.fetch_ticker_news(ticker, force=force, max_age_hours=max_age_hours)[1]

    def fetch_ticker_news(self, ticker, force=False, max_age_hours=DEFAULT_MAX_AGE_HOURS):
        """Fetch (or serve from cache) one ticker's news.

        Returns ``(status, articles)`` where status is one of:
          "cache_fresh"   - served an unexpired cache
          "fetched"       - fetched fresh articles from Yahoo
          "no_articles"   - fetch succeeded but the source returned nothing
          "fetch_failed"  - the fetch raised (network/API error); a prior
                            cached copy, if any, is returned alongside so a
                            transient outage does not blank out sentiment.
        """
        path = self._path(ticker)

        # Serve from cache while it is still fresh (unless a refresh is forced).
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

            # Save to disk with a fresh timestamp so re-runs reuse it until stale.
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
            # A failed fetch is NOT the same as "no news". Report it distinctly
            # and, if a previous cache exists, still return those articles.
            logger.warning(f"  {ticker}: news fetch failed - {e}")
            cached = self._read_cache(path) if path.exists() else None
            return "fetch_failed", (cached.get("articles", []) if cached else [])

    def get_universe_news(self, tickers, delay=0.5, force=False, max_age_hours=DEFAULT_MAX_AGE_HOURS, progress_cb=None):
        """Fetch news for every ticker, keeping fetch failures distinct from
        genuinely empty (no-article) tickers."""
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
        """True if the cached payload is older than ``max_age_hours``.

        An undated or unparseable cache is treated as stale so it gets
        refreshed. ``max_age_hours=None`` disables the freshness check (the old
        serve-forever behaviour), for callers that want it.
        """
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
