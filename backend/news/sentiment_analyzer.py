"""Score deduplicated news with FinBERT and aggregate by its first eligible session."""

import hashlib
import json
import logging
import math
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd

from backend.data import market_calendar

logger = logging.getLogger(__name__)

FINBERT_MODEL = "ProsusAI/finbert"
LABEL_SCORE = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

# Version text preprocessing to invalidate incompatible cached scores.
PREPROCESSING_VERSION = 1


def _tokenizer_version() -> str:
    """Read the transformers version used in sentiment cache identities."""
    try:
        from importlib import metadata as _md

        return _md.version("transformers")
    except Exception:
        return "unknown"


class SentimentAnalyzer:
    """Score news with FinBERT and aggregate it to trading decision dates."""

    def __init__(self, news_dir="data/news/live", sentiment_dir="data/sentiment/live", batch_size=16, max_length=512):
        self.news_dir = Path(news_dir)
        self.sentiment_dir = Path(sentiment_dir)
        self.sentiment_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.max_length = max_length
        self._pipe = None

    def load_model(self):
        """Lazily initialise the configured FinBERT inference pipeline."""
        if self._pipe is not None:
            return
        import torch
        from transformers import pipeline

        if torch.cuda.is_available():
            device = 0
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = -1
        logger.info("Loading FinBERT (first run downloads ~500MB)...")
        self._pipe = pipeline(
            "text-classification",
            model=FINBERT_MODEL,
            tokenizer=FINBERT_MODEL,
            device=device,
            truncation=True,
            max_length=self.max_length,
        )
        logger.info("FinBERT loaded")

    # Per-ticker sentiment scoring.
    def analyze_ticker(self, ticker, save=True):
        """Score cached ticker news and persist session-level sentiment features."""
        news_path = self.news_dir / f"{ticker.replace('/', '_')}.json"
        if not news_path.exists():
            logger.info(f"{ticker}: no news file - neutral/no-news fallback")
            return self._no_news_row(ticker, save)

        data = json.loads(news_path.read_text())
        articles = self._dedupe(data.get("articles", []))
        if not articles:
            return self._no_news_row(ticker, save)

        # Score articles whose cached sentiment no longer matches the current inputs.
        tok_ver = _tokenizer_version()

        def cache_valid(article):
            """Reuse scores only when text, model, tokenizer and preprocessing settings match."""
            score = article.get("sentiment_score")
            return (
                article.get("sentiment_model") == FINBERT_MODEL
                and article.get("sentiment_tokenizer_version") == tok_ver
                and article.get("sentiment_preprocess_version") == PREPROCESSING_VERSION
                and article.get("sentiment_text_sha256") == hashlib.sha256(self._text(article).encode()).hexdigest()
                and article.get("sentiment_max_length") == self.max_length
                and article.get("sentiment_label") in LABEL_SCORE
                and isinstance(score, (int, float))
                and math.isfinite(score)
                and 0 <= score <= 1
            )

        pending = [article for article in articles if not cache_valid(article)]
        if pending:
            self.load_model()
            preds = self._classify([self._text(a) for a in pending])
            for article, pred in zip(pending, preds):
                label = pred["label"].lower()
                article["sentiment_model"] = FINBERT_MODEL
                article["sentiment_tokenizer_version"] = tok_ver
                article["sentiment_preprocess_version"] = PREPROCESSING_VERSION
                article["sentiment_text_sha256"] = hashlib.sha256(self._text(article).encode()).hexdigest()
                article["sentiment_max_length"] = self.max_length
                article["sentiment_label"] = label
                article["sentiment_score"] = float(pred["score"])
                article["sentiment_compound"] = LABEL_SCORE.get(label, 0.0) * float(pred["score"])
                article["sentiment_truncated"] = bool(pred.get("truncated", False))
                article["sentiment_token_count"] = int(pred.get("n_tokens", 0))
            n_trunc = sum(1 for a in pending if a.get("sentiment_truncated"))
            self._trunc_scored = getattr(self, "_trunc_scored", 0) + len(pending)
            self._trunc_over = getattr(self, "_trunc_over", 0) + n_trunc
            if n_trunc:
                logger.info(
                    "%s: %d/%d scored inputs exceeded %d tokens and were truncated (%.1f%%)",
                    ticker,
                    n_trunc,
                    len(pending),
                    self.max_length,
                    100.0 * n_trunc / len(pending),
                )
        # Add compound scores to older cache entries.
        for article in articles:
            if article.get("sentiment_compound") is None and article.get("sentiment_label") is not None:
                lbl = str(article["sentiment_label"]).lower()
                article["sentiment_compound"] = LABEL_SCORE.get(lbl, 0.0) * float(article.get("sentiment_score") or 0.0)

        data["articles"] = articles
        data["analyzed_at"] = datetime.utcnow().isoformat() + "Z"
        if save:
            news_path.write_text(json.dumps(data, indent=2))

        daily = self._aggregate_daily(articles)
        if save and daily is not None:
            daily.to_csv(self.sentiment_dir / f"{ticker.replace('/', '_')}.csv")
            logger.info(f"  {ticker}: {len(articles)} unique articles -> {len(daily)} daily rows")
        return daily

    def analyze_universe(self, tickers, progress_cb=None):
        # Load FinBERT only when new articles require scoring.
        """Process each ticker and report successful, skipped and failed inputs."""
        results = {"success": [], "skipped": [], "failed": []}
        self._trunc_scored = 0
        self._trunc_over = 0
        n = len(tickers)
        for i, ticker in enumerate(tickers, 1):
            if not (self.news_dir / f"{ticker.replace('/', '_')}.json").exists():
                results["skipped"].append(ticker)
                continue
            try:
                df = self.analyze_ticker(ticker)
                results["success" if (df is not None and not df.empty) else "failed"].append(ticker)
            except Exception as exc:  # noqa: BLE001
                logger.error(f"  {ticker}: {exc}")
                results["failed"].append(ticker)
            if i % 25 == 0 or i == n:
                logger.info(
                    f"  [{i}/{n}] ok={len(results['success'])} "
                    f"skipped={len(results['skipped'])} failed={len(results['failed'])}"
                )
            if progress_cb is not None:
                progress_cb(i, n)
        if self._trunc_scored:
            results["truncated_fraction"] = round(self._trunc_over / self._trunc_scored, 4)
            logger.info(
                "Universe: %d/%d scored inputs truncated at %d tokens (%.1f%%)",
                self._trunc_over,
                self._trunc_scored,
                self.max_length,
                100.0 * self._trunc_over / self._trunc_scored,
            )
        return results

    def load(self, ticker):
        """Read cached session sentiment, returning None when absent."""
        path = self.sentiment_dir / f"{ticker.replace('/', '_')}.csv"
        return pd.read_csv(path, index_col=0, parse_dates=True) if path.exists() else None

    def latest(self, ticker):
        """Return the most recent stored sentiment record for a ticker."""
        df = self.load(ticker)
        if df is None or df.empty:
            return {"label": "neutral", "score": 0.0, "news_count": 0, "no_news": 1}
        row = df.iloc[-1]
        return {
            "label": row.get("sent_label", "neutral"),
            "score": float(row.get("sent_score", 0.0)),
            "news_count": int(row.get("sent_news_count", 0)),
            "no_news": int(row.get("sent_no_news", 1)),
        }

    # Sentiment preprocessing helpers.
    @staticmethod
    def _text(article):
        title = (article.get("title") or "").strip()
        content = (article.get("content") or article.get("summary") or "").strip()
        return (title + ". " + content).strip() if content else title

    @staticmethod
    @lru_cache(maxsize=40)
    def _session_closes(year):
        return market_calendar.session_closes(year)

    @classmethod
    def _decision_date(cls, ts):
        """Map publication to the next eligible close; date-only news waits until the next day."""
        try:
            t = pd.Timestamp(ts)
            if pd.isna(t):
                return None
            if isinstance(ts, str) and len(ts.strip()) == 10:
                t += pd.Timedelta(days=1)
            t = t.tz_localize("US/Eastern") if t.tzinfo is None else t.tz_convert("US/Eastern")
            closes = cls._session_closes(t.year)
            if closes is None:
                return None
            eligible = closes[closes > t.tz_convert("UTC")]
            return pd.Timestamp(eligible.index[0]).tz_localize(None) if len(eligible) else None
        except ImportError:
            logger.error("Market calendar dependency unavailable; sentiment dates cannot be validated")
            return None
        except (ValueError, TypeError, IndexError):
            return None

    @classmethod
    def _dedupe(cls, articles):
        """Deduplicate titles within decision dates and store those dates as ISO strings."""
        # Order articles by publication, title and content before retaining duplicates.
        articles = sorted(
            articles,
            key=lambda a: (
                str(a.get("published_at") or ""),
                (a.get("title") or "").strip().lower(),
                (a.get("content") or a.get("summary") or ""),
            ),
        )
        seen = set()
        out = []
        for a in articles:
            title = (a.get("title") or "").strip().lower()
            ts = a.get("published_at")
            if not title or not ts:
                continue  # skip rows without a title or date
            date = cls._decision_date(ts)
            if date is None:
                continue
            key = (title, date)
            if key in seen:
                continue
            seen.add(key)
            a["_date"] = date.isoformat()  # ISO string
            out.append(a)
        return out

    def _aggregate_daily(self, articles):
        rows = []
        for a in articles:
            ds = a.get("_date")
            if ds is None:
                continue
            date = pd.to_datetime(ds)  # _date holds an ISO string
            if pd.isna(date):
                continue
            rows.append(
                {
                    "date": date,
                    "label": a.get("sentiment_label", "neutral"),
                    "compound": a.get("sentiment_compound", 0.0),
                    "score": a.get("sentiment_score", 0.5),
                }
            )
        if not rows:
            return None
        df = pd.DataFrame(rows)

        def agg(group):
            """Summarise sentiment labels and scores within one decision session."""
            labels = group["label"]
            return pd.Series(
                {
                    "sent_score": group["compound"].mean(),
                    # Weight signed confidence by confidence again to emphasise confident articles.
                    "sent_weighted": (group["compound"] * group["score"]).mean(),
                    "sent_pos": int((labels == "positive").sum()),
                    "sent_neg": int((labels == "negative").sum()),
                    "sent_neu": int((labels == "neutral").sum()),
                    "sent_news_count": int(len(labels)),
                    # Disagreement is the spread of article sentiment that day.
                    "sent_disagreement": float(group["compound"].std(ddof=0)) if len(labels) > 1 else 0.0,
                    "sent_label": labels.value_counts().idxmax(),
                    "sent_no_news": 0,  # this day has articles
                }
            )

        # Exclude the grouping column from the aggregation callback.
        daily = df.groupby("date").apply(agg, include_groups=False)
        daily.index.name = "date"
        return daily

    def _no_news_row(self, ticker, save=True):
        """Create an explicit no-news row for sentiment fusion."""
        row = pd.DataFrame(
            [
                {
                    "date": pd.to_datetime(datetime.utcnow().date()),
                    "sent_score": 0.0,
                    "sent_weighted": 0.0,
                    "sent_pos": 0,
                    "sent_neg": 0,
                    "sent_neu": 0,
                    "sent_news_count": 0,
                    "sent_disagreement": 0.0,
                    "sent_label": "neutral",
                    "sent_no_news": 1,
                }
            ]
        ).set_index("date")
        if save:
            row.to_csv(self.sentiment_dir / f"{ticker.replace('/', '_')}.csv")
        return row

    def _classify(self, texts):
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            predictions = self._pipe(batch)
            if len(predictions) != len(batch):
                raise ValueError("FinBERT returned an incomplete batch")
            # Record token truncation when the classifier exposes its tokenizer.
            tokenizer = getattr(self._pipe, "tokenizer", None)
            if tokenizer is not None:
                try:
                    token_counts = [len(ids) for ids in tokenizer(batch, truncation=False)["input_ids"]]
                except Exception:
                    token_counts = [0] * len(batch)
            else:
                token_counts = [0] * len(batch)
            for prediction, n_tokens in zip(predictions, token_counts):
                label = str(prediction["label"]).lower()
                score = float(prediction["score"])
                if label not in LABEL_SCORE or not math.isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("FinBERT returned an invalid classification")
                results.append(
                    {
                        "label": label,
                        "score": score,
                        "n_tokens": int(n_tokens),
                        "truncated": bool(n_tokens > self.max_length),
                    }
                )
        return results
