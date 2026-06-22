import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

FINBERT_MODEL = "ProsusAI/finbert"
LABEL_SCORE   = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}


class SentimentAnalyzer:

    def __init__(self, news_dir="data/news", sentiment_dir="data/sentiment",
                 batch_size=16, max_length=512):
        self.news_dir      = Path(news_dir)
        self.sentiment_dir = Path(sentiment_dir)
        self.sentiment_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size  = batch_size
        self.max_length  = max_length
        self._pipe       = None

    def load_model(self):
        if self._pipe is not None:
            return
        import torch
        from transformers import pipeline
        device = 0 if torch.cuda.is_available() else -1
        logger.info(f"Loading FinBERT on {'GPU' if device == 0 else 'CPU'}...")
        logger.info("  First run downloads ~500MB from HuggingFace")
        self._pipe = pipeline(
            "text-classification",
            model=FINBERT_MODEL,
            tokenizer=FINBERT_MODEL,
            device=device,
            truncation=True,
            max_length=self.max_length,
        )
        logger.info("FinBERT loaded")

    def analyze_ticker(self, ticker, save=True):
        self.load_model()

        news_path = self.news_dir / f"{ticker.replace('/', '_')}.json"
        if not news_path.exists():
            logger.warning(f"{ticker}: no news file")
            return None

        data     = json.loads(news_path.read_text())
        articles = data.get("articles", [])
        if not articles:
            return None

        texts = [a["title"] for a in articles if a.get("title")]
        if not texts:
            return None

        preds = self._classify(texts)

        for article, pred in zip(articles, preds):
            label = pred["label"].lower()
            score = pred["score"]
            article["sentiment_label"]    = label
            article["sentiment_score"]    = score
            article["sentiment_compound"] = LABEL_SCORE.get(label, 0.0) * score

        data["articles"]    = articles
        data["analyzed_at"] = datetime.utcnow().isoformat() + "Z"
        news_path.write_text(json.dumps(data, indent=2))

        daily = self._aggregate_daily(articles)

        if save and daily is not None:
            out = self.sentiment_dir / f"{ticker.replace('/', '_')}.csv"
            daily.to_csv(out)
            logger.info(f"  {ticker}: {len(articles)} headlines -> {len(daily)} daily rows")

        return daily

    def analyze_universe(self, tickers):
        self.load_model()
        results = {"success": [], "skipped": [], "failed": []}

        for i, ticker in enumerate(tickers, 1):
            if not (self.news_dir / f"{ticker.replace('/', '_')}.json").exists():
                results["skipped"].append(ticker)
                continue
            try:
                df = self.analyze_ticker(ticker)
                if df is not None and not df.empty:
                    results["success"].append(ticker)
                else:
                    results["failed"].append(ticker)
            except Exception as e:
                logger.error(f"  {ticker}: {e}")
                results["failed"].append(ticker)

            if i % 10 == 0 or i == len(tickers):
                logger.info(f"  [{i}/{len(tickers)}]  done={len(results['success'])}  skipped={len(results['skipped'])}")

        return results

    def load(self, ticker):
        path = self.sentiment_dir / f"{ticker.replace('/', '_')}.csv"
        if not path.exists():
            return None
        return pd.read_csv(path, index_col=0, parse_dates=True)

    def latest(self, ticker):
        df = self.load(ticker)
        if df is None or df.empty:
            return {"label": "neutral", "score": 0.0, "news_count": 0}
        row = df.iloc[-1]
        return {
            "label":      row.get("sentiment_label", "neutral"),
            "score":      float(row.get("sentiment_score", 0.0)),
            "news_count": int(row.get("news_count", 0)),
        }

    def _classify(self, texts):
        all_results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i: i + self.batch_size]
            try:
                preds = self._pipe(batch)
                for p in preds:
                    p["label"] = p["label"].lower()
                all_results.extend(preds)
            except Exception as e:
                logger.warning(f"  batch failed: {e}")
                all_results.extend([{"label": "neutral", "score": 0.5}] * len(batch))
        return all_results

    def _aggregate_daily(self, articles):
        rows = []
        for a in articles:
            ts = a.get("published_at")
            if not ts:
                continue
            try:
                date = pd.to_datetime(ts).normalize()
            except Exception:
                continue
            rows.append({
                "date":     date,
                "label":    a.get("sentiment_label", "neutral"),
                "compound": a.get("sentiment_compound", 0.0),
                "score":    a.get("sentiment_score", 0.5),
            })

        if not rows:
            return None

        df = pd.DataFrame(rows)

        def agg(g):
            labels = g["label"]
            return pd.Series({
                "sentiment_score":  g["compound"].mean(),
                "sentiment_label":  labels.value_counts().idxmax(),
                "pos_count":        (labels == "positive").sum(),
                "neg_count":        (labels == "negative").sum(),
                "neu_count":        (labels == "neutral").sum(),
                "news_count":       len(labels),
                "weighted_score":   (g["compound"] * g["score"]).mean(),
            })

        daily = df.groupby("date").apply(agg)
        daily.index.name = "date"
        return daily
