"""Prepare deduplicated historical articles while preserving publication-time precision."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DAILY_ARTICLE_CAP = 5


def build_inputs(
    raw_dir=Path("data/fnspid/raw"),
    news_dir=Path("data/news/historical"),
    output=Path("data/fnspid/finbert_input.jsonl"),
):
    """Write historical news and a scoring manifest, capped at five articles per day."""
    news_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = {
        "tickers": 0,
        "input_articles": 0,
        "output_articles": 0,
        "date_only": 0,  # date-only publication maps to the next session
        "timestamped": 0,  # full timestamp maps to the next close
        "future_dropped": 0,  # future timestamps are dropped
    }
    _now = datetime.now(timezone.utc)
    with output.open("w") as stream:
        for path in sorted(raw_dir.glob("*.csv")):
            frame = pd.read_csv(path, dtype=str)
            counts["input_articles"] += len(frame)
            if not {"date", "title"}.issubset(frame):
                raise ValueError(f"Missing date or title column in {path.name}")
            frame["summary"] = frame["summary"].fillna("") if "summary" in frame else ""
            frame = frame.dropna(subset=["title", "date"]).copy()
            frame["title"] = frame["title"].str.strip()
            frame["date"] = frame["date"].str.strip()
            frame = frame[frame["title"].ne("")].drop_duplicates(subset=["date", "title"])
            frame["timestamp"] = pd.to_datetime(frame["date"], errors="coerce", utc=True, format="mixed")
            frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp", kind="stable")
            # Drop future-dated articles and count date-only inputs.
            _future = frame["timestamp"] > _now
            counts["future_dropped"] += int(_future.sum())
            frame = frame[~_future]
            _date_only = frame["date"].str.len() == 10
            counts["date_only"] += int(_date_only.sum())
            counts["timestamped"] += int((~_date_only).sum())
            frame["day"] = frame["timestamp"].dt.strftime("%Y-%m-%d")
            frame = frame.groupby("day", group_keys=False).head(DAILY_ARTICLE_CAP)
            articles = []
            for _, row in frame.iterrows():
                summary = row["summary"].strip()
                text = row["title"] + (". " + summary if summary else "")
                # Preserve date-only values so the scorer can apply conservative timing.
                article = {"title": row["title"], "content": summary, "published_at": row["date"]}
                articles.append(article)
                stream.write(json.dumps({"ticker": path.stem, "published_at": row["date"], "text": text}) + "\n")
            destination = news_dir / f"{path.stem}.json"
            destination.write_text(
                json.dumps({"ticker": path.stem, "count": len(articles), "source": "FNSPID", "articles": articles})
            )
            counts["tickers"] += 1
            counts["output_articles"] += len(articles)
    return counts


if __name__ == "__main__":
    print(json.dumps(build_inputs(), indent=2))
