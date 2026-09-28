"""Score historical FNSPID news with FinBERT into per-ticker session sentiment files."""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.news.sentiment_analyzer import SentimentAnalyzer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="score only the first N tickers")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    news_dir = Path("data/news/historical")
    tickers = sorted(p.stem for p in news_dir.glob("*.json"))
    if args.limit:
        tickers = tickers[: args.limit]
    analyzer = SentimentAnalyzer(news_dir=news_dir, sentiment_dir="data/sentiment/historical")
    result = analyzer.analyze_universe(tickers)
    print(json.dumps({key: len(value) for key, value in result.items() if isinstance(value, list)}))


if __name__ == "__main__":
    main()
