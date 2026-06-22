import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.data.cleaner import DataCleaner
from backend.data.feature_engineer import FeatureEngineer

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")


def load_universe():
    files = sorted(Path("data/universe").glob("*.json"), key=lambda f: f.stat().st_mtime)
    if not files:
        print("No universe file found — run fetch_market_data.py first")
        sys.exit(1)
    data = json.loads(files[-1].read_text())
    print(f"Universe: {data['market']} / {data['index']} ({data['count']} stocks)")
    return data["tickers"]


def main():
    tickers = load_universe()

    print("\nCleaning raw data...")
    cleaner = DataCleaner()
    cl = cleaner.clean_universe(tickers)
    print(f"  Cleaned: {len(cl['success'])}  Failed: {len(cl['failed'])}")

    print("\nBuilding technical indicators...")
    engineer = FeatureEngineer()
    fe = engineer.generate_universe(cl["success"])
    print(f"  Features generated: {len(fe['success'])}")

    if fe["success"]:
        sample = fe["success"][0]
        df = engineer.load(sample)
        print(f"\nSample ({sample}) — last row:")
        cols = ["close", "rsi", "macd", "sma20", "sma50", "momentum", "daily_return"]
        print(df[[c for c in cols if c in df.columns]].tail(1).to_string())

    print(f"\nSaved to data/processed/ and data/features/")
    print("Next: python scripts/run_advisor.py")


if __name__ == "__main__":
    main()
