"""Interactive command-line walkthrough of the full pipeline using session-only models."""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.data.cleaner import DataCleaner
from backend.data.downloader import MarketDataDownloader
from backend.data.feature_engineer import FeatureEngineer
from backend.data.fusion import FeatureFusion
from backend.explain.explainer import Explainer
from backend.news.news_collector import NewsCollector
from backend.news.sentiment_analyzer import SentimentAnalyzer
from backend.portfolio.allocation import build_markowitz_portfolio
from backend.prediction.lstm_model import LSTMForecaster
from backend.prediction.ranker import StockRanker
from backend.prediction.recommender import RecommendationEngine
from backend.prediction.xgboost_model import XGBoostForecaster
from backend.universe.universe_builder import UniverseBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")


def ask(prompt, options):
    print(f"\n{prompt}")
    for i, o in enumerate(options, 1):
        print(f"  {i}. {o}")
    while True:
        r = input("  > ").strip()
        if r.isdigit() and 1 <= int(r) <= len(options):
            return options[int(r) - 1]
        print("  Enter a number from the list")


def main():
    from backend.infra.reproducibility import set_seed

    set_seed(42)
    print("\nAI Financial Advisor")
    print("-" * 40)

    builder = UniverseBuilder()
    markets = builder.list_markets()
    market = ask("Market:", list(markets.keys()))
    index = ask(f"Index ({market}):", markets[market])

    print(f"\nFetching {index} constituents...")
    import pandas as pd

    from backend.config.settings import HISTORY_START

    tickers = builder.eligible_between(HISTORY_START, str(pd.Timestamp.today().date()))
    print(f"  {len(tickers)} stocks")

    scope = ask(
        "Scope?",
        [
            f"Full universe ({len(tickers)} stocks)",
            "Sample - 10 stocks (quick test)",
        ],
    )
    target = tickers if "Full" in scope else tickers[:10]

    print("\nDownloading price history...")
    dl = MarketDataDownloader()
    results = dl.download_universe_incremental(target)
    print(
        f"  {len(results['success'])} downloaded, "
        f"{len(results.get('skipped', []))} cached, {len(results['failed'])} failed"
    )

    print("\nCleaning data...")
    c = DataCleaner()
    cl = c.clean_universe(target)

    print("\nGenerating technical indicators...")
    fe = FeatureEngineer()
    features = fe.generate_universe(cl["success"])

    print("\nCollecting news...")
    nc = NewsCollector()
    nr = nc.get_universe_news(target)
    print(f"  {len(nr['success'])} tickers with news")

    print("\nRunning FinBERT sentiment analysis...")
    sa = SentimentAnalyzer()
    sa.analyze_universe(target)

    print("\nMerging features and sentiment...")
    fuser = FeatureFusion()
    fused = fuser.fuse_universe(features["success"])
    print(f"  {len(fused['success'])} master datasets created")

    combined = fuser.load_all(fused["success"])
    if combined is None or combined.empty:
        print("No master data found - check data pipeline")
        sys.exit(1)

    # Keep rows only for dates each stock was an S&P 500 member.
    combined = builder.filter_eligible_rows(combined, strict=True)

    print(f"\nTraining XGBoost on {len(combined)} rows...")
    # Session-only model that is not registered.
    xgb = XGBoostForecaster(models_dir="models/_session/xgboost")
    xgb_r = xgb.train(combined, register=False)
    print(f"  CV AUC={xgb_r.get('cv_auc_mean')}  DirAcc={xgb_r.get('cv_direction_acc')}")

    print("\nTraining LSTM...")
    # Session-only model in a temporary directory.
    lstm = LSTMForecaster(models_dir="models/_session/lstm")
    lstm_r = lstm.train(combined, epochs=50, batch_size=256)
    print(f"  val_loss={lstm_r.get('best_val_loss')}  DirAcc={lstm_r.get('best_dir_acc')}")

    risk = ask("\nRisk profile:", ["conservative", "moderate", "aggressive"])

    budget_raw = input("\nInvestment budget in USD (e.g. 10000): $").strip()
    budget = float(budget_raw) if budget_raw.replace(".", "").isdigit() else 10000.0

    top_n_raw = input("Number of stock picks (e.g. 5): ").strip()
    top_n = int(top_n_raw) if top_n_raw.isdigit() else 5

    print("\nGenerating predictions...")
    predictions = {}
    master_data = {}

    for ticker in fused["success"]:
        df = fuser.load_master(ticker)
        if df is None or df.empty:
            continue
        master_data[ticker] = df
        prob_up = xgb.predict_proba_up(df) if hasattr(xgb, "predict_proba_up") else None
        exp_ret = lstm.predict_ticker(df)
        if prob_up is None and exp_ret is None:
            continue
        predictions[ticker] = {
            "prob_up": prob_up,
            "expected_return": exp_ret,
            "horizon": 21,
            "xgb_available": prob_up is not None,
            "lstm_available": exp_ret is not None,
        }

    print(f"  Predictions for {len(predictions)} stocks")

    rk = StockRanker()
    rc = RecommendationEngine()
    ex = Explainer()

    # Rank with the selected risk profile.
    ranked = rk.rank(predictions, master_data, risk)
    result = rc.recommend(ranked, risk_profile=risk, top_n=top_n)
    _mlm = (
        {
            r["ticker"]: {
                "predicted_return": r.get("predicted_return"),
                "sentiment": r.get("sentiment"),
                "composite_score": r.get("composite_score"),
            }
            for r in ranked.to_dict("records")
        }
        if hasattr(ranked, "to_dict")
        else {}
    )
    portfolio = build_markowitz_portfolio(master_data, list(master_data.keys()), budget, risk, ml_meta=_mlm)
    ex.explain_batch(result["top_picks"], master_data)

    print(f"\n{'=' * 50}")
    print(f"  {risk.title()} profile  |  ${budget:,.0f}")
    print(f"{'=' * 50}\n")

    for rec in result["top_picks"]:
        print(
            f"  {rec['ticker']:6s}  {rec['signal']:4s}  "
            f"score={rec['composite_score']:.0f}  "
            f"pred={rec['predicted_return']:+.2f}%"
        )
        if rec.get("explanation"):
            print(f"         {rec['explanation'][:110]}")
        print()

    s = result["summary"]
    print(f"  BUY {s['buy_count']}  |  HOLD {s['hold_count']}  |  SELL {s['sell_count']}\n")

    p = portfolio["portfolio"]
    print(f"  Portfolio (${budget:,.0f} | {risk}):")
    print(f"  {'Ticker':<8} {'Shares':>6}  {'Price':>8}  {'Cost':>10}  {'Weight':>7}")
    print(f"  {'-' * 46}")
    for h in p["holdings"]:
        print(
            f"  {h['ticker']:<8} {h['shares']:>6}  ${h['price']:>7.2f}  "
            f"${h['total_cost']:>9,.2f}  {h['weight_pct']:>6.1f}%"
        )
    print(f"  {'-' * 46}")
    print(f"  Total invested : ${p['total_invested']:>10,.2f}")
    print(f"  Cash remaining : ${p['cash_remaining']:>10,.2f}")
    print(f"  Expected return: {p['expected_portfolio_return']:>+9.2f}%\n")
    print("  Open frontend/index.html in Chrome to see the dashboard\n")


if __name__ == "__main__":
    main()
