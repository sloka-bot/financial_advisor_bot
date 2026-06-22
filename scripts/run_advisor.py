import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.universe.universe_builder import UniverseBuilder
from backend.data.downloader import MarketDataDownloader
from backend.data.cleaner import DataCleaner
from backend.data.feature_engineer import FeatureEngineer
from backend.data.fusion import FeatureFusion
from backend.news.news_collector import NewsCollector
from backend.news.sentiment_analyzer import SentimentAnalyzer
from backend.models.xgboost_model import XGBoostForecaster
from backend.models.lstm_model import LSTMForecaster
from backend.engine.ranker import StockRanker
from backend.engine.recommender import RecommendationEngine
from backend.engine.explainer import Explainer
from backend.engine.portfolio import PortfolioConstructor

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s', datefmt='%H:%M:%S')


def ask(prompt, options):
    print(f'\n{prompt}')
    for i, o in enumerate(options, 1):
        print(f'  {i}. {o}')
    while True:
        r = input('  > ').strip()
        if r.isdigit() and 1 <= int(r) <= len(options):
            return options[int(r) - 1]
        print('  Enter a number from the list')


def main():
    print('\nAI Financial Advisor')
    print('-' * 40)

    builder = UniverseBuilder()
    markets = builder.list_markets()
    market  = ask('Market:', list(markets.keys()))
    index   = ask(f'Index ({market}):', markets[market])

    print(f'\nFetching {index} constituents...')
    tickers = builder.get_universe(market, index)
    print(f'  {len(tickers)} stocks')

    scope = ask('Scope?', [
        f'Full universe ({len(tickers)} stocks)',
        'Sample — 10 stocks (quick test)',
    ])
    target = tickers if 'Full' in scope else tickers[:10]

    print('\nDownloading price history...')
    dl      = MarketDataDownloader()
    results = dl.download_universe(target)
    print(f'  {len(results["success"])} downloaded, {len(results["skipped"])} cached, {len(results["failed"])} failed')

    print('\nCleaning data...')
    c  = DataCleaner()
    cl = c.clean_universe(target)

    print('\nGenerating technical indicators...')
    fe       = FeatureEngineer()
    features = fe.generate_universe(cl['success'])

    print('\nCollecting news...')
    nc = NewsCollector()
    nr = nc.get_universe_news(target)
    print(f'  {len(nr["success"])} tickers with news')

    print('\nRunning FinBERT sentiment analysis...')
    sa = SentimentAnalyzer()
    sa.analyze_universe(target)

    print('\nMerging features and sentiment...')
    fuser = FeatureFusion()
    fused = fuser.fuse_universe(features['success'])
    print(f'  {len(fused["success"])} master datasets created')

    combined = fuser.load_all(fused['success'])
    if combined is None or combined.empty:
        print('No master data found — check data pipeline')
        sys.exit(1)

    print(f'\nTraining XGBoost on {len(combined)} rows...')
    xgb   = XGBoostForecaster()
    xgb_r = xgb.train(combined)
    print(f'  CV AUC={xgb_r.get("cv_auc_mean")}  DirAcc={xgb_r.get("cv_direction_acc")}')

    print('\nTraining LSTM...')
    lstm   = LSTMForecaster()
    lstm_r = lstm.train(combined, epochs=50)
    print(f'  val_loss={lstm_r.get("best_val_loss")}  DirAcc={lstm_r.get("best_dir_acc")}')

    risk = ask('\nRisk profile:', ['conservative', 'moderate', 'aggressive'])

    budget_raw = input('\nInvestment budget in USD (e.g. 10000): $').strip()
    budget     = float(budget_raw) if budget_raw.replace('.','').isdigit() else 10000.0

    top_n_raw = input('Number of stock picks (e.g. 5): ').strip()
    top_n     = int(top_n_raw) if top_n_raw.isdigit() else 5

    print('\nGenerating predictions...')
    predictions = {}
    master_data = {}

    for ticker in fused['success']:
        df = fuser.load_master(ticker)
        if df is None or df.empty:
            continue
        master_data[ticker] = df
        preds = []
        p = xgb.predict_ticker(df)
        if p is not None: preds.append(p)
        p = lstm.predict_ticker(df)
        if p is not None: preds.append(p)
        if preds:
            predictions[ticker] = sum(preds) / len(preds)

    print(f'  Predictions for {len(predictions)} stocks')

    rk     = StockRanker()
    rc     = RecommendationEngine()
    ex     = Explainer()
    pc     = PortfolioConstructor()

    # bug fixes: pass risk_profile to rank(), pass correct args to construct()
    ranked    = rk.rank(predictions, master_data, risk)
    result    = rc.recommend(ranked, risk_profile=risk, top_n=top_n)
    portfolio = pc.construct(ranked, master_data, budget, risk, top_n)
    ex.explain_batch(result['top_picks'], master_data)

    print(f'\n{"="*50}')
    print(f'  {risk.title()} profile  |  ${budget:,.0f}')
    print(f'{"="*50}\n')

    for rec in result['top_picks']:
        print(f'  {rec["ticker"]:6s}  {rec["signal"]:4s}  '
              f'score={rec["composite_score"]:.0f}  '
              f'pred={rec["predicted_return"]:+.2f}%')
        if rec.get('explanation'):
            print(f'         {rec["explanation"][:110]}')
        print()

    s = result['summary']
    print(f'  BUY {s["buy_count"]}  |  HOLD {s["hold_count"]}  |  SELL {s["sell_count"]}\n')

    p = portfolio['portfolio']
    print(f'  Portfolio (${budget:,.0f} | {risk}):')
    print(f'  {"Ticker":<8} {"Shares":>6}  {"Price":>8}  {"Cost":>10}  {"Weight":>7}')
    print(f'  {"-"*46}')
    for h in p['holdings']:
        print(f'  {h["ticker"]:<8} {h["shares"]:>6}  ${h["price"]:>7.2f}  ${h["total_cost"]:>9,.2f}  {h["weight_pct"]:>6.1f}%')
    print(f'  {"-"*46}')
    print(f'  Total invested : ${p["total_invested"]:>10,.2f}')
    print(f'  Cash remaining : ${p["cash_remaining"]:>10,.2f}')
    print(f'  Expected return: {p["expected_portfolio_return"]:>+9.2f}%\n')
    print('  Open frontend/index.html in Chrome to see the dashboard\n')


if __name__ == '__main__':
    main()
