import json
import logging
import sys
import threading
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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
from backend.engine.backtester import Backtester

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

app = FastAPI(title='AI Financial Advisor', version='2.0.0')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

universe_builder = UniverseBuilder()
downloader       = MarketDataDownloader()
cleaner          = DataCleaner()
engineer         = FeatureEngineer()
fusion           = FeatureFusion()
news_collector   = NewsCollector()
sentiment        = SentimentAnalyzer()
xgb_model        = XGBoostForecaster()
lstm_model       = LSTMForecaster()
ranker           = StockRanker()
recommender      = RecommendationEngine()
explainer        = Explainer()
constructor      = PortfolioConstructor()
backtester       = Backtester()

# shared pipeline status — read by /api/pipeline-status so the frontend
# can poll progress without the request blocking
_status = {'step': 'idle', 'message': '', 'progress': 0, 'error': None}
_lock   = threading.Lock()

def _set(step, message, progress=0, error=None):
    with _lock:
        _status.update({'step': step, 'message': message, 'progress': progress, 'error': error})


# ── request models ──────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    market:       str
    index:        str
    risk_profile: str   = 'moderate'
    budget:       float = 10000.0
    top_n:        int   = 10
    scope:        str   = 'all'   # 'all' or 'sample'

class BacktestRequest(BaseModel):
    tickers:        list[str]
    capital:        float = 10000.0
    portfolio_mode: bool  = False

class ChatRequest(BaseModel):
    message:   str
    portfolio: dict | None = None


# ── health ──────────────────────────────────────────────────────────────────

@app.get('/')
def health():
    return {'status': 'ok'}


# ── markets ─────────────────────────────────────────────────────────────────

@app.get('/api/markets')
def get_markets():
    return {'markets': universe_builder.list_markets()}


@app.get('/api/universe/{market}/{index}')
def get_universe(market: str, index: str, refresh: bool = False):
    market = market.replace('_', ' ')
    index  = index.replace('_', ' ')
    try:
        tickers = universe_builder.get_universe(market, index, force_refresh=refresh)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(404, str(e))
    return {'market': market, 'index': index, 'count': len(tickers), 'tickers': tickers}


# ── FULL PIPELINE (single call from the frontend Run button) ────────────────

@app.post('/api/run')
def run_full_pipeline(req: RunRequest, bg: BackgroundTasks):
    """
    Kicks off the complete pipeline in a background thread.
    The frontend polls /api/pipeline-status to track progress.
    Returns immediately so the browser doesn't time out.
    """
    bg.add_task(_pipeline_task, req)
    return {'message': 'Pipeline started', 'poll': '/api/pipeline-status'}


def _pipeline_task(req: RunRequest):
    try:
        _set('fetching', 'Fetching universe tickers…', 2)
        tickers = universe_builder.get_universe(req.market, req.index)
        target  = tickers if req.scope == 'all' else tickers[:10]
        logger.info(f'Pipeline: {len(target)} tickers — {req.market} / {req.index}')

        _set('downloading', f'Downloading price history for {len(target)} stocks…', 10)
        downloader.download_universe(target)

        _set('cleaning', 'Cleaning data…', 25)
        cleaned = cleaner.clean_universe(target)

        _set('features', 'Building 38 technical indicators…', 35)
        feat = engineer.generate_universe(cleaned.get('success', target))

        _set('news', 'Collecting news headlines…', 50)
        news_collector.get_universe_news(target)

        _set('sentiment', 'Running FinBERT sentiment analysis…', 60)
        sentiment.analyze_universe(target)

        _set('fusion', 'Merging features and sentiment…', 70)
        fused = fusion.fuse_universe(feat.get('success', target))

        combined = fusion.load_all(fused.get('success', target))
        if combined is None or combined.empty:
            _set('error', 'No fused data available — check data pipeline', error='no_data')
            return

        _set('xgboost', 'Training XGBoost (5-fold cross-validation)…', 78)
        xgb_model.train(combined)

        _set('lstm', 'Training LSTM (sequence model)…', 88)
        lstm_model.train(combined, epochs=50)

        _set('done', 'Pipeline complete — ready to analyse', 100)
        logger.info('Pipeline complete')

    except Exception as e:
        logger.error(f'Pipeline error: {e}', exc_info=True)
        _set('error', str(e), error=str(e))


@app.get('/api/pipeline-status')
def pipeline_status():
    with _lock:
        return dict(_status)


# ── individual pipeline steps (kept for compatibility) ──────────────────────

@app.post('/api/pipeline/download')
def start_download(req: dict, bg: BackgroundTasks):
    market = req.get('market', '')
    index  = req.get('index', '')
    tickers = universe_builder.get_universe(market, index)
    bg.add_task(lambda: (downloader.download_universe(tickers),
                         cleaner.clean_universe(tickers),
                         engineer.generate_universe(tickers)))
    return {'message': 'Download started', 'count': len(tickers)}


@app.post('/api/pipeline/news')
def start_news(req: dict, bg: BackgroundTasks):
    market = req.get('market', '')
    index  = req.get('index', '')
    tickers = universe_builder.get_universe(market, index)
    bg.add_task(lambda: (news_collector.get_universe_news(tickers),
                         sentiment.analyze_universe(tickers),
                         fusion.fuse_universe(tickers)))
    return {'message': 'News pipeline started'}


@app.post('/api/train')
def start_training(req: dict, bg: BackgroundTasks):
    market = req.get('market', '')
    index  = req.get('index', '')
    tickers = universe_builder.get_universe(market, index)
    def run():
        combined = fusion.load_all(tickers)
        if combined is not None:
            xgb_model.train(combined)
            lstm_model.train(combined)
    bg.add_task(run)
    return {'message': 'Training started'}


# ── recommendations + portfolio ─────────────────────────────────────────────

@app.post('/api/recommend')
def get_recommendations(req: dict):
    market       = req.get('market', '')
    index        = req.get('index', '')
    risk_profile = req.get('risk_profile', 'moderate')
    top_n        = req.get('top_n', 10)

    if not xgb_model.is_trained() and not lstm_model.is_trained():
        raise HTTPException(400, 'Models not trained yet — run the pipeline first')

    tickers = universe_builder.get_universe(market, index)
    predictions, master_data = _run_inference(tickers)
    if not predictions:
        raise HTTPException(422, 'No predictions — run the full pipeline first')

    ranked = ranker.rank(predictions, master_data, risk_profile)
    result = recommender.recommend(ranked, risk_profile=risk_profile, top_n=top_n)
    explainer.explain_batch(result.get('top_picks', []), master_data)
    explainer.explain_batch(result.get('all_signals', []), master_data)
    return {'market': market, 'index': index, **result}


@app.post('/api/portfolio')
def get_portfolio(req: dict):
    market       = req.get('market', '')
    index        = req.get('index', '')
    risk_profile = req.get('risk_profile', 'moderate')
    budget       = float(req.get('budget', 10000))
    top_n        = req.get('top_n', 10)

    if not xgb_model.is_trained() and not lstm_model.is_trained():
        raise HTTPException(400, 'Models not trained yet')

    tickers = universe_builder.get_universe(market, index)
    predictions, master_data = _run_inference(tickers)
    if not predictions:
        raise HTTPException(422, 'No predictions')

    ranked    = ranker.rank(predictions, master_data, risk_profile)
    result    = recommender.recommend(ranked, risk_profile=risk_profile, top_n=top_n)
    portfolio = constructor.construct(ranked, master_data, budget, risk_profile, top_n)
    explainer.explain_batch(result.get('all_signals', []), master_data)

    return {
        'market': market, 'index': index, 'risk_profile': risk_profile, 'budget': budget,
        **portfolio, 'signals': result.get('summary', {}),
        'all_signals': result.get('all_signals', []),
        'top_picks':   result.get('top_picks', []),
    }


# ── backtest + evaluate ──────────────────────────────────────────────────────

@app.post('/api/backtest')
def run_backtest(req: BacktestRequest):
    if not xgb_model.is_trained():
        raise HTTPException(400, 'XGBoost not trained yet')
    master_data = {t: df for t in req.tickers
                   if (df := fusion.load_master(t)) is not None and not df.empty}
    if not master_data:
        raise HTTPException(404, 'No data found')
    if req.portfolio_mode or len(req.tickers) > 1:
        return backtester.run_portfolio(req.tickers, master_data, xgb_model, req.capital)
    ticker = req.tickers[0]
    return backtester.run(ticker, master_data[ticker], xgb_model, req.capital)


@app.get('/api/evaluate/{ticker}')
def evaluate_ticker(ticker: str):
    df = fusion.load_master(ticker)
    if df is None:
        raise HTTPException(404, f'No data for {ticker}')
    return backtester.run(ticker, df, xgb_model)


# ── stock data ───────────────────────────────────────────────────────────────

@app.get('/api/stock/{ticker}')
def get_stock(ticker: str):
    df = fusion.load_master(ticker)
    if df is None or df.empty:
        raise HTTPException(404, f'{ticker} not found')
    latest    = df.iloc[-1]
    pred_xgb  = xgb_model.predict_ticker(df)  if xgb_model.is_trained()  else None
    pred_lstm = lstm_model.predict_ticker(df)  if lstm_model.is_trained() else None
    preds     = [p for p in [pred_xgb, pred_lstm] if p is not None]
    return {
        'ticker': ticker,
        'close':  round(float(latest.get('close', 0)), 2),
        'predicted_return_avg': round(sum(preds)/len(preds)*100, 3) if preds else None,
    }


@app.get('/api/stock/{ticker}/history')
def stock_history(ticker: str, days: int = 120):
    df = fusion.load_master(ticker) or engineer.load(ticker)
    if df is None or df.empty:
        raise HTTPException(404, f'No data for {ticker}')
    df = df.tail(days).copy()

    def safe(v):
        try:
            f = float(v)
            return None if f != f else round(f, 6)
        except Exception:
            return None

    rows = []
    for date, row in df.iterrows():
        rows.append({
            'date':            str(date.date()) if hasattr(date, 'date') else str(date),
            'open':  safe(row.get('open')),  'high': safe(row.get('high')),
            'low':   safe(row.get('low')),   'close': safe(row.get('close')),
            'volume': int(row.get('volume', 0)) if pd.notna(row.get('volume')) else 0,
            'sma20': safe(row.get('sma20')), 'sma50': safe(row.get('sma50')),
            'bb_upper': safe(row.get('bb_upper')), 'bb_lower': safe(row.get('bb_lower')),
            'rsi':  safe(row.get('rsi')),    'macd': safe(row.get('macd')),
            'macd_signal': safe(row.get('macd_signal')), 'macd_hist': safe(row.get('macd_hist')),
            'volume_ratio': safe(row.get('volume_ratio')), 'adx': safe(row.get('adx')),
            'week52_position': safe(row.get('week52_position')),
            'sent_score': safe(row.get('sent_score')),
        })
    return {'ticker': ticker, 'days': len(rows), 'history': rows}


@app.get('/api/analysis/{ticker}')
def full_analysis(ticker: str):
    df = fusion.load_master(ticker) or engineer.load(ticker)
    if df is None or df.empty:
        raise HTTPException(404, f'No data for {ticker}')
    latest    = df.iloc[-1]
    pred_xgb  = xgb_model.predict_ticker(df)  if xgb_model.is_trained()  else None
    pred_lstm = lstm_model.predict_ticker(df)  if lstm_model.is_trained() else None
    preds     = [p for p in [pred_xgb, pred_lstm] if p is not None]
    avg_pred  = sum(preds)/len(preds) if preds else 0.0
    vol       = float(latest.get('volatility', 0.02) or 0.02)
    score     = min(100, max(0, 50 + (avg_pred / (vol + 1e-6)) * 1000))
    mom       = float(latest.get('momentum_10d', 0) or 0)
    adx       = float(latest.get('adx', 20) or 20)
    regime    = ('trending_up'   if adx > 25 and mom > 0 else
                 'trending_down' if adx > 25 and mom < 0 else 'ranging')
    return {
        'ticker': ticker, 'signal': 'BUY' if score>=65 else ('SELL' if score<=35 else 'HOLD'),
        'score': round(score,1), 'regime': regime,
        'close': round(float(latest.get('close',0) or 0), 2),
        'prediction': {
            'xgboost': round(pred_xgb*100,3) if pred_xgb else None,
            'lstm':    round(pred_lstm*100,3) if pred_lstm else None,
            'ensemble': round(avg_pred*100,3),
        },
        'indicators': {
            'rsi': round(float(latest.get('rsi',50) or 50), 2),
            'adx': round(adx, 2),
            'macd_hist': round(float(latest.get('macd_hist',0) or 0), 4),
            'bb_pct':    round(float(latest.get('bb_pct',0.5) or 0.5), 4),
            'atr_pct':   round(float(latest.get('atr_pct',0) or 0), 4),
            'stoch_k':   round(float(latest.get('stoch_k',50) or 50), 2),
            'week52_pos':round(float(latest.get('week52_position',0.5) or 0.5), 4),
            'volume_ratio': round(float(latest.get('volume_ratio',1) or 1), 3),
            'momentum_10d': round(mom*100, 2),
            'volatility':   round(vol, 4),
        },
        'sentiment': {
            'score': round(float(latest.get('sent_score',0) or 0), 4),
            'label': str(latest.get('sent_label','neutral') or 'neutral'),
            'count': int(latest.get('sent_news_count',0) or 0),
        },
        'top_features': dict(list(xgb_model.feature_importance().items())[:8]) if xgb_model.is_trained() else {},
    }


@app.get('/api/model-status')
def model_status():
    return {
        'xgboost_trained': xgb_model.is_trained(),
        'lstm_trained':    lstm_model.is_trained(),
    }


# ── ollama + chat ────────────────────────────────────────────────────────────

@app.get('/api/ollama-status')
def ollama_status():
    available = explainer.ollama_available()
    models    = []
    if available:
        try:
            r = requests.get('http://localhost:11434/api/tags', timeout=2)
            models = [m['name'] for m in r.json().get('models', [])]
        except Exception:
            pass
    return {'ollama_available': available, 'loaded_models': models}


@app.post('/api/chat')
def chat(req: ChatRequest):
    p        = (req.portfolio or {}).get('portfolio', {})
    holdings = p.get('holdings', [])
    rm       = p.get('risk_metrics', {})
    ctx      = (
        f'Portfolio: {p.get("n_positions",0)} positions, '
        f'${p.get("total_invested",0):,.0f} invested, '
        f'expected return {p.get("expected_portfolio_return",0):+.2f}%\n'
        f'Risk: Sharpe {rm.get("annualized_sharpe",0):.2f}, '
        f'VaR {rm.get("var_95_1day",0)*100:.2f}%\n'
        f'Holdings: ' + ', '.join(f'{h["ticker"]} ({h["signal"]})' for h in holdings[:5])
    )
    prompt = (
        'You are a professional AI financial advisor. Answer in 3-4 sentences. '
        'Be specific about numbers. Do not start with "I".\n\n'
        f'{ctx}\n\nQuestion: {req.message}\nAnswer:'
    )
    try:
        r = requests.post('http://localhost:11434/api/generate',
                          json={'model':'llama3.2','prompt':prompt,'stream':False,
                                'options':{'temperature':0.3,'num_predict':300}}, timeout=12)
        if r.status_code == 200:
            reply = r.json().get('response','').strip()
            if reply:
                return {'reply': reply, 'source': 'ollama'}
    except Exception as e:
        logger.debug(f'Ollama unavailable: {e}')

    # template fallback
    msg = req.message.lower()
    if any(w in msg for w in ['sharpe','var','risk','cvar']):
        return {'reply': f'Sharpe: {rm.get("annualized_sharpe",0):.2f}. VaR: {rm.get("var_95_1day",0)*100:.2f}%.', 'source':'template'}
    if any(w in msg for w in ['buy','best','pick']):
        buys = [h for h in holdings if h.get('signal')=='BUY'][:3]
        return {'reply': 'Top BUY: ' + ', '.join(f'{h["ticker"]} ({h.get("predicted_return",0):+.2f}%)' for h in buys) + '.', 'source':'template'}
    return {'reply': 'Ask me about signals, risk metrics, or how the models work.', 'source':'template'}


# ── validation ───────────────────────────────────────────────────────────────

@app.get('/api/validation-results')
def validation_results():
    path = Path('data/validation_results.json')
    if not path.exists():
        return {'error': 'No results yet — run scripts/evaluate_models.py'}
    return json.loads(path.read_text())


@app.post('/api/evaluate-all')
def run_evaluation(bg: BackgroundTasks):
    import subprocess
    bg.add_task(lambda: subprocess.run([sys.executable, 'scripts/evaluate_models.py'], check=False))
    return {'status': 'running'}


# ── helper ───────────────────────────────────────────────────────────────────

def _run_inference(tickers):
    predictions, master_data = {}, {}
    for ticker in tickers:
        df = fusion.load_master(ticker)
        if df is None or df.empty:
            continue
        master_data[ticker] = df
        preds = []
        if xgb_model.is_trained():
            p = xgb_model.predict_ticker(df)
            if p is not None: preds.append(p)
        if lstm_model.is_trained():
            p = lstm_model.predict_ticker(df)
            if p is not None: preds.append(p)
        if preds:
            predictions[ticker] = sum(preds)/len(preds)
    return predictions, master_data
