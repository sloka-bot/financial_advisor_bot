import json
import logging
import os
import sys
import threading
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.universe.universe_builder import UniverseBuilder
from backend.data.downloader import MarketDataDownloader
from backend.data.cleaner import DataCleaner
from backend.data.feature_engineer import FeatureEngineer
from backend.data.fusion import FeatureFusion
from backend.data.user_store import UserStore
from backend.news.news_collector import NewsCollector
from backend.news.sentiment_analyzer import SentimentAnalyzer
from backend.models.xgboost_model import XGBoostForecaster
from backend.models.lstm_model import LSTMForecaster
from backend.engine.ranker import StockRanker
from backend.engine.recommender import RecommendationEngine
from backend.engine.explainer import Explainer
from backend.engine.portfolio import PortfolioConstructor
from backend.engine.portfolio_manager import PortfolioManager
from backend.engine.backtester import Backtester
from backend.engine.regime_detector import RegimeDetector

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

app = FastAPI(title='AI Financial Advisor', version='3.0.0')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['*'], allow_headers=['*'])

universe_builder  = UniverseBuilder()
downloader        = MarketDataDownloader()
cleaner           = DataCleaner()
engineer          = FeatureEngineer()
fusion            = FeatureFusion()
user_store        = UserStore()
news_collector    = NewsCollector()
sentiment         = SentimentAnalyzer()
xgb_model         = XGBoostForecaster()
lstm_model        = LSTMForecaster()
ranker            = StockRanker()
recommender       = RecommendationEngine()
explainer         = Explainer()
constructor       = PortfolioConstructor()
portfolio_manager = PortfolioManager()
backtester        = Backtester()
regime_detector   = RegimeDetector()

_status = {'step': 'idle', 'message': '', 'progress': 0, 'error': None, 'processed_tickers': []}
_lock   = threading.Lock()

def _set(step, message, progress=0, error=None, tickers=None):
    with _lock:
        _status.update({'step': step, 'message': message, 'progress': progress, 'error': error})
        if tickers is not None:
            _status['processed_tickers'] = tickers


# ── Request models ──────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    market:       str   = 'United States'
    index:        str   = 'NASDAQ 100'
    risk_profile: str   = 'moderate'
    budget:       float = 10000.0
    top_n:        int   = 10
    scope:        str   = 'sample'
    user_id:      str   = ''

class ChatRequest(BaseModel):
    message:   str
    portfolio: dict | None = None
    context:   str  | None = None

class ProfileRequest(BaseModel):
    name:                   str   = ''
    risk_profile:           str   = 'moderate'
    investment_horizon:     str   = '5-10 years'
    goal:                   str   = 'growth'
    budget:                 float = 10000.0
    monthly_contribution:   float = 0.0

class ApprovalRequest(BaseModel):
    user_id: str
    rec_id:  int


# ── Config ──────────────────────────────────────────────────────────────────

@app.get('/')
def health():
    return {'status': 'ok', 'version': '3.0.0'}

@app.get('/api/config')
def get_config():
    return {'clerk_key': os.getenv('CLERK_PUBLISHABLE_KEY', '')}


# ── Markets ─────────────────────────────────────────────────────────────────

@app.get('/api/markets')
def get_markets():
    """
    Only US equities are fully supported in this prototype.
    Other markets are shown as future work.
    """
    return {
        'markets': {
            'United States': ['NASDAQ 100', 'S&P 500', 'Dow Jones'],
        },
        'future_markets': ['SGX Singapore', 'NSE India', 'LSE United Kingdom'],
        'note': 'Global market support is planned for a future release.'
    }

@app.get('/api/universe/{market}/{index}')
def get_universe(market: str, index: str, refresh: bool = False):
    market = market.replace('_', ' ')
    index  = index.replace('_', ' ')
    try:
        tickers = universe_builder.get_universe(market, index, force_refresh=refresh)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(404, str(e))
    return {'market': market, 'index': index, 'count': len(tickers), 'tickers': tickers}


# ── User profiles ────────────────────────────────────────────────────────────

@app.get('/api/user/{user_id}')
def get_profile(user_id: str):
    profile = user_store.get(user_id)
    if not profile:
        return {'exists': False, 'is_new': True}
    return {'exists': True, 'is_new': False, **profile}

@app.post('/api/user/{user_id}')
def create_profile(user_id: str, req: ProfileRequest):
    profile = user_store.create_profile(user_id, req.dict())
    return {'created': True, **profile}

@app.put('/api/user/{user_id}')
def update_profile(user_id: str, req: ProfileRequest):
    profile = user_store.update_profile(user_id, req.dict())
    return {'updated': True, **profile}

@app.get('/api/user/{user_id}/portfolio')
def get_user_portfolio(user_id: str):
    profile = user_store.get(user_id)
    if not profile:
        raise HTTPException(404, 'User not found')
    return profile.get('portfolio', {'holdings': [], 'cash': 0})

@app.get('/api/user/{user_id}/recommendations')
def get_recommendations(user_id: str):
    profile = user_store.get(user_id)
    if not profile:
        return {'pending': [], 'history': []}
    return {
        'pending': profile.get('pending_recommendations', []),
        'history': profile.get('recommendation_history', [])[-20:],
    }

@app.post('/api/recommendations/approve')
def approve_recommendation(req: ApprovalRequest):
    rec = user_store.approve_recommendation(req.user_id, req.rec_id)
    if not rec:
        raise HTTPException(404, 'Recommendation not found')

    # apply the change to the portfolio
    profile  = user_store.get(req.user_id)
    port     = profile.get('portfolio', {})
    holdings = port.get('holdings', [])
    cash     = port.get('cash', 0)

    with _lock:
        tickers = list(_status.get('processed_tickers', []))
    master_data = {t: df for t in tickers
                   if (df := fusion.load_master(t)) is not None and not df.empty}

    result = portfolio_manager.apply_recommendation(rec, holdings, cash, master_data)
    if 'error' not in result:
        user_store.update_portfolio(req.user_id, result['holdings'], result['cash'])

    return {'approved': True, 'portfolio': result}

@app.post('/api/recommendations/reject')
def reject_recommendation(req: ApprovalRequest):
    rec = user_store.reject_recommendation(req.user_id, req.rec_id)
    if not rec:
        raise HTTPException(404, 'Recommendation not found')
    return {'rejected': True}


# ── Full pipeline ─────────────────────────────────────────────────────────────

@app.post('/api/run')
def run_pipeline(req: RunRequest, bg: BackgroundTasks):
    bg.add_task(_pipeline_task, req)
    return {'message': 'Pipeline started', 'poll': '/api/pipeline-status'}


def _pipeline_task(req: RunRequest):
    try:
        _set('fetching', f'Fetching {req.index} tickers…', 3)
        # only US markets are supported
        market = 'United States'
        tickers = universe_builder.get_universe(market, req.index)
        target  = tickers if req.scope == 'all' else tickers[:10]

        _set('downloading', f'Downloading {len(target)} stocks…', 10)
        dl_r = downloader.download_universe(target)
        downloaded = dl_r.get('success', []) + dl_r.get('skipped', [])

        _set('cleaning',  'Cleaning OHLCV data…',              22)
        cl_r    = cleaner.clean_universe(downloaded)
        cleaned = cl_r.get('success', downloaded)

        _set('features',  'Building 38 technical indicators…', 34)
        fe_r     = engineer.generate_universe(cleaned)
        featured = fe_r.get('success', cleaned)

        _set('news',      'Collecting news headlines…',         48)
        news_collector.get_universe_news(featured)

        _set('sentiment', 'Running FinBERT sentiment…',         60)
        sentiment.analyze_universe(featured)

        _set('fusion',    'Merging features + sentiment…',      72)
        fused_r = fusion.fuse_universe(featured)
        fused   = fused_r.get('success', featured)

        combined = fusion.load_all(fused)
        if combined is None or (hasattr(combined, 'empty') and combined.empty):
            _set('error', 'No fused data — check logs', error='no_data')
            return

        _set('xgboost',  'Training XGBoost (5-fold CV)…',      80)
        xgb_model.train(combined)

        _set('lstm',     'Training LSTM sequences…',            90)
        lstm_model.train(combined, epochs=50)

        _set('done', 'Pipeline complete ✓', 100, tickers=fused)

        # if a user_id was provided, generate and queue recommendations
        if req.user_id and user_store.get(req.user_id):
            _generate_user_recommendations(req.user_id, fused, req.risk_profile, req.budget)

    except Exception as e:
        logger.error(f'Pipeline error: {e}', exc_info=True)
        _set('error', str(e), error=str(e))


def _generate_user_recommendations(user_id, tickers, risk_profile, budget):
    """After training, generate recommendations for a specific user."""
    predictions, master_data = _run_inference(tickers)
    if not predictions:
        return
    ranked = ranker.rank(predictions, master_data, risk_profile)
    result = recommender.recommend(ranked, risk_profile=risk_profile, top_n=10)

    # build recommendations with confidence scores
    recs = []
    for signal in result.get('all_signals', []):
        if signal['signal'] in ('BUY', 'SELL'):
            df         = master_data.get(signal['ticker'])
            confidence = portfolio_manager.confidence_score(
                signal['ticker'], df,
                signal.get('predicted_return', 0) / 100,
            )
            recs.append({
                'action':           signal['signal'],
                'ticker':           signal['ticker'],
                'signal':           signal['signal'],
                'score':            signal.get('composite_score', 50),
                'confidence':       confidence['overall'],
                'factors':          confidence['factors'],
                'reason':           portfolio_manager._buy_reason(signal, confidence) if signal['signal'] == 'BUY' else f'{signal["ticker"]} rated SELL — score {signal.get("composite_score",0):.0f}/100.',
                'predicted_return': signal.get('predicted_return', 0),
            })
            if len(recs) >= 5:
                break   # queue max 5 recommendations at once

    user_store.add_recommendations(user_id, recs)
    logger.info(f'Queued {len(recs)} recommendations for {user_id}')


@app.get('/api/pipeline-status')
def pipeline_status():
    with _lock:
        return dict(_status)


# ── Regime ───────────────────────────────────────────────────────────────────

@app.get('/api/regime')
def get_regime():
    with _lock:
        tickers = list(_status.get('processed_tickers', []))
    if not tickers:
        return {'regime': 'unknown', 'confidence': 0, 'metrics': {}}
    master_data = {t: df for t in tickers[:20]   # sample 20 for speed
                   if (df := fusion.load_master(t)) is not None and not df.empty}
    return regime_detector.detect(master_data)


# ── Advisor run (using stored user profile) ───────────────────────────────────

@app.post('/api/advisor/run/{user_id}')
def advisor_run(user_id: str, bg: BackgroundTasks):
    """
    Run the full analysis using the user's stored risk profile and budget.
    No need to pass risk profile — it's read from the stored profile.
    """
    profile = user_store.get(user_id)
    if not profile:
        raise HTTPException(404, 'User profile not found — complete onboarding first')

    req = RunRequest(
        market       = 'United States',
        index        = 'NASDAQ 100',
        risk_profile = profile.get('risk_profile', 'moderate'),
        budget       = profile.get('budget', 10000),
        scope        = 'sample',
        user_id      = user_id,
    )
    bg.add_task(_pipeline_task, req)
    return {'message': 'Analysis started using your stored profile', 'risk_profile': req.risk_profile}


# ── Recommendations + portfolio (stateless, for dashboard charts) ─────────────

@app.post('/api/recommend')
def get_recommendations_api(req: dict):
    risk_profile = req.get('risk_profile', 'moderate')
    top_n        = int(req.get('top_n', 10))

    if not xgb_model.is_trained() and not lstm_model.is_trained():
        raise HTTPException(400, 'Models not trained — run the pipeline first')

    with _lock:
        processed = list(_status.get('processed_tickers', []))
    if not processed:
        raise HTTPException(422, 'No processed tickers — run the pipeline first')

    predictions, master_data = _run_inference(processed)
    if not predictions:
        raise HTTPException(422, 'No predictions generated')

    ranked = ranker.rank(predictions, master_data, risk_profile)
    result = recommender.recommend(ranked, risk_profile=risk_profile, top_n=top_n)

    # add confidence scores to all signals
    for signal in result.get('all_signals', []):
        df         = master_data.get(signal['ticker'])
        confidence = portfolio_manager.confidence_score(
            signal['ticker'], df, signal.get('predicted_return', 0) / 100)
        signal['confidence']       = confidence['overall']
        signal['confidence_factors'] = confidence['factors']

    explainer.explain_batch(result.get('all_signals', []), master_data)
    return {'market': req.get('market', 'United States'), 'index': req.get('index', 'NASDAQ 100'), **result}


@app.post('/api/portfolio')
def get_portfolio(req: dict):
    risk_profile = req.get('risk_profile', 'moderate')
    budget       = float(req.get('budget', 10000))
    top_n        = int(req.get('top_n', 10))

    if not xgb_model.is_trained() and not lstm_model.is_trained():
        raise HTTPException(400, 'Models not trained')

    with _lock:
        processed = list(_status.get('processed_tickers', []))
    if not processed:
        raise HTTPException(422, 'No processed tickers')

    predictions, master_data = _run_inference(processed)
    if not predictions:
        raise HTTPException(422, 'No predictions')

    ranked    = ranker.rank(predictions, master_data, risk_profile)
    result    = recommender.recommend(ranked, risk_profile=risk_profile, top_n=top_n)
    portfolio = constructor.construct(ranked, master_data, budget, risk_profile, top_n)

    # enrich holdings with confidence and drift
    holdings = portfolio.get('portfolio', {}).get('holdings', [])
    for h in holdings:
        df = master_data.get(h['ticker'])
        c  = portfolio_manager.confidence_score(h['ticker'], df, h.get('predicted_return', 0) / 100)
        h['confidence'] = c['overall']
        h['confidence_factors'] = c['factors']

    # check for drift (only if user has an existing portfolio)
    drift_suggestions = []
    if holdings:
        drift_suggestions = portfolio_manager.suggest_rebalance(holdings, master_data)

    explainer.explain_batch(result.get('all_signals', []), master_data)

    # add market regime
    regime = regime_detector.detect(master_data)

    return {
        'market': req.get('market', 'United States'),
        'index':  req.get('index', 'NASDAQ 100'),
        'risk_profile': risk_profile, 'budget': budget,
        **portfolio,
        'signals':           result.get('summary', {}),
        'all_signals':       result.get('all_signals', []),
        'top_picks':         result.get('top_picks', []),
        'drift_suggestions': drift_suggestions,
        'market_regime':     regime,
    }


# ── Backtest + analysis ────────────────────────────────────────────────────

@app.post('/api/backtest')
def run_backtest(req: dict):
    tickers = req.get('tickers', [])
    capital = float(req.get('capital', 10000))
    mode    = req.get('portfolio_mode', False)

    if not xgb_model.is_trained():
        raise HTTPException(400, 'XGBoost not trained')
    master_data = {t: df for t in tickers
                   if (df := fusion.load_master(t)) is not None and not df.empty}
    if not master_data:
        raise HTTPException(404, 'No data for these tickers')
    if mode or len(tickers) > 1:
        return backtester.run_portfolio(tickers, master_data, xgb_model, capital)
    return backtester.run(tickers[0], master_data[tickers[0]], xgb_model, capital)


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
            'date': str(date.date()) if hasattr(date, 'date') else str(date),
            'open': safe(row.get('open')),   'high': safe(row.get('high')),
            'low':  safe(row.get('low')),    'close': safe(row.get('close')),
            'volume': int(row.get('volume', 0)) if pd.notna(row.get('volume')) else 0,
            'sma20': safe(row.get('sma20')), 'sma50': safe(row.get('sma50')),
            'bb_upper': safe(row.get('bb_upper')), 'bb_lower': safe(row.get('bb_lower')),
            'rsi': safe(row.get('rsi')),     'macd': safe(row.get('macd')),
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
    regime    = 'trending_up' if adx>25 and mom>0 else ('trending_down' if adx>25 and mom<0 else 'ranging')
    confidence = portfolio_manager.confidence_score(ticker, df, avg_pred)

    return {
        'ticker': ticker,
        'signal': 'BUY' if score>=65 else ('SELL' if score<=35 else 'HOLD'),
        'score':   round(score, 1), 'regime': regime,
        'close':   round(float(latest.get('close', 0) or 0), 2),
        'confidence': confidence,
        'prediction': {
            'xgboost':  round(pred_xgb*100,  3) if pred_xgb  is not None else None,
            'lstm':     round(pred_lstm*100, 3) if pred_lstm is not None else None,
            'ensemble': round(avg_pred*100,  3),
        },
        'indicators': {
            'rsi':          round(float(latest.get('rsi', 50)           or 50),   2),
            'adx':          round(adx, 2),
            'macd_hist':    round(float(latest.get('macd_hist', 0)      or 0),    4),
            'bb_pct':       round(float(latest.get('bb_pct', 0.5)       or 0.5),  4),
            'atr_pct':      round(float(latest.get('atr_pct', 0)        or 0),    4),
            'stoch_k':      round(float(latest.get('stoch_k', 50)       or 50),   2),
            'week52_pos':   round(float(latest.get('week52_position', 0.5) or 0.5), 4),
            'volume_ratio': round(float(latest.get('volume_ratio', 1)   or 1),    3),
            'momentum_10d': round(mom * 100, 2),
            'volatility':   round(vol, 4),
        },
        'sentiment': {
            'score': round(float(latest.get('sent_score', 0)  or 0), 4),
            'label': str(latest.get('sent_label', 'neutral')  or 'neutral'),
            'count': int(latest.get('sent_news_count', 0)     or 0),
        },
        'top_features': dict(list(xgb_model.feature_importance().items())[:8]) if xgb_model.is_trained() else {},
    }


@app.get('/api/model-status')
def model_status():
    return {'xgboost_trained': xgb_model.is_trained(), 'lstm_trained': lstm_model.is_trained()}


# ── Ollama + chat ─────────────────────────────────────────────────────────────

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
        f'VaR {rm.get("var_95_1day",0)*100:.2f}%'
        + ('\nHoldings: ' + ', '.join(f'{h["ticker"]} ({h["signal"]})' for h in holdings[:5]) if holdings else '')
    )
    history_block = f'\n\nConversation today:\n{req.context}' if req.context else ''
    prompt = (
        'You are a professional AI financial advisor. '
        'Answer in 3-4 concise sentences using the data and conversation history. '
        'Reference specific numbers. Do not start with "I".\n\n'
        f'{ctx}{history_block}\n\nQuestion: {req.message}\nAnswer:'
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

    msg = req.message.lower()
    if not p:
        return {'reply': 'Complete onboarding and run an analysis first.', 'source': 'template'}
    if any(w in msg for w in ['sharpe','var','risk']):
        return {'reply': f'Sharpe: {rm.get("annualized_sharpe",0):.2f}. VaR: {rm.get("var_95_1day",0)*100:.2f}%.', 'source':'template'}
    if any(w in msg for w in ['buy','best','top']):
        buys = [h for h in holdings if h.get('signal')=='BUY'][:3]
        if not buys: return {'reply': 'No BUY signals currently.', 'source':'template'}
        return {'reply': 'Top BUY signals: ' + ', '.join(f'{h["ticker"]} (conf. {h.get("confidence",50)}%)' for h in buys) + '.', 'source':'template'}
    return {'reply': f'{p.get("n_positions",0)} positions, ${p.get("total_invested",0):,.0f} invested, expected return {p.get("expected_portfolio_return",0):+.2f}%.', 'source':'template'}


# ── Validation ────────────────────────────────────────────────────────────────

@app.get('/api/validation-results')
def validation_results():
    path = Path('data/validation_results.json')
    if not path.exists():
        return {'error': 'No results — run scripts/evaluate_models.py'}
    return json.loads(path.read_text())

@app.post('/api/evaluate-all')
def run_evaluation(bg: BackgroundTasks):
    import subprocess
    bg.add_task(lambda: subprocess.run([sys.executable, 'scripts/evaluate_models.py'], check=False))
    return {'status': 'running'}


# ── Helper ────────────────────────────────────────────────────────────────────

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
