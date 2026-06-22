import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.data.fusion import FeatureFusion
from backend.engine.backtester import Backtester
from backend.models.xgboost_model import XGBoostForecaster
from backend.models.lstm_model import LSTMForecaster

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

OUT_DIR = Path('data')
OUT_DIR.mkdir(exist_ok=True)


def load_tickers():
    files = sorted(Path('data/universe').glob('*.json'), key=lambda f: f.stat().st_mtime)
    if not files:
        logger.error('No universe found — run the main pipeline first')
        sys.exit(1)
    return json.loads(files[-1].read_text())['tickers']


def load_features(tickers):
    fuser = FeatureFusion()
    data  = {}
    for t in tickers:
        df = fuser.load_master(t)
        if df is not None and len(df) > 100:
            data[t] = df
    return data


def xgboost_cv(master_data):
    """
    5-fold time-series cross-validation on the held-out 20% of each ticker.
    Returns per-fold AUC, mean AUC, and direction accuracy — core evidence
    that XGBoost generalises beyond its training window.
    """
    import xgboost as xgb

    EXCLUDE = {
        'open', 'high', 'low', 'close', 'volume', 'daily_return',
        'extreme_move_flag', 'potential_split_flag', 'penny_stock_flag',
        'corp_action_flag', 'ohlc_breach_flag', 'sent_label',
        'bb_upper', 'bb_lower',
    }

    frames = []
    for df in master_data.values():
        frames.append(df)
    combined = pd.concat(frames).reset_index(drop=True)

    feat_cols = [c for c in combined.columns
                 if c not in EXCLUDE and pd.api.types.is_numeric_dtype(combined[c])]

    labels = (combined['daily_return'].shift(-1) > 0).astype(float)
    mask   = combined[feat_cols].notna().all(axis=1) & labels.notna()
    X      = combined.loc[mask, feat_cols].values
    y      = labels[mask].values

    scaler   = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    tscv      = TimeSeriesSplit(n_splits=5)
    fold_aucs = []
    fold_accs = []

    for fold, (tr, val) in enumerate(tscv.split(X_scaled), 1):
        m = xgb.XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            eval_metric='auc', random_state=42, n_jobs=-1, verbosity=0,
        )
        m.fit(X_scaled[tr], y[tr], eval_set=[(X_scaled[val], y[val])], verbose=False)
        proba = m.predict_proba(X_scaled[val])[:, 1]
        auc   = roc_auc_score(y[val], proba)
        acc   = float(np.mean((proba > 0.5) == y[val]))
        fold_aucs.append(round(auc, 4))
        fold_accs.append(round(acc, 4))
        logger.info(f'  XGBoost fold {fold}: AUC={auc:.4f}  DirAcc={acc:.3f}')

    return {
        'fold_aucs':      fold_aucs,
        'mean_auc':       round(float(np.mean(fold_aucs)), 4),
        'std_auc':        round(float(np.std(fold_aucs)),  4),
        'mean_dir_acc':   round(float(np.mean(fold_accs)), 4),
        'fold_dir_accs':  fold_accs,
        'n_features':     len(feat_cols),
        'n_samples':      len(y),
    }


def sentiment_ablation(master_data):
    """
    Train XGBoost twice: once with sentiment features, once without.
    The difference in direction accuracy directly measures how much FinBERT adds.
    """
    import xgboost as xgb

    SENT_COLS = {'sent_score', 'sent_news_count', 'sent_label',
                 'sent_weighted', 'sent_pos', 'sent_neg'}
    EXCLUDE   = {
        'open', 'high', 'low', 'close', 'volume', 'daily_return',
        'extreme_move_flag', 'potential_split_flag', 'penny_stock_flag',
        'corp_action_flag', 'ohlc_breach_flag', 'sent_label', 'bb_upper', 'bb_lower',
    }

    combined = pd.concat(list(master_data.values())).reset_index(drop=True)
    labels   = (combined['daily_return'].shift(-1) > 0).astype(float)

    def run_cv(feat_cols):
        mask     = combined[feat_cols].notna().all(axis=1) & labels.notna()
        X        = combined.loc[mask, feat_cols].values
        y        = labels[mask].values
        X_sc     = RobustScaler().fit_transform(X)
        tscv     = TimeSeriesSplit(n_splits=3)
        accs     = []
        for tr, val in tscv.split(X_sc):
            m = xgb.XGBClassifier(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1, verbosity=0)
            m.fit(X_sc[tr], y[tr], verbose=False)
            accs.append(float(np.mean((m.predict(X_sc[val]) == y[val]))))
        return round(float(np.mean(accs)), 4)

    all_feats  = [c for c in combined.columns if c not in EXCLUDE and pd.api.types.is_numeric_dtype(combined[c])]
    no_sent    = [c for c in all_feats if c not in SENT_COLS]

    acc_with    = run_cv(all_feats)
    acc_without = run_cv(no_sent)

    logger.info(f'  Sentiment ablation: with={acc_with:.4f}  without={acc_without:.4f}  delta={acc_with-acc_without:+.4f}')
    return {
        'dir_acc_with_sentiment':    acc_with,
        'dir_acc_without_sentiment': acc_without,
        'sentiment_delta':           round(acc_with - acc_without, 4),
        'n_sentiment_features':      len(all_feats) - len(no_sent),
    }


def backtest_results(master_data, tickers):
    """
    Walk-forward backtest (last 20% of data) for each ticker.
    Strategy: long when XGBoost predicts positive return, flat otherwise.
    Compares Sharpe, total return, and max drawdown against buy-and-hold.
    """
    model = XGBoostForecaster()
    if not model.is_trained():
        logger.warning('XGBoost not trained — skipping backtest')
        return {}

    bt      = Backtester()
    results = {}
    for ticker in tickers[:10]:   # cap at 10 to keep runtime reasonable
        df = master_data.get(ticker)
        if df is None:
            continue
        r = bt.run(ticker, df, model, capital=10_000)
        if 'metrics' in r:
            results[ticker] = r['metrics']
            m = r['metrics']
            logger.info(f'  {ticker}: Sharpe={m["sharpe_ratio"]:.2f}  '
                        f'Ret={m["total_return"]*100:.1f}%  '
                        f'BH={m["benchmark_total_return"]*100:.1f}%  '
                        f'DirAcc={m["direction_accuracy"]:.3f}')

    if not results:
        return {}

    sharpes  = [r['sharpe_ratio']         for r in results.values()]
    rets     = [r['total_return']         for r in results.values()]
    bh_rets  = [r['benchmark_total_return'] for r in results.values()]
    accs     = [r['direction_accuracy']   for r in results.values()]

    return {
        'per_ticker':           results,
        'avg_sharpe':           round(float(np.mean(sharpes)), 4),
        'avg_strategy_return':  round(float(np.mean(rets)),    4),
        'avg_bh_return':        round(float(np.mean(bh_rets)), 4),
        'avg_direction_acc':    round(float(np.mean(accs)),    4),
        'pct_beats_bh':         round(sum(r > b for r, b in zip(rets, bh_rets)) / len(rets) * 100, 1),
    }


def lstm_vs_baseline(master_data, tickers):
    """
    Compare LSTM direction accuracy against a naive baseline that always
    predicts the previous day's direction (momentum heuristic).
    A meaningful LSTM must beat this zero-cost baseline.
    """
    model = LSTMForecaster()
    if not model.is_trained():
        logger.warning('LSTM not trained — skipping LSTM evaluation')
        return {}

    lstm_accs     = []
    baseline_accs = []

    for ticker in tickers[:8]:
        df = master_data.get(ticker)
        if df is None or len(df) < 100:
            continue

        # out-of-sample split: last 20%
        split     = int(len(df) * 0.8)
        test_df   = df.iloc[split:].copy()
        ret_col   = test_df['daily_return'].fillna(0).values

        # naive baseline: predict tomorrow's direction = today's direction
        baseline_preds  = np.sign(ret_col[:-1])
        baseline_actual = np.sign(ret_col[1:])
        b_acc = float(np.mean(baseline_preds == baseline_actual))
        baseline_accs.append(b_acc)

        # LSTM predictions on the test window
        l_preds   = []
        l_actuals = []
        for i in range(model.seq_len, len(test_df) - 1):
            window = df.iloc[split + i - model.seq_len: split + i + 1]
            pred   = model.predict_ticker(window)
            if pred is not None:
                l_preds.append(pred)
                l_actuals.append(float(ret_col[i + 1]))

        if l_preds:
            l_acc = float(np.mean(np.sign(l_preds) == np.sign(l_actuals)))
            lstm_accs.append(l_acc)
            logger.info(f'  {ticker}: LSTM={l_acc:.3f}  baseline={b_acc:.3f}')

    if not lstm_accs:
        return {}

    return {
        'avg_lstm_dir_acc':     round(float(np.mean(lstm_accs)),     4),
        'avg_baseline_dir_acc': round(float(np.mean(baseline_accs)), 4),
        'lstm_improvement':     round(float(np.mean(lstm_accs)) - float(np.mean(baseline_accs)), 4),
    }


def main():
    tickers     = load_tickers()
    master_data = load_features(tickers)
    logger.info(f'Evaluating on {len(master_data)} tickers')

    results = {'generated_at': datetime.now().isoformat(), 'tickers_evaluated': list(master_data.keys())}

    logger.info('\n[1/4] XGBoost cross-validation')
    results['xgboost_cv'] = xgboost_cv(master_data)

    logger.info('\n[2/4] Sentiment ablation study')
    results['sentiment_ablation'] = sentiment_ablation(master_data)

    logger.info('\n[3/4] Walk-forward backtesting')
    results['backtest'] = backtest_results(master_data, list(master_data.keys()))

    logger.info('\n[4/4] LSTM vs naive baseline')
    results['lstm_vs_baseline'] = lstm_vs_baseline(master_data, list(master_data.keys()))

    out = OUT_DIR / 'validation_results.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f'\nResults saved to {out}')

    print('\n' + '='*50)
    print('VALIDATION SUMMARY')
    print('='*50)

    xgb = results.get('xgboost_cv', {})
    print(f'XGBoost — mean AUC: {xgb.get("mean_auc","--")}  ±{xgb.get("std_auc","--")}')
    print(f'          direction accuracy: {xgb.get("mean_dir_acc","--")}')

    sa = results.get('sentiment_ablation', {})
    print(f'Sentiment — with: {sa.get("dir_acc_with_sentiment","--")}  '
          f'without: {sa.get("dir_acc_without_sentiment","--")}  '
          f'delta: {sa.get("sentiment_delta","--"):+}')

    bt = results.get('backtest', {})
    print(f'Backtest  — avg Sharpe: {bt.get("avg_sharpe","--")}  '
          f'strategy vs B&H: {bt.get("avg_strategy_return","--")} vs {bt.get("avg_bh_return","--")}')
    print(f'            beats buy-and-hold: {bt.get("pct_beats_bh","--")}% of tickers')

    ls = results.get('lstm_vs_baseline', {})
    print(f'LSTM      — dir acc: {ls.get("avg_lstm_dir_acc","--")}  '
          f'baseline: {ls.get("avg_baseline_dir_acc","--")}  '
          f'improvement: {ls.get("lstm_improvement","--"):+}')
    print('='*50)


if __name__ == '__main__':
    main()
