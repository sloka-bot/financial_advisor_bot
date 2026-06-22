import json
import logging
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler

logger     = logging.getLogger(__name__)
MODELS_DIR = Path('models/xgboost')

EXCLUDED = {
    'open', 'high', 'low', 'close', 'volume', 'daily_return', 'target_return',
    'extreme_move_flag', 'potential_split_flag', 'penny_stock_flag',
    'corp_action_flag', 'ohlc_breach_flag', 'sent_label',
    'bb_upper', 'bb_lower',
}


class XGBoostForecaster:

    def __init__(self):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.model      = None
        self.scaler     = RobustScaler()
        self._feat_cols = []

    def train(self, df: pd.DataFrame) -> dict:
        import xgboost as xgb

        feat_cols = self._select_features(df)
        if not feat_cols:
            return {'error': 'No usable features'}

        labels = (df['daily_return'].shift(-1) > 0).astype(float)
        mask   = df[feat_cols].notna().all(axis=1) & labels.notna()
        # fit scaler on numpy arrays — avoids feature-name warnings at predict time
        X      = df.loc[mask, feat_cols].values
        y      = labels[mask].values
        self._feat_cols = feat_cols

        X_sc = self.scaler.fit_transform(X)

        tscv      = TimeSeriesSplit(n_splits=5)
        fold_aucs = []
        fold_accs = []

        for fold, (tr, val) in enumerate(tscv.split(X_sc), 1):
            m = xgb.XGBClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
                gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
                eval_metric='auc', random_state=42, n_jobs=1, verbosity=0,
            )
            m.fit(X_sc[tr], y[tr], eval_set=[(X_sc[val], y[val])], verbose=False)
            proba = m.predict_proba(X_sc[val])[:, 1]
            fold_aucs.append(round(float(roc_auc_score(y[val], proba)), 4))
            fold_accs.append(round(float(np.mean((proba > 0.5) == y[val])), 4))
            logger.info(f'  Fold {fold}: AUC={fold_aucs[-1]:.4f}  DirAcc={fold_accs[-1]:.3f}')

        self.model = xgb.XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
            eval_metric='auc', random_state=42, n_jobs=1, verbosity=0,
        )
        self.model.fit(X_sc, y, verbose=False)
        self._save()

        results = {
            'cv_auc_mean':      round(float(np.mean(fold_aucs)), 4),
            'cv_auc_std':       round(float(np.std(fold_aucs)),  4),
            'cv_fold_aucs':     fold_aucs,
            'cv_direction_acc': round(float(np.mean(fold_accs)), 4),
            'n_features':       len(feat_cols),
            'n_samples':        int(len(y)),
            'trained_at':       datetime.now().isoformat(),
        }
        with open(MODELS_DIR / 'training_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        return results

    def predict_ticker(self, df: pd.DataFrame) -> float | None:
        if not self._loaded():
            return None
        cols   = [c for c in self._feat_cols if c in df.columns]
        if not cols:
            return None
        latest = df[cols].iloc[[-1]].copy().ffill().bfill()
        if latest.isnull().any(axis=1).values[0]:
            return None
        try:
            # always pass numpy array to scaler — avoids feature-name warning
            X       = self.scaler.transform(latest.values)
            up_prob = float(self.model.predict_proba(X)[0, 1])
            return (up_prob - 0.5) * 0.02
        except Exception as e:
            logger.warning(f'XGBoost predict error: {e}')
            return None

    def feature_importance(self) -> dict:
        if not self._loaded():
            return {}
        scores = self.model.feature_importances_
        pairs  = sorted(zip(self._feat_cols, scores), key=lambda x: x[1], reverse=True)
        return {n: round(float(s), 4) for n, s in pairs[:20]}

    def is_trained(self) -> bool:
        return (MODELS_DIR / 'model.pkl').exists()

    def _select_features(self, df):
        return [c for c in df.columns
                if c not in EXCLUDED and pd.api.types.is_numeric_dtype(df[c])]

    def _save(self):
        with open(MODELS_DIR / 'model.pkl',     'wb') as f: pickle.dump(self.model,      f)
        with open(MODELS_DIR / 'scaler.pkl',    'wb') as f: pickle.dump(self.scaler,     f)
        with open(MODELS_DIR / 'feat_cols.pkl', 'wb') as f: pickle.dump(self._feat_cols, f)

    def _loaded(self) -> bool:
        if self.model:
            return True
        mp = MODELS_DIR / 'model.pkl'
        sp = MODELS_DIR / 'scaler.pkl'
        fp = MODELS_DIR / 'feat_cols.pkl'
        if not (mp.exists() and sp.exists()):
            return False
        try:
            with open(mp, 'rb') as f: self.model       = pickle.load(f)
            with open(sp, 'rb') as f: self.scaler      = pickle.load(f)
            if fp.exists():
                with open(fp, 'rb') as f: self._feat_cols = pickle.load(f)
            return True
        except Exception as e:
            logger.warning(f'XGBoost load error: {e}')
            return False
