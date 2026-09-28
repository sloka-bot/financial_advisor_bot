"""Train and calibrate a direction classifier with chronological validation and versioning."""
# Author: Sloka Mudunuru

import json
import logging
import pickle
import shutil
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler

from backend.config.settings import PREDICTION_HORIZON, embargo_for
from backend.prediction.temporal_split import date_split_three

logger = logging.getLogger(__name__)
MODELS_DIR = Path("models/xgboost")


EXCLUDED = {
    "trade_return",
    "execution_date",
    "exit_date",
    "label_end",
    "fwd_ret",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "daily_return",
    "target_return",
    "target_direction",
    "ticker",  # label and metadata columns excluded from features
    "extreme_move_flag",
    "potential_split_flag",
    "penny_stock_flag",
    "corp_action_flag",
    "ohlc_breach_flag",
    "sent_label",
    "bb_upper",
    "bb_lower",
    "close_unadj",
    "exec_close",
    "adj_close",
}


def _make_labels(df, horizon=PREDICTION_HORIZON):
    """Use existing direction labels or compute them from horizon returns."""
    if "target_direction" in df.columns:
        return df["target_direction"].astype(float)
    fwd = df["close"].shift(-horizon) / df["close"] - 1.0
    return (fwd > 0).astype(float).where(fwd.notna())


# Use recent calendar years as expanding-window validation folds.
WALK_FORWARD_FOLDS = 4

# Require sufficient labelled training rows before deployment.
MIN_DEPLOY_SAMPLES = 500


class XGBoostForecaster:
    """Fit and calibrate a classifier for the forward price-direction target."""

    def __init__(self, models_dir=None, horizon=None):
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        self.horizon = int(horizon) if horizon is not None else PREDICTION_HORIZON
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.calibrated_model = None
        self.scaler = RobustScaler()
        self._feat_cols = []

    def train(
        self, df: pd.DataFrame, *, persist: bool = True, force_deploy: bool = False, register: bool = True
    ) -> dict:
        """Validate across time, fit the final classifier and record its deployment decision."""
        import xgboost as xgb

        from backend.infra.model_registry import ModelRegistry

        feat_cols = self._select_features(df)
        if not feat_cols:
            return {"error": "No usable features"}
        self._feat_cols = feat_cols

        labels = _make_labels(df, self.horizon)
        finite_mask = df[feat_cols].apply(lambda col: np.isfinite(col)).all(axis=1)
        mask = finite_mask & labels.notna()
        X = df.loc[mask, feat_cols].values
        y = labels[mask].values

        if len(X) < 50:
            # Keep deployed models when data is insufficient; bootstrap only on a cold start.
            if (self.models_dir / "model.pkl").exists():
                logger.warning(f"XGBoost: only {len(X)} rows - keeping the existing model, not retraining")
                return {
                    "error": "too_few_rows",
                    "n_samples": int(len(y)),
                    "note": "kept existing model (insufficient data to retrain)",
                    "deployed": False,
                }
            logger.warning(f"XGBoost: only {len(X)} rows - cold-start bootstrap fit (NOT walk-forward validated)")
            Xs = self.scaler.fit_transform(X)  # fit and use the scaled matrix
            self.model = self._make_classifier()
            self.model.fit(Xs, y, verbose=False)  # predict_proba_up scales inputs
            self._save()
            return {
                "cv_auc_mean": None,
                "cv_ran": False,
                "n_samples": int(len(y)),
                "note": "cold-start bootstrap - not walk-forward validated",
                "deployed": True,
            }

        # Walk-forward validation.
        fold_aucs, fold_dirs = self._walk_forward_cv(df, feat_cols)

        # Separate training, early stopping and calibration by date with label embargoes.
        sub = df.loc[mask]
        y_ser = pd.Series(y, index=sub.index)
        udates = np.array(sorted(sub.index.unique()))
        emb = embargo_for(self.horizon)

        # Build the three chronological blocks through the shared split utility.
        tr_dates, es_dates, cal_dates = date_split_three(udates, embargo=emb)

        def _rows(dts):
            m = sub.index.isin(dts)
            return sub.loc[m, feat_cols].values, y_ser.values[m]

        X_tr, y_tr = _rows(tr_dates)
        X_es, y_es = _rows(es_dates)
        X_cal, y_cal = _rows(cal_dates)

        self.scaler = RobustScaler()
        if len(X_tr) >= 20 and len(X_es) >= 10:
            X_tr_sc = self.scaler.fit_transform(X_tr)  # scaler fitted on training rows only
            X_es_sc = self.scaler.transform(X_es)
            # Keep regularised hyperparameters fixed across folds; validation selects tree count.
            self.model = xgb.XGBClassifier(
                n_estimators=600,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                eval_metric="auc",
                early_stopping_rounds=30,
                random_state=42,
                n_jobs=1,
                verbosity=0,
            )
            self.model.fit(X_tr_sc, y_tr, eval_set=[(X_es_sc, y_es)], verbose=False)
            best_iteration = getattr(self.model, "best_iteration", self.model.n_estimators)
            early_stopping = True
            logger.info(
                f"XGBoost: best iteration = {best_iteration} "
                f"(train={len(X_tr)}, es_val={len(X_es)}, calib={len(X_cal)})"
            )
        else:
            logger.warning(
                "XGBoost: too few dated rows for a date split "
                f"(train={len(X_tr)}, es={len(X_es)}) - training on all rows without early stopping"
            )
            self.model = self._make_classifier()
            self.model.fit(self.scaler.fit_transform(sub[feat_cols].values), y_ser.values, verbose=False)
            best_iteration = self.model.n_estimators
            early_stopping = False

        # Fit probability calibration on a block separate from early stopping.
        self.calibrated_model = None
        if len(X_cal) >= 10:
            try:
                from sklearn.isotonic import IsotonicRegression

                raw_probs = self.model.predict_proba(self.scaler.transform(X_cal))[:, 1]
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(raw_probs, y_cal)
                self.calibrated_model = iso
                logger.info(f"XGBoost: isotonic calibration fitted on a separate {len(X_cal)}-row block")
            except Exception as e:
                logger.warning(f"Calibration skipped ({e})")

        # Report unavailable AUC when no validation folds were scored.
        cv_ran = bool(fold_aucs)
        auc_mean = round(float(np.mean(fold_aucs)), 4) if cv_ran else None
        auc_std = round(float(np.std(fold_aucs)), 4) if cv_ran else None
        dir_mean = round(float(np.mean(fold_dirs)), 4) if fold_dirs else None

        results = {
            "auc": auc_mean,
            "cv_auc_mean": auc_mean,
            "cv_auc_std": auc_std,
            "cv_fold_aucs": fold_aucs,
            "cv_ran": cv_ran,
            "cv_direction_acc": dir_mean,
            "best_iteration": best_iteration,
            "early_stopping": early_stopping,  # False when the date split is too small
            "n_features": len(feat_cols),
            "n_samples": int(len(y)),
            "trained_at": datetime.now().isoformat(),
            "horizon": self.horizon,
            "date_range": [str(df.index.min())[:10], str(df.index.max())[:10]]
            if isinstance(df.index, pd.DatetimeIndex)
            else None,
            "hyperparameters": {
                "n_estimators": 600,
                "max_depth": 6,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_weight": 3,
                "gamma": 0.1,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "early_stopping_rounds": 30,
            },
        }

        # Versioning and deployment gate.
        if persist and not register:
            # Save alternate-horizon artifacts without updating the primary model registry.
            self._save()
            results["deployed"] = True
            results["registered"] = False
            with open(self.models_dir / "training_results.json", "w") as f:
                json.dump(results, f, indent=2)
            return results
        if not persist:
            # Return dry-run metrics without changing deployed artifacts or registry entries.
            logger.info("XGBoost: dry run (persist=False) - metrics only; registry and artifacts left untouched")
            results["deployed"] = False
            results["persisted"] = False
            return results
        registry = ModelRegistry()
        existing = registry.get_best("xgboost")
        enough_data = len(y) >= MIN_DEPLOY_SAMPLES
        if (not cv_ran or not enough_data) and existing is not None:
            # Retain the deployed model when the candidate lacks sufficient validation evidence.
            logger.warning(
                f"XGBoost: candidate not deployable (cv_ran={cv_ran}, "
                f"n={len(y)} < {MIN_DEPLOY_SAMPLES}={not enough_data}) - "
                "keeping the existing deployed model"
            )
            self.model = None
            self.calibrated_model = None
            self._loaded()
            results["deployed"] = False
        elif force_deploy or registry.should_deploy("xgboost", results):
            self._save()
            version = registry.register(
                "xgboost",
                results,
                feat_cols,
                train_end_year=int(df.index.year.max()) if hasattr(df.index, "year") else None,
                force_best=force_deploy,
            )
            results["version"] = version
            results["deployed"] = True
            logger.info(f"XGBoost v{version} deployed (AUC={auc_mean}{' [forced]' if force_deploy else ''})")
        else:
            # Clear the rejected in-memory candidate before reloading deployed artifacts.
            logger.info("XGBoost: new model did not beat current best - reverting to deployed model")
            self.model = None
            self.calibrated_model = None
            self._loaded()  # reload the saved best artifacts
            results["deployed"] = False

        with open(self.models_dir / "training_results.json", "w") as f:
            json.dump(results, f, indent=2)
        return results

    def _walk_forward_cv(self, df: pd.DataFrame, feat_cols: list):
        """Evaluate annual test folds using earlier training data and label embargoes."""
        import xgboost as xgb

        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                return [], []

        labels = _make_labels(df, self.horizon)
        finite = df[feat_cols].apply(lambda c: np.isfinite(c)).all(axis=1)
        valid = finite & labels.notna()
        emb = embargo_for(self.horizon)  # sessions purged at each year boundary

        years = sorted(df.index.year.unique())
        # Use the last N years as test folds, each with at least a year of training data.
        test_years = years[max(1, len(years) - WALK_FORWARD_FOLDS) :]

        fold_aucs = []
        fold_dirs = []

        for test_year in test_years:
            train_mask = (df.index.year < test_year) & valid
            test_mask = (df.index.year == test_year) & valid

            # Purge training labels that would extend into the test year.
            if emb > 0:
                tr_dates = np.array(sorted(df.index[train_mask].unique()))
                if len(tr_dates) > emb:
                    keep = set(tr_dates[:-emb])
                    train_mask = train_mask & df.index.isin(keep)

            if train_mask.sum() < 200 or test_mask.sum() < 30:
                logger.debug(f"Walk-forward {test_year}: skipped (train={train_mask.sum()}, test={test_mask.sum()})")
                continue

            tr_row_dates = pd.DatetimeIndex(df.index[train_mask])
            X_tr = df.loc[train_mask, feat_cols].values
            y_tr = labels[train_mask].values
            X_te = df.loc[test_mask, feat_cols].values
            y_te = labels[test_mask].values

            # Recheck sizes after masking.
            if len(X_tr) < 50 or len(X_te) < 5:
                continue

            # Scaler fitted on training rows only.
            fold_scaler = RobustScaler()
            X_tr_sc = fold_scaler.fit_transform(X_tr)
            X_te_sc = fold_scaler.transform(X_te)

            m = xgb.XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                eval_metric="auc",
                early_stopping_rounds=20,
                random_state=42,
                n_jobs=1,
                verbosity=0,
            )
            # Reserve later training dates for early stopping and embargo the boundary.
            uniq = np.array(sorted(tr_row_dates.unique()))
            n_v = max(1, int(len(uniq) * 0.15))
            es_set = set(uniq[-n_v:])
            core_d = uniq[:-n_v]
            if emb > 0 and len(core_d) > emb:
                core_d = core_d[:-emb]
            core_set = set(core_d)
            is_es = np.asarray(tr_row_dates.isin(es_set))
            is_core = np.asarray(tr_row_dates.isin(core_set))
            if is_es.sum() >= 10 and is_core.sum() >= 40:
                m.fit(X_tr_sc[is_core], y_tr[is_core], eval_set=[(X_tr_sc[is_es], y_tr[is_es])], verbose=False)
            else:
                m.set_params(early_stopping_rounds=None)
                m.fit(X_tr_sc, y_tr, verbose=False)

            proba = m.predict_proba(X_te_sc)[:, 1]
            try:
                auc = float(roc_auc_score(y_te, proba))
            except ValueError:
                auc = None  # single-class folds are excluded from the mean
            dir_acc = float(np.mean((proba > 0.5) == y_te))

            if auc is not None:
                fold_aucs.append(round(auc, 4))
            fold_dirs.append(round(dir_acc, 4))
            auc_str = f"{auc:.4f}" if auc is not None else "n/a (single-class fold)"
            logger.info(
                f"  Walk-forward {test_year}: AUC={auc_str}  DirAcc={dir_acc:.3f}"
                f"  (train={train_mask.sum()} rows, test={test_mask.sum()} rows)"
            )

        return fold_aucs, fold_dirs

    def _make_classifier(self):
        import xgboost as xgb

        return xgb.XGBClassifier(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="auc",
            random_state=42,
            n_jobs=1,
            verbosity=0,
        )

    def update(self, df: pd.DataFrame) -> dict:
        """Retrain through the full chronological validation and deployment workflow."""
        return self.train(df)

    def predict_proba_up(self, df: pd.DataFrame) -> float | None:
        """Return the calibrated horizon up-probability, or None when unavailable."""
        warnings.filterwarnings("ignore", message="X has feature names")
        if not self._loaded():
            return None
        cols = self._feat_cols if all(c in df.columns for c in self._feat_cols) else []
        if not cols:
            return None
        latest = df[cols].iloc[[-1]].copy().ffill().bfill()
        if latest.isnull().any(axis=1).values[0]:
            return None
        try:
            X = self.scaler.transform(latest.values)
            raw = float(self.model.predict_proba(X)[0, 1])
            if self.calibrated_model is not None:
                return float(self.calibrated_model.predict([raw])[0])
            return raw
        except Exception as e:
            logger.warning(f"XGBoost predict_proba_up error: {e}")
            return None

    def predict_with_uncertainty(self, df: pd.DataFrame) -> dict:
        """Return up-probability and its distance from 0.5; this is not a return interval."""
        empty = {"prob_up": None, "confidence": 0.0, "calibrated": False, "interpretation": None}
        if not self._loaded():
            return dict(empty)
        cols = self._feat_cols if all(c in df.columns for c in self._feat_cols) else []
        if not cols:
            return dict(empty)
        try:
            latest = df[cols].iloc[[-1]].copy().ffill().bfill()
            X = self.scaler.transform(latest.values)
            raw = float(self.model.predict_proba(X)[0, 1])
            up_prob = float(self.calibrated_model.predict([raw])[0]) if self.calibrated_model is not None else raw
            return {
                "prob_up": round(up_prob, 4),
                "confidence": round(abs(up_prob - 0.5) * 2, 4),  # 0 is a coin flip, 1 is certain
                "calibrated": self.calibrated_model is not None,
                "interpretation": "UP" if up_prob > 0.5 else "DOWN",
            }
        except Exception as e:
            logger.warning(f"XGBoost uncertainty predict error: {e}")
            return dict(empty)

    def feature_importance(self) -> dict:
        """Return fitted feature importances, or an empty mapping without a model."""
        if not self._loaded():
            return {}
        scores = self.model.feature_importances_
        pairs = sorted(zip(self._feat_cols, scores), key=lambda x: x[1], reverse=True)
        return {n: round(float(s), 4) for n, s in pairs[:20]}

    def is_trained(self) -> bool:
        """Check whether the saved classifier artifact exists."""
        return (self.models_dir / "model.pkl").exists()

    def _select_features(self, df):
        from backend.evaluation.experiments import all_features

        return all_features(df.columns)

    def _save(self):
        # Save the current model.
        with open(self.models_dir / "model.pkl", "wb") as f:
            pickle.dump(self.model, f)
        with open(self.models_dir / "scaler.pkl", "wb") as f:
            pickle.dump(self.scaler, f)
        with open(self.models_dir / "feat_cols.pkl", "wb") as f:
            pickle.dump(self._feat_cols, f)
        # Save library and feature-schema compatibility metadata with the model.
        from backend.infra.model_registry import artifact_fingerprint

        with open(self.models_dir / "artifact_manifest.json", "w") as f:
            json.dump(artifact_fingerprint(self._feat_cols), f, indent=2)
        # Remove stale calibration artifacts when the saved model has no calibrator.
        cp = self.models_dir / "calibrated.pkl"
        if self.calibrated_model:
            with open(cp, "wb") as f:
                pickle.dump(self.calibrated_model, f)
        elif cp.exists():
            cp.unlink()

        # Store timestamped backups separately from active model artifacts.
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            archive = self.models_dir / "archive"
            archive.mkdir(parents=True, exist_ok=True)
            shutil.copy(self.models_dir / "model.pkl", archive / f"model_{ts}.pkl")
        except Exception:
            pass  # backup failure is not fatal

    def _loaded(self) -> bool:
        if self.model:
            return True
        mp = self.models_dir / "model.pkl"
        sp = self.models_dir / "scaler.pkl"
        fp = self.models_dir / "feat_cols.pkl"
        if not (mp.exists() and sp.exists()):
            return False
        try:
            with open(mp, "rb") as f:
                self.model = pickle.load(f)
            with open(sp, "rb") as f:
                self.scaler = pickle.load(f)
            if fp.exists():
                with open(fp, "rb") as f:
                    self._feat_cols = pickle.load(f)
            mfp = self.models_dir / "artifact_manifest.json"
            if mfp.exists() and self._feat_cols:
                from backend.infra.model_registry import check_artifact_compatibility

                with open(mfp) as f:
                    manifest = json.load(f)
                ok, issues = check_artifact_compatibility(manifest, self._feat_cols)
                if not ok:
                    logger.error("XGBoost artifact rejected as incompatible: %s", "; ".join(issues))
                    self.model = None
                    self.scaler = None
                    self.calibrated_model = None
                    return False
                if issues:
                    logger.warning("XGBoost artifact compatibility warnings: %s", "; ".join(issues))
            cp = self.models_dir / "calibrated.pkl"
            if cp.exists():
                try:
                    with open(cp, "rb") as f:
                        self.calibrated_model = pickle.load(f)
                except Exception:
                    self.calibrated_model = None
            return True
        except Exception as e:
            logger.warning(f"XGBoost load error: {e}")
            return False
