"""
evaluate_deployed.py

Evaluate the deployed model types against fair baselines on matched,
chronological, leakage-free test data, so the deployed model types are evaluated
out-of-sample rather than only the experimental variants.

Design (mirrors the deployed tasks, not the experiment regressor):
  * Split unique calendar dates once: first (1-test_frac) = development, last
    test_frac = untouched final test. An H-session embargo drops dev dates whose forward
    label reaches into the test block. Preprocessing is fit on the development split only.
  * CLASSIFIER track (the deployed XGBoost = P(rise) direction classifier):
      baselines  - majority-class, logistic regression (technical)
      models     - XGBoost technical, XGBoost technical+sentiment
      report     - balanced accuracy, ROC-AUC, confusion matrix, class balance
      calibration- isotonic fitted on a DEV calibration slice, then Brier + ECE
                   (raw vs calibrated) on the test block
  * REGRESSION track (the deployed LSTM = horizon-return regressor):
      baselines  - zero-return, dev historical-mean return
      model      - XGBoost regressor (always) and, with --with-lstm, the actual
                   LSTMForecaster retrained on dev and scored on test
      report     - MAE, RMSE, directional accuracy
  * Sentiment contribution and model-vs-baseline gaps get a paired, date-grouped
    bootstrap CI on matched (date, ticker) rows (experiments.paired_date_bootstrap).

Everything is retrained under the clean split here; we do NOT score the on-disk
deployed weights, because those were fit on all history and have no clean holdout.
This evaluates the model the app uses, out-of-sample.

Usage:
    python scripts/evaluate_deployed.py --limit 80
    python scripts/evaluate_deployed.py --sp500-only --with-lstm
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config.settings import PRIMARY_HORIZON, embargo_for
from backend.data.contracts import json_safe
from backend.evaluation import experiments as ex

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

FEATURES_DIR = Path("data/features")
OUT = Path("data/experiments")
OUT.mkdir(parents=True, exist_ok=True)


def load_master_data(limit, sp500_only, start, end):
    from scripts.run_experiments import (
        load_master_data as _load,  # reuse the sliced loader
    )

    return _load(limit=limit, sp500_only=sp500_only, start=start, end=end)


def _date_split(panel, horizon, test_frac=0.25):
    """(dev_dates, test_dates) with an H-session embargo on the dev side."""
    dates = np.array(sorted(panel.index.unique()))
    if horizon < 1 or len(dates) < 2 * (horizon + 2):
        raise ValueError("Insufficient dates for a purged holdout")
    n_test = max(horizon + 2, int(len(dates) * test_frac))
    dev, test = dates[:-n_test], dates[-n_test:]
    emb = embargo_for(horizon)
    if emb > 0:
        dev = dev[:-emb]
    return pd.DatetimeIndex(dev), pd.DatetimeIndex(test)


def _xy(panel, dates, feat_cols, label_before=None):
    sub = panel.loc[panel.index.isin(dates)]
    if label_before is not None and "label_end" in sub.columns:
        sub = sub[sub["label_end"] < label_before]
    sub = sub[np.isfinite(sub[feat_cols]).all(axis=1) & sub["fwd_ret"].notna()]
    return sub


def _confusion(y_true, y_pred):
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def _ece(probs, y, n_bins=10):
    """Expected calibration error with equal-width bins (binning stated)."""
    probs = np.asarray(probs, float)
    y = np.asarray(y, float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece, n = 0.0, len(y)
    for i in range(n_bins):
        m = (probs >= edges[i]) & (probs < edges[i + 1] if i < n_bins - 1 else probs <= edges[i + 1])
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(y[m].mean() - probs[m].mean())
    return float(ece)


def classifier_track(panel, dev, test, tech, both, horizon=PRIMARY_HORIZON):
    import xgboost as xgb
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score
    from sklearn.preprocessing import RobustScaler, StandardScaler

    def label(sub):
        return (sub["fwd_ret"].values > 0).astype(int)

    panel = panel[np.isfinite(panel[both]).all(axis=1)]
    tr, te = _xy(panel, dev, both, label_before=test.min()), _xy(panel, test, both)
    if len(tr) < 200 or len(te) < 30:
        return {"error": f"insufficient rows (dev={len(tr)}, test={len(te)})"}
    ytr, yte = label(tr), label(te)
    if len(np.unique(ytr)) < 2:
        return {"error": "Development window contains only one direction class"}

    out = {
        "n_dev": int(len(tr)),
        "n_test": int(len(te)),
        "test_class_balance": {"up": float(yte.mean()), "down": float(1 - yte.mean())},
        "models": {},
    }

    # majority-class baseline
    maj = int(round(ytr.mean()))
    yp = np.full(len(yte), maj)
    out["models"]["majority_class"] = {
        "balanced_accuracy": round(balanced_accuracy_score(yte, yp), 4),
        "roc_auc": None,
        "confusion": _confusion(yte, yp),
        "note": f"always predicts {'up' if maj else 'down'}",
    }

    # logistic regression (technical)
    trT, teT = _xy(panel, dev, tech, label_before=test.min()), _xy(panel, test, tech)
    scL = StandardScaler().fit(trT[tech].values)
    import warnings

    from sklearn.exceptions import ConvergenceWarning

    with warnings.catch_warnings(record=True) as fit_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        lr = LogisticRegression(solver="newton-cholesky", max_iter=200, random_state=42).fit(
            scL.transform(trT[tech].values), (trT["fwd_ret"].values > 0).astype(int)
        )
    if any(issubclass(w.category, ConvergenceWarning) for w in fit_warnings):
        raise RuntimeError("Logistic baseline did not converge; evaluation results were not published")
    for warning in fit_warnings:
        logger.warning("Logistic fit: %s", warning.message)
    pL = lr.predict_proba(scL.transform(teT[tech].values))[:, 1]
    yteT = (teT["fwd_ret"].values > 0).astype(int)
    out["models"]["logistic_technical"] = {
        "balanced_accuracy": round(balanced_accuracy_score(yteT, (pL > 0.5).astype(int)), 4),
        "roc_auc": round(float(roc_auc_score(yteT, pL)), 4) if len(set(yteT)) > 1 else None,
        "confusion": _confusion(yteT, (pL > 0.5).astype(int)),
        "fit": {
            "converged": True,
            "iterations": int(lr.n_iter_.max()),
            "solver": lr.solver,
            "scaler": "StandardScaler",
        },
    }

    # XGBoost classifier: technical, and technical+sentiment
    def xgb_clf(feat):
        trf, tef = _xy(panel, dev, feat, label_before=test.min()), _xy(panel, test, feat)
        sc = RobustScaler().fit(trf[feat].values)
        m = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=1,
            verbosity=0,
        )
        m.fit(sc.transform(trf[feat].values), (trf["fwd_ret"].values > 0).astype(int))
        p = m.predict_proba(sc.transform(tef[feat].values))[:, 1]
        yt = (tef["fwd_ret"].values > 0).astype(int)
        return m, sc, p, yt, trf, tef

    _, _, pT, ytT, _, _ = xgb_clf(tech)
    out["models"]["xgb_technical"] = {
        "balanced_accuracy": round(balanced_accuracy_score(ytT, (pT > 0.5).astype(int)), 4),
        "roc_auc": round(float(roc_auc_score(ytT, pT)), 4) if len(set(ytT)) > 1 else None,
        "confusion": _confusion(ytT, (pT > 0.5).astype(int)),
    }
    mB, scB, pB, ytB, trB, teB = xgb_clf(both)
    out["models"]["xgb_technical_sentiment"] = {
        "balanced_accuracy": round(balanced_accuracy_score(ytB, (pB > 0.5).astype(int)), 4),
        "roc_auc": round(float(roc_auc_score(ytB, pB)), 4) if len(set(ytB)) > 1 else None,
        "confusion": _confusion(ytB, (pB > 0.5).astype(int)),
    }

    # calibration: fit isotonic on a DEV calibration slice (last 15% of dev dates),
    # refit the model on the earlier dev part, then score raw vs calibrated on TEST
    devd = np.array(sorted(dev))
    n_cal = max(1, int(len(devd) * 0.15))
    fit_d, cal_d = pd.DatetimeIndex(devd[:-n_cal]), pd.DatetimeIndex(devd[-n_cal:])
    fit_d = fit_d[: -embargo_for(horizon)]
    trf = _xy(panel, fit_d, both, label_before=cal_d.min())
    calf = _xy(panel, cal_d, both, label_before=test.min())
    tef = _xy(panel, test, both)
    if len(trf) > 200 and len(calf) > 30:
        sc = RobustScaler().fit(trf[both].values)
        m = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=1,
            verbosity=0,
        )
        m.fit(sc.transform(trf[both].values), (trf["fwd_ret"].values > 0).astype(int))
        raw_cal = m.predict_proba(sc.transform(calf[both].values))[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip").fit(raw_cal, (calf["fwd_ret"].values > 0).astype(int))
        raw_te = m.predict_proba(sc.transform(tef[both].values))[:, 1]
        cal_te = iso.predict(raw_te)
        yt = (tef["fwd_ret"].values > 0).astype(int)
        out["calibration"] = {
            "raw": {"brier": round(float(brier_score_loss(yt, raw_te)), 5), "ece_10bin": round(_ece(raw_te, yt), 5)},
            "calibrated": {
                "brier": round(float(brier_score_loss(yt, cal_te)), 5),
                "ece_10bin": round(_ece(cal_te, yt), 5),
            },
            "n_calibration_rows": int(len(calf)),
            "n_test_rows": int(len(tef)),
            "bins": 10,
        }

    # sentiment contribution: paired improvement (tech+sent vs tech) on matched rows
    a = pd.DataFrame({"y_true": teB["fwd_ret"].values, "y_pred": pB, "ticker": teB["ticker"].values}, index=teB.index)
    b = pd.DataFrame(
        {"y_true": ytT.astype(float), "y_pred": pT, "ticker": _xy(panel, test, tech)["ticker"].values},
        index=_xy(panel, test, tech).index,
    )
    a["y_pred"] = np.where(a["y_pred"] > 0.5, 1.0, -1.0)
    b["y_pred"] = np.where(b["y_pred"] > 0.5, 1.0, -1.0)
    if "sent_news_count" in panel and (panel["sent_news_count"] > 0).any():
        out["sentiment_contribution_paired_dir_hit_ci"] = ex.paired_metric_ci(a, b, metric="dir_hit")
    else:
        out["sentiment_contribution_paired_dir_hit_ci"] = {"unavailable": "No observed sentiment data"}

    # Lift over a random classifier (0.5) so "slightly above random" is quantified.
    for _name, _m in out["models"].items():
        _bacc, _auc = _m.get("balanced_accuracy"), _m.get("roc_auc")
        _m["lift_vs_random"] = {
            "balanced_accuracy_minus_0.5": round(_bacc - 0.5, 4) if _bacc is not None else None,
            "roc_auc_minus_0.5": round(_auc - 0.5, 4) if _auc is not None else None,
        }
    # Explicit class (im)balance on the test window, logged when skewed.
    _up = int(yte.sum())
    _down = int(len(yte) - _up)
    _up_frac = float(yte.mean())
    out["class_imbalance"] = {
        "up": _up, "down": _down, "up_fraction": round(_up_frac, 4),
        "imbalanced": bool(abs(_up_frac - 0.5) > 0.1),
    }
    if abs(_up_frac - 0.5) > 0.1:
        logger.warning("Test-window class imbalance: up_fraction=%.3f (up=%d, down=%d)", _up_frac, _up, _down)
    # Realized market regime over the test window - descriptive context (not a feature)
    # for interpreting near-random accuracy.
    _fwd = panel.loc[panel.index.isin(test), "fwd_ret"]
    _mean_fwd = float(_fwd.mean()) if len(_fwd) else float("nan")
    out["test_window_regime"] = {
        "mean_forward_return": round(_mean_fwd, 5) if np.isfinite(_mean_fwd) else None,
        "label": (
            ("bull" if _mean_fwd > 0.01 else "bear" if _mean_fwd < -0.01 else "sideways")
            if np.isfinite(_mean_fwd) else None
        ),
        "note": "Realized mean forward return over the test window; context, not a predictor.",
    }
    logger.info("Test-window regime label=%s (mean fwd ret %.4f)", out["test_window_regime"]["label"], _mean_fwd)
    return out


def _sequence_eligible(history, observations, seq_len):
    """Mark rows with a complete trailing window, using only observation dates."""
    eligible = np.zeros(len(observations), dtype=bool)
    grouped = history.groupby("ticker", sort=False)
    requested = observations.assign(_position=np.arange(len(observations)))
    for ticker, rows in requested.groupby("ticker", sort=False):
        if ticker not in grouped.indices:
            continue
        dates = grouped.get_group(ticker).sort_index().index.drop_duplicates(keep="last")
        ends = dates.get_indexer(rows.index)
        eligible[rows["_position"].to_numpy()] = ends >= seq_len - 1
    return eligible


def regression_track(panel, dev, test, tech, both, with_lstm=False):
    panel = panel[np.isfinite(panel[both]).all(axis=1)]
    tr, te = _xy(panel, dev, both, label_before=test.min()), _xy(panel, test, both)
    if len(tr) < 200 or len(te) < 30:
        return {"error": f"insufficient rows (dev={len(tr)}, test={len(te)})"}
    yte = te["fwd_ret"].values

    def scores(pred, mask=None):
        pred = np.asarray(pred, float)
        actual = yte if mask is None else yte[mask]
        return {
            "mae": round(float(np.mean(np.abs(pred - actual))), 6),
            "rmse": round(float(np.sqrt(np.mean((pred - actual) ** 2))), 6),
            "dir_acc": round(float(np.mean(np.sign(pred) == np.sign(actual))), 4),
        }

    out = {"n_dev": int(len(tr)), "n_test": int(len(te)), "models": {}}
    out["models"]["zero_return"] = scores(np.zeros(len(yte)))
    out["models"]["dev_hist_mean"] = scores(np.full(len(yte), float(tr["fwd_ret"].mean())))

    import xgboost as xgb
    from sklearn.preprocessing import RobustScaler

    sc = RobustScaler().fit(tr[both].values)
    m = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=1,
        verbosity=0,
    )
    m.fit(sc.transform(tr[both].values), tr["fwd_ret"].values)
    xgb_predictions = m.predict(sc.transform(te[both].values))
    out["models"]["xgb_regressor"] = scores(xgb_predictions)

    if with_lstm:
        try:
            from backend.prediction.lstm_model import LSTMForecaster

            # Fit the deployed LSTM architecture on development data only.
            dev_df = tr.copy()
            # Isolated directory so evaluation NEVER overwrites the deployed model
            # (the model class binds its own directory at construction).
            evaluation_dir = OUT / "lstm_evaluation"
            lf = LSTMForecaster(models_dir=evaluation_dir)

            eligible = _sequence_eligible(panel, te, lf.seq_len)
            if eligible.sum() < 30:
                raise ValueError("Fewer than 30 test observations have a complete LSTM history window")
            coverage = {
                "n_requested": int(len(te)),
                "n_scored": int(eligible.sum()),
                "n_excluded_short_history": int((~eligible).sum()),
                "sequence_length": lf.seq_len,
                "exclusion_rule": "Fewer than sequence_length rows in the filtered ticker history",
                "excluded_by_ticker": {str(k): int(v) for k, v in te.loc[~eligible, "ticker"].value_counts().items()},
            }
            logger.info(
                "LSTM eligibility: %s/%s test rows; %s short-history rows excluded",
                eligible.sum(),
                len(te),
                (~eligible).sum(),
            )
            out["lstm_coverage"] = coverage
            dev_df["target_return"] = dev_df["fwd_ret"]
            training = lf.train(dev_df, epochs=15, batch_size=256)
            if training.get("error"):
                raise ValueError(training["error"])
            preds = lf.predict_panel(panel, te.loc[eligible], batch_size=256)
            if not np.isfinite(preds).all():
                raise ValueError("LSTM returned non-finite predictions for eligible observations")
            out["models"]["lstm_actual"] = {
                **scores(preds, eligible),
                "n_test": int(eligible.sum()),
                "evaluation_sample": "sequence_eligible_test_rows",
            }
            out["lstm_matched_comparison"] = {
                "n_test": int(eligible.sum()),
                "models": {
                    "zero_return": scores(np.zeros(int(eligible.sum())), eligible),
                    "dev_hist_mean": scores(np.full(int(eligible.sum()), float(tr["fwd_ret"].mean())), eligible),
                    "xgb_regressor": scores(xgb_predictions[eligible], eligible),
                    "lstm_actual": scores(preds, eligible),
                },
            }

        except Exception as e:
            out["models"]["lstm_actual"] = {"error": str(e)}
    else:
        out["lstm_status"] = {"status": "not_evaluated", "reason": "disabled_in_run_configuration"}
    return out


def main():
    from backend.infra.reproducibility import set_seed

    set_seed(42)
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sp500-only", action="store_true")
    ap.add_argument("--with-lstm", action="store_true", help="also retrain+score the actual LSTM (slow)")
    ap.add_argument("--horizon", type=int, default=PRIMARY_HORIZON)
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default="2023-12-31")
    args = ap.parse_args()
    if args.with_lstm and args.horizon != PRIMARY_HORIZON:
        ap.error("The deployed LSTM is defined for the primary horizon only")

    data = load_master_data(args.limit, args.sp500_only, args.start, args.end)
    if not data:
        logger.error("No master data - run the fusion step first")
        sys.exit(1)
    panel = ex.build_panel(data, args.horizon)
    if panel.empty:
        logger.error("Empty panel")
        sys.exit(1)
    if args.sp500_only:
        from backend.universe.universe_builder import UniverseBuilder

        panel = UniverseBuilder().filter_eligible_rows(panel, strict=True)
    avail = set(panel.columns)
    tech = ex.technical_features(avail)
    both = ex.all_features(avail, include_sentiment=True)
    dev, test = _date_split(panel, args.horizon)
    logger.info(f"dev {len(dev)} dates .. test {len(test)} dates ({str(test.min().date())}..{str(test.max().date())})")

    results = {
        "schema_version": 3,
        "generated_at": datetime.now().isoformat(),
        "horizon": args.horizon,
        "split": {
            "dev_dates": int(len(dev)),
            "test_dates": int(len(test)),
            "test_start": str(test.min().date()),
            "test_end": str(test.max().date()),
            "embargo_sessions": embargo_for(args.horizon),
        },
        "note": (
            "models retrained under a clean chronological split with train-only "
            "preprocessing and an H-session embargo; the deployed model TYPES, evaluated "
            "out-of-sample - not the on-disk all-data weights"
        ),
        "classifier_track": classifier_track(panel, dev, test, tech, both, horizon=args.horizon),
        "regression_track": regression_track(panel, dev, test, tech, both, with_lstm=args.with_lstm),
    }
    out_name = (
        "deployed_evaluation.json"
        if args.horizon == PRIMARY_HORIZON
        else f"deployed_evaluation_h{args.horizon}.json"
    )
    (OUT / out_name).write_text(
        json.dumps(json_safe(results), indent=2, default=str, allow_nan=False)
    )
    logger.info(f"Results -> {OUT / out_name}")
    print(json.dumps({k: results[k] for k in ("split", "classifier_track", "regression_track")}, indent=2, default=str))


if __name__ == "__main__":
    main()
