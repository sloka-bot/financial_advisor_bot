"""Run the forecasting and trading experiment matrix."""

import argparse
import hashlib
import inspect
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config.settings import HORIZONS, PRIMARY_HORIZON
from backend.data.contracts import to_jsonable
from backend.evaluation import experiments as ex
from backend.universe.sp500_membership import SP500Membership, normalize
from backend.universe.universe_builder import UniverseBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

FEATURES_DIR = Path("data/features")
OUT = Path("data/experiments")
OUT.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUT / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Result cache keyed on model source, shared code, features, horizon, data and config.
_RUN = {"data_fp": "none", "start": "", "end": "", "no_cache": False}

# Labels for the experimental models (distinct from the deployed models).
XGB_REG_LABEL = (
    "XGBoost REGRESSOR predicting forward return - distinct from the "
    "deployed XGBoost CLASSIFIER (P(rise), isotonic-calibrated)"
)
GRU_LABEL = (
    "single-step GRU(16) return regressor (one timestep) - a deep-learning "
    "BASELINE, NOT the deployed 2-layer sequence LSTM over 30-day windows"
)


def _sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _algo_fp(funcs):
    """Hash the source of the given functions, or their names when source is unavailable."""
    parts = []
    for f in funcs:
        try:
            parts.append(inspect.getsource(f))
        except (OSError, TypeError):
            parts.append(getattr(f, "__qualname__", str(f)))
    return hashlib.sha256("".join(parts).encode()).hexdigest()[:16]


def _data_fp(tickers):
    """Fingerprint inputs by ticker set and file size and mtime."""
    parts = []
    for t in sorted(tickers):
        fp = FEATURES_DIR / f"{t}_master.csv"
        if fp.exists():
            st = fp.stat()
            parts.append((t, int(st.st_size), int(st.st_mtime)))
    return _sha(parts)


def _oos(panel, feat_cols, horizon, model_fn):
    """Cached wrapper around experiments._oos_predictions."""
    if _RUN.get("no_cache"):
        return ex._oos_predictions(panel, feat_cols, horizon, model_fn)
    key = _sha(
        {
            "protocol": 3,
            "model": model_fn.__name__,
            "horizon": horizon,
            "features": sorted(feat_cols),
            "algo": _algo_fp(
                [model_fn, ex._oos_predictions, ex.build_panel, ex.date_folds, ex.technical_features, ex.all_features]
            ),
            "data": _RUN["data_fp"],
            "cfg": {"start": _RUN["start"], "end": _RUN["end"], "membership": _RUN.get("universe_builder") is not None},
        }
    )
    cache_file = CACHE_DIR / f"oos_{key}.csv"
    if cache_file.exists():
        try:
            logger.info(f"    cache hit [{model_fn.__name__} H{horizon} {len(feat_cols)}f] - skipping recompute")
            return pd.read_csv(cache_file, index_col=0, parse_dates=True)
        except (OSError, ValueError) as exc:
            logger.warning("Ignoring unreadable prediction cache %s: %s", cache_file, exc)
    preds = ex._oos_predictions(panel, feat_cols, horizon, model_fn)
    try:
        preds.to_csv(cache_file)
    except OSError as exc:
        logger.warning("Could not save prediction cache %s: %s", cache_file, exc)
    return preds


def load_master_data(limit=None, sp500_only=False, start="2010-01-01", end="2023-12-31"):
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    files = sorted(FEATURES_DIR.glob("*_master.csv"))
    tickers = [f.stem.replace("_master", "") for f in files]
    if sp500_only:
        try:
            m = SP500Membership()
            eligible = set(m.eligible_between(start, end))
            tickers = [t for t in tickers if normalize(t) in eligible]
            logger.info(f"Restricted to {len(tickers)} point-in-time S&P 500 members")
        except Exception as e:
            raise ValueError(f"Cannot establish S&P 500 eligibility: {e}") from e
    if limit:
        tickers = tickers[:limit]
    data, coverage = {}, {}
    for t in tickers:
        p = FEATURES_DIR / f"{t}_master.csv"
        try:
            df = pd.read_csv(p, index_col=0, parse_dates=True)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, errors="coerce")
            df = df[~df.index.isna()].sort_index()
            # Every experiment uses the recorded start and end dates.
            df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
            if len(df) > 200:
                data[t] = df
                coverage[t] = {
                    "first": str(df.index.min().date()),
                    "last": str(df.index.max().date()),
                    "rows": int(len(df)),
                }
        except Exception as exc:
            raise ValueError(f"Cannot read experiment input {p}: {exc}") from exc
    if data:
        firsts = [pd.Timestamp(c["first"]) for c in coverage.values()]
        lasts = [pd.Timestamp(c["last"]) for c in coverage.values()]
        _RUN["coverage"] = {
            "requested_window": {"start": str(start_ts.date()), "end": str(end_ts.date())},
            "actual_earliest": str(min(firsts).date()),
            "actual_latest": str(max(lasts).date()),
            "sp500_only": bool(sp500_only),
            "n_tickers": len(data),
            "per_ticker": coverage,
        }
        logger.info(
            f"Loaded {len(data)} tickers "
            f"({min(firsts).date()} .. {max(lasts).date()}, "
            f"window {start_ts.date()}..{end_ts.date()})"
        )
    else:
        logger.info("Loaded 0 tickers")
    return data


def _pred_and_trade(panel, feat_cols, horizon, model_fn, top_k):
    """Return pooled, validation and test prediction metrics plus final-test trading results."""
    preds = _oos(panel, feat_cols, horizon, model_fn)
    if preds.empty:
        return None, preds
    pm = ex.prediction_metrics(preds["y_true"], preds["y_pred"])
    val = preds[preds["fold_kind"] == "val"]
    tst = preds[preds["fold_kind"] == "test"]
    pm_val = ex.prediction_metrics(val["y_true"], val["y_pred"]) if not val.empty else None
    pm_tst = ex.prediction_metrics(tst["y_true"], tst["y_pred"]) if not tst.empty else None
    # Trading comparisons use the final test dates and matched ticker observations.
    pred_map = {(d, t): p for d, t, p in zip(tst.index, tst["ticker"], tst["y_pred"])}
    sub = panel.loc[[(d, t) in pred_map for d, t in zip(panel.index, panel["ticker"])]].copy()

    def sig(row):
        return pred_map[(row.name, row["ticker"])]

    trade = ex.trading_backtest(sub, sig, horizon, top_k=top_k) if not sub.empty else {}
    return {
        "prediction": pm,
        "prediction_val": pm_val,
        "prediction_test": pm_tst,
        "trading": trade,
        "trading_sample": "final_test_matched_rows",
    }, preds


def run_horizon(master_data, horizon, top_k, deep=False):
    logger.info(f"\n===== HORIZON {horizon}d =====")
    panel = ex.build_panel(master_data, horizon)
    if panel.empty:
        return {"error": "empty panel"}

    # Filter rows by point-in-time membership, matching the live app.
    membership_report = None
    ub = _RUN.get("universe_builder")
    if ub is not None:
        before = int(len(panel))
        # Stop when point-in-time membership cannot be applied.
        panel = ub.filter_eligible_rows(panel, strict=True)
        if panel.empty:
            return {"error": "empty panel after point-in-time membership filter"}
        membership_report = {
            "rows_before": before,
            "rows_after": int(len(panel)),
            "rows_dropped_off_membership": before - int(len(panel)),
            "note": (
                "point-in-time membership enforced per date; a "
                "historical list cannot recover prices never "
                "downloaded - see the membership audit CSV"
            ),
        }
        logger.info(f"  membership: kept {membership_report['rows_after']}/{before} rows")

    avail = set(panel.columns)
    tech = ex.technical_features(avail)
    sent = [c for c in ex.SENTIMENT_FEATURES if c in avail]
    both = ex.all_features(avail, include_sentiment=True)

    if not both:
        return {"error": "No recognised predictors"}
    # All model variants and rule baselines share the same complete observations.
    panel = panel[np.isfinite(panel[both]).all(axis=1)].copy()
    if panel.empty:
        return {"error": "No complete feature observations"}
    _, test_fold = ex.date_folds(panel, horizon)
    test_panel = panel.loc[panel.index.isin(test_fold.val_dates)]
    news_rows = int((panel.get("sent_news_count", pd.Series(0, index=panel.index)) > 0).sum())
    res = {
        "evaluation_window": "final_test",
        "test_start": str(test_fold.val_dates.min().date()),
        "test_end": str(test_fold.val_dates.max().date()),
        "sentiment_rows": news_rows,
        "limitations": (["No observed news: sentiment comparisons are unavailable."] if not news_rows else []),
        "n_rows": int(len(panel)),
        "n_tickers": int(panel["ticker"].nunique()),
        "features": {"technical": len(tech), "sentiment": len(sent)},
        "membership_enforcement": membership_report,
    }

    # A: technical rule.
    logger.info("  A: technical rule")
    res["A_technical_rule"] = {
        "trading": ex.trading_backtest(test_panel, ex.technical_rule_signal, horizon, top_k=top_k)
    }
    # B: sentiment rule.
    logger.info("  B: sentiment rule")
    res["B_sentiment_rule"] = (
        {"trading": ex.trading_backtest(test_panel, ex.sentiment_rule_signal, horizon, top_k=top_k)}
        if news_rows
        else {"unavailable": "No observed sentiment data"}
    )
    # C, D and E are return regressors, distinct from the deployed classifier.
    logger.info("  C: XGB-regressor technical")
    res["C_xgb_technical"], preds_C = _pred_and_trade(panel, tech, horizon, ex._fit_predict_xgb, top_k)
    logger.info("  D: XGB-regressor sentiment")
    res["D_xgb_sentiment"], _ = (
        _pred_and_trade(panel, sent, horizon, ex._fit_predict_xgb, top_k)
        if sent and news_rows
        else ({"unavailable": "No observed sentiment data"}, None)
    )
    logger.info("  E: XGB-regressor technical + sentiment")
    res["E_xgb_tech_sent"], preds_E = (
        _pred_and_trade(panel, both, horizon, ex._fit_predict_xgb, top_k)
        if news_rows
        else ({"unavailable": "No observed sentiment data"}, None)
    )
    for k in ("C_xgb_technical", "D_xgb_sentiment", "E_xgb_tech_sent"):
        if isinstance(res.get(k), dict):
            res[k]["model"] = XGB_REG_LABEL

    # Sentiment contribution (E vs C): point deltas and a paired date-block bootstrap CI.
    if preds_E is not None and preds_C is not None and not preds_E.empty and not preds_C.empty:
        e = res["E_xgb_tech_sent"]["prediction_test"]
        c = res["C_xgb_technical"]["prediction_test"]
        if e and c and e["bal_dir_acc"] is not None and c["bal_dir_acc"] is not None:
            res["sentiment_contribution_E_minus_C"] = {
                "bal_dir_acc_delta": round(e["bal_dir_acc"] - c["bal_dir_acc"], 4),
                "ic_delta": round((e["ic"] or 0) - (c["ic"] or 0), 4),
                "mae_delta": round((e["mae"] or 0) - (c["mae"] or 0), 6),
                # Validation observations only.
                "paired_abs_error_improvement_ci_dev": ex.paired_metric_ci(
                    preds_E[preds_E["fold_kind"] == "val"], preds_C[preds_C["fold_kind"] == "val"], metric="abs_error"
                ),
                "paired_direction_hit_improvement_ci_dev": ex.paired_metric_ci(
                    preds_E[preds_E["fold_kind"] == "val"], preds_C[preds_C["fold_kind"] == "val"], metric="dir_hit"
                ),
                # Final test block only.
                "paired_abs_error_improvement_ci_test": ex.paired_metric_ci(
                    preds_E[preds_E["fold_kind"] == "test"], preds_C[preds_C["fold_kind"] == "test"], metric="abs_error"
                ),
                "paired_direction_hit_improvement_ci_test": ex.paired_metric_ci(
                    preds_E[preds_E["fold_kind"] == "test"], preds_C[preds_C["fold_kind"] == "test"], metric="dir_hit"
                ),
            }

    # Feature-group ablation at the primary horizon.
    if horizon == PRIMARY_HORIZON:
        logger.info("  feature-group ablation (primary horizon)")
        full = _oos_metrics(panel, tech, horizon)
        ablation = {"full_technical": full}
        for g, cols in ex.FEATURE_GROUPS.items():
            reduced = [c for c in tech if c not in cols]
            if len(reduced) < 3:
                continue
            m = _oos_metrics(panel, reduced, horizon)
            ablation[f"minus_{g}"] = {
                **m,
                "bal_dir_acc_drop": (
                    round(full["bal_dir_acc"] - m["bal_dir_acc"], 4)
                    if full["bal_dir_acc"] is not None and m["bal_dir_acc"] is not None
                    else None
                ),
            }
        res["feature_group_ablation"] = ablation

    if deep and horizon == PRIMARY_HORIZON and news_rows:
        _run_deep(panel, both, horizon, top_k, res, preds_E)
    return res


def _oos_metrics(panel, feat_cols, horizon):
    preds = _oos(panel, feat_cols, horizon, ex._fit_predict_xgb)
    preds = preds[preds["fold_kind"] == "test"]
    return (
        ex.prediction_metrics(preds["y_true"], preds["y_pred"])
        if not preds.empty
        else {"n": 0, "mae": None, "bal_dir_acc": None, "ic": None}
    )


# F, G, H: GRU, ensemble and HMM regime selection.
def _run_deep(panel, feat_cols, horizon, top_k, res, preds_E):
    if _RUN.get("no_gru"):
        logger.info("  F/G/H skipped (--no-gru)")
        return
    # F: single-step GRU baseline.
    try:
        logger.info("  F: GRU baseline technical+sentiment")
        res["F_gru_baseline"], preds_F = _pred_and_trade(panel, feat_cols, horizon, _fit_predict_gru, top_k)
        if isinstance(res.get("F_gru_baseline"), dict):
            res["F_gru_baseline"]["model"] = GRU_LABEL
    except Exception as e:
        logger.warning(f"F (GRU baseline) skipped: {e}")
        res["F_gru_baseline"] = {"error": str(e)}
        preds_F = None

    # G: average of E and F predictions.
    if preds_E is not None and preds_F is not None and not preds_F.empty:
        logger.info("  G: fixed XGB-regressor + GRU ensemble")
        # Align predictions on date and ticker.
        e = preds_E.copy()
        e["_date"] = e.index
        f = preds_F.copy()
        f["_date"] = f.index
        merged = e.merge(f[["_date", "ticker", "y_pred"]], on=["_date", "ticker"], suffixes=("", "_gru")).set_index(
            "_date"
        )
        merged["y_ens"] = 0.5 * merged["y_pred"] + 0.5 * merged["y_pred_gru"]
        e_t = merged[merged["fold_kind"] == "test"]
        res["G_ensemble"] = {
            "prediction": ex.prediction_metrics(merged["y_true"], merged["y_ens"]),
            "prediction_test": (ex.prediction_metrics(e_t["y_true"], e_t["y_ens"]) if not e_t.empty else None),
            "model": "fixed 50/50 average of the XGBoost REGRESSOR (E) and the GRU "
            "baseline (F), aligned on (date, ticker)",
        }

        # H: pick the better of E or F per regime on validation.
        try:
            logger.info("  H: HMM-selected")
            res["H_hmm_select"] = _hmm_select(panel, merged, horizon)
        except Exception as e:
            logger.warning(f"H (HMM) skipped: {e}")
            res["H_hmm_select"] = {"error": str(e)}


def _fit_predict_gru(Xtr, ytr, Xva, max_rows=15000, epochs=6, batch=256):
    """Fit a single-step GRU return regressor for experiment F."""
    import time

    import torch
    import torch.nn as nn
    from sklearn.preprocessing import RobustScaler

    logger.info("        loading torch...")
    torch.manual_seed(42)
    torch.set_num_threads(1)  # single thread for a small model
    logger.info(f"        torch {torch.__version__} ready (cpu, 1 thread)")

    if len(Xtr) > max_rows:  # row cap for the pointwise regressor
        idx = np.random.default_rng(42).choice(len(Xtr), max_rows, replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]

    sc = RobustScaler().fit(Xtr)
    Xtr_s = torch.tensor(sc.transform(Xtr), dtype=torch.float32).unsqueeze(1)
    ytr_t = torch.tensor(ytr, dtype=torch.float32).view(-1, 1)

    class Net(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.gru = nn.GRU(d, 16, batch_first=True)
            self.fc = nn.Linear(16, 1)

        def forward(self, x):
            o, _ = self.gru(x)
            return self.fc(o[:, -1, :])

    net = Net(Xtr.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    n = len(Xtr_s)

    net.train()
    for ep in range(epochs):
        t0 = time.perf_counter()
        perm = torch.randperm(n)
        last = 0.0
        for i in range(0, n, batch):
            j = perm[i : i + batch]
            opt.zero_grad()
            loss = lossf(net(Xtr_s[j]), ytr_t[j])
            loss.backward()
            opt.step()
            last = float(loss.item())
        logger.info(f"          epoch {ep + 1}/{epochs} loss={last:.5f} ({time.perf_counter() - t0:.1f}s, {n:,} rows)")

    net.eval()
    out = []
    with torch.no_grad():
        Xva_s = torch.tensor(sc.transform(Xva), dtype=torch.float32).unsqueeze(1)
        for i in range(0, len(Xva_s), 8192):
            out.append(net(Xva_s[i : i + 8192]).view(-1).numpy())
    return np.concatenate(out) if out else np.zeros(len(Xva))


def _hmm_select(panel, merged, horizon):
    """Select E or F per HMM regime using a regime model fitted on development dates only."""
    from hmmlearn.hmm import GaussianHMM

    # Market series: mean realised daily return and its rolling volatility.
    if "daily_return" not in panel.columns:
        raise ValueError(
            "HMM regime detection requires a causal 'daily_return' column; "
            "refusing to substitute forward-looking 'fwd_ret'"
        )
    mkt = panel.groupby(panel.index)["daily_return"].mean()
    # Drop leading dates without a full 21-session volatility window.
    vol = mkt.rolling(21).std()
    keep = mkt.notna() & vol.notna()
    mkt, vol = mkt[keep], vol[keep]
    feats = np.column_stack([mkt.values, vol.values])
    dates = mkt.index

    # Development and test boundary from the fold labels.
    test_dates = set(pd.DatetimeIndex(merged[merged["fold_kind"] == "test"].index.unique()))
    if not test_dates:
        raise ValueError("No final test observations for HMM comparison")
    test_start = min(test_dates)
    is_test = np.array([d >= test_start for d in dates])
    dev_mask = np.array([d < test_start for d in dates])
    # Require enough development history to fit the HMM.
    if dev_mask.sum() < 60 or is_test.sum() == 0:
        raise ValueError(
            "insufficient dev history for a leakage-free HMM fit "
            f"(dev={int(dev_mask.sum())}, test={int(is_test.sum())}); "
            "need >=60 dev dates and >=1 test date"
        )

    # Standardise with development statistics.
    mu_d = feats[dev_mask].mean(axis=0)
    sd_d = feats[dev_mask].std(axis=0) + 1e-9
    feats_z = (feats - mu_d) / sd_d

    hmm = GaussianHMM(n_components=3, covariance_type="diag", n_iter=100, random_state=42)
    hmm.fit(feats_z[dev_mask])  # fit on development dates

    # Causal decode up to each date.
    states = np.empty(len(dates), dtype=int)
    for i in range(len(dates)):
        states[i] = int(hmm.predict(feats_z[: i + 1])[-1])
    regimes = pd.Series(states, index=dates)

    m = merged.copy()
    m["regime"] = regimes.reindex(m.index).ffill().values
    val = m[m["fold_kind"] == "val"]
    # Use validation outcomes that mature before the test block.
    label_dates = panel.reset_index(names="_date")[["_date", "ticker", "label_end"]]
    val = val.reset_index(names="_date").merge(label_dates, on=["_date", "ticker"], validate="one_to_one")
    val = val[val["label_end"] < test_start]
    test = m[m["fold_kind"] == "test"]
    # Per regime, choose XGBoost or GRU by validation balanced accuracy.
    choice = {}
    for r, g in val.groupby("regime"):
        a_x = ex.balanced_directional_accuracy(g["y_true"], g["y_pred"])
        a_l = ex.balanced_directional_accuracy(g["y_true"], g["y_pred_gru"])
        choice[int(r)] = "xgb" if (a_x >= a_l) else "gru"
    test = test.copy()
    test["y_sel"] = [
        row["y_pred"] if choice.get(int(row["regime"]), "xgb") == "xgb" else row["y_pred_gru"]
        for _, row in test.iterrows()
    ]
    test["y_avg"] = 0.5 * test["y_pred"] + 0.5 * test["y_pred_gru"]
    occ = regimes.value_counts(normalize=True).round(3).to_dict()

    # Compare all four methods on the same final-test observations.
    comparison_on_test = {
        "xgb_only": ex.prediction_metrics(test["y_true"], test["y_pred"]),
        "gru_only": ex.prediction_metrics(test["y_true"], test["y_pred_gru"]),
        "average": ex.prediction_metrics(test["y_true"], test["y_avg"]),
        "hmm_select": ex.prediction_metrics(test["y_true"], test["y_sel"]),
    }
    return {
        "regime_to_model": {str(k): v for k, v in choice.items()},
        "regime_occupancy": {str(k): float(v) for k, v in occ.items()},
        "prediction_test": comparison_on_test["hmm_select"],
        "comparison_on_test": comparison_on_test,  # same rows for all four methods
        "n_test_rows": int(len(test)),
        "walk_forward": "partial",
        "methodology_note": (
            "HMM parameters are fitted once on development dates. The regime-to-model "
            "mapping uses pooled validation outcomes that mature before the test starts. "
            "The final test is excluded from fitting and selection. Development folds "
            "are not fully nested because the HMM is not refitted within each fold."
        ),
    }


def hyperparameter_search(master_data, horizon, top_k=10, out_dir=OUT):
    """Grid search XGBoost classifier hyperparameters on embargoed development folds."""
    import itertools

    import xgboost as xgb
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import RobustScaler

    panel = ex.build_panel(master_data, horizon)
    if panel.empty:
        return {"error": "empty panel"}
    if _RUN.get("universe_builder") is not None:
        panel = _RUN["universe_builder"].filter_eligible_rows(panel, strict=True)
    feat_cols = [c for c in ex.all_features(panel.columns, include_sentiment=True) if c in panel.columns]
    folds, _test = ex.date_folds(panel, horizon)  # development folds only

    grid = {
        "max_depth": [4, 6, 8],
        "learning_rate": [0.03, 0.05, 0.1],
        "min_child_weight": [1, 3, 5],
    }
    fixed = {
        "n_estimators": 300,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": 1,
        "verbosity": 0,
        "eval_metric": "auc",
    }
    keys = list(grid)
    results = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        params = dict(zip(keys, combo))
        fold_aucs = []
        for fold in folds:
            tr = panel.loc[panel.index.isin(fold.train_dates)]
            if "label_end" in tr.columns:
                tr = tr[tr["label_end"] < fold.val_dates.min()]  # purge labels crossing the boundary
            va = panel.loc[panel.index.isin(fold.val_dates)]
            tr = tr[np.isfinite(tr[feat_cols]).all(axis=1) & tr["fwd_ret"].notna()]
            va = va[np.isfinite(va[feat_cols]).all(axis=1) & va["fwd_ret"].notna()]
            if len(tr) < 200 or len(va) < 20:
                continue
            ytr = (tr["fwd_ret"].values > 0).astype(int)
            yva = (va["fwd_ret"].values > 0).astype(int)
            if len(np.unique(yva)) < 2:
                continue
            sc = RobustScaler().fit(tr[feat_cols].values)  # scaler fitted on training rows only
            m = xgb.XGBClassifier(**fixed, **params)
            m.fit(sc.transform(tr[feat_cols].values), ytr)
            proba = m.predict_proba(sc.transform(va[feat_cols].values))[:, 1]
            fold_aucs.append(float(roc_auc_score(yva, proba)))
        if fold_aucs:
            results.append(
                {
                    "params": params,
                    "cv_auc_mean": round(float(np.mean(fold_aucs)), 4),
                    "cv_auc_std": round(float(np.std(fold_aucs)), 4),
                    "n_folds": len(fold_aucs),
                }
            )
    results.sort(key=lambda r: r["cv_auc_mean"], reverse=True)
    report = {
        "search_type": "purged expanding-window grid search (experiments.date_folds: embargo + label-end purge)",
        "objective": "mean validation ROC-AUC of the XGBoost direction classifier",
        "cv_protocol": "expanding-window date folds; final test block excluded from tuning",
        "horizon": horizon,
        "n_features": len(feat_cols),
        "grid": grid,
        "fixed_params": fixed,
        "deployed_params_note": (
            "The deployed model uses fixed project hyperparameters; this search "
            "validates them against a grid and is NOT auto-deployed."
        ),
        "results": results,
        "best": results[0] if results else None,
    }
    (out_dir / "hyperparameter_search.json").write_text(json.dumps(to_jsonable(report), indent=2, default=str))
    logger.info(
        "Hyperparameter search -> %s (best mean AUC=%s)",
        out_dir / "hyperparameter_search.json",
        report["best"]["cv_auc_mean"] if report["best"] else "n/a",
    )
    return report


def main():
    from backend.infra.reproducibility import set_seed

    set_seed(42)
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sp500-only", action="store_true")
    ap.add_argument("--deep", action="store_true", help="also run F (GRU), G (ensemble), H (HMM)")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--horizons", type=int, nargs="*", default=HORIZONS)
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default="2023-12-31")
    ap.add_argument("--no-cache", action="store_true", help="ignore cached results and recompute all")
    ap.add_argument("--no-gru", action="store_true", help="skip F/G/H (GRU/ensemble/HMM)")
    ap.add_argument("--tune", action="store_true", help="run the leakage-safe hyperparameter grid search and exit")
    args = ap.parse_args()

    data = load_master_data(limit=args.limit, sp500_only=args.sp500_only, start=args.start, end=args.end)
    if not data:
        logger.error("No master data - run the fusion step first")
        sys.exit(1)

    _RUN.update(
        {
            "data_fp": _data_fp(list(data)),
            "start": args.start,
            "end": args.end,
            "no_cache": args.no_cache,
            "no_gru": args.no_gru,
        }
    )
    # Filter every horizon panel by point-in-time membership and write the audit.
    if args.sp500_only:
        try:
            ub = UniverseBuilder()
            ub._membership.write_audit(allow_fetch=False)
            _RUN["universe_builder"] = ub
        except Exception as e:
            raise ValueError(f"Point-in-time membership filter unavailable: {e}") from e
    logger.info(
        f"Cache {'DISABLED' if args.no_cache else 'ENABLED'} "
        f"(data fp {_RUN['data_fp']}) - unchanged experiments are skipped"
    )

    if args.tune:
        hyperparameter_search(data, args.horizons[0], top_k=args.top_k)
        return

    results = {
        "schema_version": 3,
        "trading_protocol": "next_close_v2",
        "generated_at": datetime.now().isoformat(),
        "config": {
            "horizons": args.horizons,
            "top_k": args.top_k,
            "deep": args.deep,
            "sp500_only": args.sp500_only,
            "n_tickers": len(data),
        },
        "data_coverage": _RUN.get("coverage"),
        "horizons": {},
    }
    for h in args.horizons:
        results["horizons"][str(h)] = run_horizon(data, h, args.top_k, deep=args.deep)
        # Write results after each horizon.
        (OUT / "experiment_results.json").write_text(
            json.dumps(to_jsonable(results), indent=2, default=str, allow_nan=False)
        )

    logger.info(f"\nResults -> {OUT / 'experiment_results.json'}")
    _print_summary(results)


def _print_summary(results):
    print("\n" + "=" * 74)
    print("EXPERIMENT SUMMARY  (TEST-block bal.dir.acc | IC | net% vs B&H%)")
    print("  metrics shown are the untouched final-test block only (val in JSON)")
    print("=" * 74)
    for h, r in results["horizons"].items():
        if "error" in r:
            continue
        print(f"\nHorizon {h}d  ({r['n_rows']} rows, {r['n_tickers']} tickers)")
        for key in (
            "A_technical_rule",
            "B_sentiment_rule",
            "C_xgb_technical",
            "D_xgb_sentiment",
            "E_xgb_tech_sent",
            "F_gru_baseline",
            "G_ensemble",
            "H_hmm_select",
        ):
            b = r.get(key)
            if not b:
                continue
            pm = (b.get("prediction_test") or {}) if isinstance(b, dict) else {}
            tr = b.get("trading", {}) if isinstance(b, dict) else {}
            print(
                f"  {key:20s} "
                f"{str(pm.get('bal_dir_acc', '--')):>6} | {str(pm.get('ic', '--')):>6} | "
                f"{str(tr.get('net_return_pct', '--')):>7} vs {str(tr.get('buy_hold_return_pct', '--')):>7}"
            )
        sc = r.get("sentiment_contribution_E_minus_C")
        if sc:
            print(f"  sentiment (E-C) bal.dir.acc delta: {sc['bal_dir_acc_delta']:+}")
    print("=" * 66)


if __name__ == "__main__":
    main()
