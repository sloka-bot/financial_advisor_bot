"""Build chronological forecasting experiments, trading comparisons and uncertainty estimates."""

import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from backend.config.settings import TRADING_DAYS, TX_COST, embargo_for
from backend.evaluation.metrics import annualized_sharpe, max_drawdown
from backend.portfolio.transaction_costs import transaction_cost

logger = logging.getLogger(__name__)

# Technical feature groups and ablation definitions.
FEATURE_GROUPS = {
    "trend": [
        "sma20",
        "sma50",
        "sma200",
        "ema12",
        "ema26",
        "sma20_slope",
        "golden_cross",
        "macd",
        "macd_signal",
        "macd_hist",
        "close_to_sma20",
        "close_to_sma50",
    ],
    "momentum": [
        "rsi",
        "stoch_k",
        "stoch_d",
        "williams_r",
        "momentum_10d",
        "momentum_21d",
        "roc_10",
        "return_accel",
        "weekly_return",
        "monthly_return",
        "quarterly_return",
        "week52_position",
    ],
    "volatility": ["atr", "atr_pct", "bb_pct", "bb_bandwidth", "volatility", "vol_ratio"],
    "volume": ["obv_momentum", "volume_ratio", "acc_dist"],
    "strength": ["cci", "adx", "candle_body_ratio"],
}
SENTIMENT_FEATURES = [
    "sent_score",
    "sent_weighted",
    "sent_pos",
    "sent_neg",
    "sent_neu",
    "sent_news_count",
    "sent_disagreement",
    "sent_no_news",
]

# Label, metadata and execution columns excluded from features.
NON_FEATURES = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_unadj",
    "adj_close",
    "exec_close",
    "daily_return",
    "target_return",
    "target_direction",
    "ticker",
    "sent_label",
    "bb_upper",
    "bb_lower",
    "y",
    "y_dir",
    "fwd_ret",
    "label_end",
    "target_end_date",
    "trade_return",
    "execution_date",
    "exit_date",
    "extreme_move_flag",
    "outlier_flag",
    "potential_split_flag",
    "corp_action_flag",
    "ohlc_breach_flag",
    "penny_stock_flag",
    "imputed_flag",
    "suspension_flag",
}


def technical_features(available):
    """Select available technical predictors while excluding targets and execution fields."""
    cols = []
    for g in ("trend", "momentum", "volatility", "volume", "strength"):
        cols += FEATURE_GROUPS[g]
    return [c for c in cols if c in available]


def all_features(available, include_sentiment=True):
    """Combine the available technical and sentiment predictor columns."""
    cols = technical_features(available)
    if include_sentiment:
        cols += [c for c in SENTIMENT_FEATURES if c in available]
    return cols


# Panel construction.
def build_panel(master_data: dict, horizon: int) -> pd.DataFrame:
    """Combine ticker histories with independently computed forward-return labels."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    frames = []
    for t, df in master_data.items():
        if df is None or len(df) <= horizon + 5:
            continue
        d = df.copy()
        if not isinstance(d.index, pd.DatetimeIndex):
            d.index = pd.to_datetime(d.index, errors="coerce")
        d = d[~d.index.isna()].sort_index()
        d = d[~d.index.duplicated(keep="last")]
        d["label_end"] = pd.Series(d.index, index=d.index).shift(-horizon)
        d["fwd_ret"] = d["close"].shift(-horizon) / d["close"] - 1.0
        d["trade_return"] = d["close"].shift(-(horizon + 1)) / d["close"].shift(-1) - 1.0
        d["execution_date"] = pd.Series(d.index, index=d.index).shift(-1)
        d["exit_date"] = pd.Series(d.index, index=d.index).shift(-(horizon + 1))
        d["ticker"] = t
        d = d.dropna(subset=["fwd_ret"])
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames).sort_index()
    return panel


# Expanding date folds and embargoes.
@dataclass
class Fold:
    """Describe the training and validation dates of one chronological split."""

    train_dates: pd.DatetimeIndex
    val_dates: pd.DatetimeIndex
    kind: str = "val"  # "val" or "test"


def date_folds(panel: pd.DataFrame, horizon: int, n_splits: int = 4, test_frac: float = 0.25):
    """Build expanding date folds with label embargoes and a reserved final test block."""
    if horizon < 1 or n_splits < 1 or not 0 < test_frac < 1:
        raise ValueError("Invalid fold configuration")
    dates = np.array(sorted(panel.index.unique()))
    if len(dates) < 2 * (horizon + 1):
        raise ValueError("Insufficient dates for purged development and test windows")
    if len(dates) < (n_splits + 2) * (horizon + 1):
        n_splits = max(1, len(dates) // (5 * (horizon + 1)) - 1)
    n_test = max(horizon + 1, int(len(dates) * test_frac))
    dev_dates, test_dates = dates[:-n_test], dates[-n_test:]

    emb = embargo_for(horizon)
    folds = []
    block_bounds = np.array_split(dev_dates, n_splits + 1)
    for k in range(1, len(block_bounds)):
        val = block_bounds[k]
        if len(val) == 0:
            continue
        train_end = val[0]
        # Drop the embargo sessions before the validation block.
        train_pool = dev_dates[dev_dates < train_end]
        if emb > 0:
            train_pool = train_pool[:-emb]
        if len(train_pool) == 0 or len(val) == 0:
            continue
        folds.append(Fold(pd.DatetimeIndex(train_pool), pd.DatetimeIndex(val), "val"))

    # Final test fold trains on all development dates minus the embargo.
    train_pool = dev_dates[:-emb] if emb > 0 else dev_dates
    test_fold = Fold(pd.DatetimeIndex(train_pool), pd.DatetimeIndex(test_dates), "test")
    return folds, test_fold


# Forecast metrics.
def balanced_directional_accuracy(y_true, y_pred):
    """Balanced accuracy of the sign of the forward return."""
    yt = np.sign(np.asarray(y_true))
    yp = np.sign(np.asarray(y_pred))
    accs = []
    for cls in (1.0, -1.0):
        m = yt == cls
        if m.sum() > 0:
            accs.append(float(np.mean(yp[m] == cls)))
    return float(np.mean(accs)) if accs else float("nan")


def prediction_metrics(y_true, y_pred):
    """Calculate regression error and thresholded direction metrics on finite pairs."""
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[ok], y_pred[ok]
    if len(y_true) < 5:
        return {"n": int(len(y_true)), "mae": None, "bal_dir_acc": None, "ic": None}
    ic = spearmanr(y_pred, y_true).correlation if np.ptp(y_pred) > 0 and np.ptp(y_true) > 0 else None
    return {
        "n": int(len(y_true)),
        "mae": round(float(np.mean(np.abs(y_pred - y_true))), 6),
        "bal_dir_acc": round(balanced_directional_accuracy(y_true, y_pred), 4),
        "ic": round(float(ic), 4) if ic is not None and np.isfinite(ic) else None,
    }


# Fit fold-specific regressors with training-only scaling.
def _fit_predict_xgb(Xtr, ytr, Xva):
    import xgboost as xgb
    from sklearn.preprocessing import RobustScaler

    sc = RobustScaler().fit(Xtr)  # fit on training rows only
    m = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=1,
        verbosity=0,
    )
    m.fit(sc.transform(Xtr), ytr)
    return m.predict(sc.transform(Xva))


def _oos_predictions(panel, feat_cols, horizon, model_fn, n_splits=4):
    """Collect predictions on validation and test rows excluded from each fold's training."""
    folds, test_fold = date_folds(panel, horizon, n_splits=n_splits)
    feat_cols = [c for c in feat_cols if c in panel.columns]
    all_folds = folds + [test_fold]
    n_folds = len(all_folds)
    model_name = getattr(model_fn, "__name__", "model")
    logger.info(f"      training {model_name} across {n_folds} folds ({len(feat_cols)} features)...")
    rows = []
    for i, fold in enumerate(all_folds, 1):
        tr = panel.loc[panel.index.isin(fold.train_dates)]
        if "label_end" in tr.columns:
            tr = tr[tr["label_end"] < fold.val_dates.min()]
        va = panel.loc[panel.index.isin(fold.val_dates)]
        tr = tr[np.isfinite(tr[feat_cols]).all(axis=1) & tr["fwd_ret"].notna()]
        va = va[np.isfinite(va[feat_cols]).all(axis=1) & va["fwd_ret"].notna()]
        if len(tr) < 200 or len(va) < 20:
            logger.info(f"        fold {i}/{n_folds} [{fold.kind}] skipped (train={len(tr)}, val={len(va)})")
            continue
        t0 = time.perf_counter()
        yhat = model_fn(tr[feat_cols].values, tr["fwd_ret"].values, va[feat_cols].values)
        logger.info(
            f"        fold {i}/{n_folds} [{fold.kind}] train={len(tr):,} val={len(va):,} "
            f"done in {time.perf_counter() - t0:.1f}s"
        )
        out = pd.DataFrame(
            {
                "y_true": va["fwd_ret"].values,
                "y_pred": np.asarray(yhat),
                "ticker": va["ticker"].values,
                "fold_kind": fold.kind,
            },
            index=va.index,
        )
        rows.append(out)
    if not rows:
        return pd.DataFrame(columns=["y_true", "y_pred", "ticker", "fold_kind"])
    return pd.concat(rows)


# Technical and sentiment rule baselines.
def technical_rule_signal(row):
    """Interpretable long/flat/short rule (baseline A)."""
    score = 0
    if row.get("rsi", 50) < 35:
        score += 1
    if row.get("rsi", 50) > 65:
        score -= 1
    if row.get("macd", 0) > row.get("macd_signal", 0):
        score += 1
    else:
        score -= 1
    if row.get("momentum_10d", 0) > 0:
        score += 1
    else:
        score -= 1
    if row.get("bb_pct", 0.5) < 0.2:
        score += 1
    if row.get("bb_pct", 0.5) > 0.8:
        score -= 1
    return 1 if score >= 2 else (-1 if score <= -2 else 0)


def sentiment_rule_signal(row, thr=0.15):
    """Sentiment threshold rule (baseline B)."""
    if row.get("sent_news_count", 0) <= 0:
        return 0
    s = row.get("sent_weighted", row.get("sent_score", 0.0))
    return 1 if s > thr else (-1 if s < -thr else 0)


# Non-overlapping trading simulation.
def trading_backtest(panel, signal_by_row, horizon, capital=1000.0, top_k=10, tx_cost=TX_COST):
    """Simulate non-overlapping top-ranked holdings, turnover costs and two benchmarks."""
    if "trade_return" not in panel:
        raise ValueError("Trading evaluation requires next-session trade_return values")
    dates = np.array(sorted(panel.index.unique()))
    # Exclude only the trailing decision dates with no complete execution window.
    complete = panel.groupby(level=0)["trade_return"].apply(lambda values: np.isfinite(values).any())
    if not complete.any():
        raise ValueError("No complete next-session execution windows")
    dates = dates[dates <= complete[complete].index.max()]
    rebal = dates[::horizon]
    equity = capital
    bh_equity = capital  # buy-and-hold on the initial universe
    ew_equity = capital  # equal weight, re-selected each period
    trades = []
    per_period = []
    prev_weights = {}  # name -> weight held going into this period
    bh_values = {}
    bh_names = None  # universe fixed at the first rebalance
    missing_benchmark = set()
    ew_valid = True
    for d in rebal:
        day = panel.loc[panel.index == d]
        if day.empty:
            continue
        if bh_names is None:
            bh_names = set(day["ticker"])
            bh_values = {ticker: capital / len(bh_names) for ticker in bh_names}
        day = day.assign(_sig=[signal_by_row(r) for _, r in day.iterrows()])

        # Benchmarks are computed every period.
        ew_valid = ew_valid and bool(np.isfinite(day["trade_return"]).all())
        ew_equity *= 1 + day["trade_return"].mean()  # periodic equal-weight
        bh_slice = day[day["ticker"].isin(bh_names)]
        available = set(bh_slice.loc[np.isfinite(bh_slice["trade_return"]), "ticker"])
        missing_benchmark.update(bh_names - available)
        for _, observation in bh_slice.iterrows():
            if np.isfinite(observation["trade_return"]):
                bh_values[observation["ticker"]] *= 1 + observation["trade_return"]
        bh_equity = sum(bh_values.values())

        longs = day[day["_sig"] > 0].sort_values("_sig", ascending=False).head(top_k)  # ranked by signal
        if not np.isfinite(longs["trade_return"]).all():
            return {
                "valid": False,
                "unavailable": "Selected holdings lack realised execution prices",
                "date": str(pd.Timestamp(d).date()),
                "missing_tickers": longs.loc[~np.isfinite(longs["trade_return"]), "ticker"].tolist(),
                "net_return_pct": None,
                "buy_hold_return_pct": None,
                "trading_protocol": "next_close_v2",
            }
        holdings = list(longs["ticker"])
        # Charge turnover for entries, exits and changes to retained weights.
        w_new = {t: 1.0 / len(holdings) for t in holdings} if holdings else {}
        turnover = sum(abs(w_new.get(t, 0.0) - prev_weights.get(t, 0.0)) for t in set(w_new) | set(prev_weights))
        cost = transaction_cost(turnover, tx_cost)
        gross = float(longs["trade_return"].mean()) if holdings else 0.0
        net = (1 - cost) * (1 + gross) - 1
        equity *= 1 + net
        for _, r in longs.iterrows():
            trades.append(float(r["trade_return"]))
        per_period.append(
            {
                "date": str(pd.Timestamp(d).date()),
                "held": len(holdings),
                "ret": net,
                "turnover": turnover,
            }
        )
        prev_weights = {
            r["ticker"]: w_new[r["ticker"]] * (1 + r["trade_return"]) / max(1 + gross, 1e-9)
            for _, r in longs.iterrows()
        }

    terminal_cost = transaction_cost(sum(prev_weights.values()), tx_cost)
    equity *= 1 - terminal_cost
    if per_period:
        per_period[-1]["ret"] = (1 + per_period[-1]["ret"]) * (1 - terminal_cost) - 1
    benchmark_valid = not missing_benchmark
    trades = np.asarray(trades, float)
    period_rets = np.array([p["ret"] for p in per_period], float)
    # Include initial capital when measuring drawdown.
    cum = np.concatenate([[1.0], np.cumprod(1 + period_rets)]) if len(period_rets) else np.array([1.0])
    max_dd = max_drawdown(cum) if len(cum) else 0.0
    ann = TRADING_DAYS / horizon
    sharpe = annualized_sharpe(period_rets, ann)
    return {
        "valid": True,
        "final_value": round(equity, 2),
        "net_return_pct": round((equity / capital - 1) * 100, 2),
        "buy_hold_value": round(bh_equity, 2) if benchmark_valid else None,
        "buy_hold_return_pct": round((bh_equity / capital - 1) * 100, 2) if benchmark_valid else None,
        "buy_hold_net_return_pct": round((bh_equity * (1 - tx_cost) ** 2 / capital - 1) * 100, 2)
        if benchmark_valid
        else None,
        "benchmark_status": "complete" if benchmark_valid else "unavailable_missing_prices",
        "benchmark_missing_tickers": sorted(missing_benchmark),
        "trading_protocol": "next_close_v2",
        "execution": "signal at close; execute next observed session close; hold H sessions",
        "terminal_cost_fraction": terminal_cost,
        "periodic_ew_return_pct": round((ew_equity / capital - 1) * 100, 2) if ew_valid else None,
        "excess_vs_bh_pct": round((equity - bh_equity) / capital * 100, 2) if benchmark_valid else None,
        "n_trades": int(len(trades)),
        "n_rebalances": int(len(per_period)),
        "win_rate": round(float(np.mean(trades > 0)), 4) if len(trades) else None,
        "avg_return_per_trade_pct": round(float(np.mean(trades)) * 100, 4) if len(trades) else None,
        "max_drawdown_pct": round(max_dd * 100, 2),
        "sharpe": round(sharpe, 3),
        "avg_turnover": round(float(np.mean([p.get("turnover", 0) for p in per_period])), 3) if per_period else 0.0,
        "total_cost_pct": round(
            (transaction_cost(float(np.sum([p.get("turnover", 0) for p in per_period])), tx_cost) + terminal_cost)
            * 100,
            3,
        ),
    }


# Historical range evaluation and bootstrap intervals.
def range_backtest(master_data: dict, start, end, horizon=None, feature_set="both"):
    """Fit an embargoed regressor before a requested window and score its forecasts."""
    from backend.config.settings import PRIMARY_HORIZON

    horizon = int(horizon or PRIMARY_HORIZON)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    emb = embargo_for(horizon)

    panel = build_panel(master_data, horizon)
    if panel.empty:
        return {"error": "no data"}
    avail = set(panel.columns)
    feats = (
        technical_features(avail)
        if feature_set == "technical"
        else all_features(avail, include_sentiment=(feature_set != "technical"))
    )

    # Train before start, minus the embargo window.
    train_dates = np.array(sorted(d for d in panel.index.unique() if d < start))
    if emb > 0:
        train_dates = train_dates[:-emb]
    tr = panel.loc[panel.index.isin(train_dates)]
    tr = tr[tr["label_end"] < start]
    tr = tr[np.isfinite(tr[feats]).all(axis=1) & tr["fwd_ret"].notna()]
    if len(tr) < 200:
        return {"error": f"not enough training data before {start.date()} ({len(tr)} rows)"}

    import xgboost as xgb
    from sklearn.preprocessing import RobustScaler

    sc = RobustScaler().fit(tr[feats].values)
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=1,
        verbosity=0,
    )
    model.fit(sc.transform(tr[feats].values), tr["fwd_ret"].values)

    win = panel[(panel.index >= start) & (panel.index <= end)]
    win = win[np.isfinite(win[feats]).all(axis=1)]
    if win.empty:
        return {"error": "no rows in the requested window"}
    yhat = model.predict(sc.transform(win[feats].values))

    rows = []
    for (dt, row), pred in zip(win.iterrows(), yhat):
        close = float(row["close"])
        actual = row["fwd_ret"]
        known = bool(np.isfinite(actual))
        rows.append(
            {
                "date": str(pd.Timestamp(dt).date()),
                "ticker": row["ticker"],
                "predicted_return_pct": round(float(pred) * 100, 3),
                "predicted_price": round(close * (1 + float(pred)), 2),
                "target_date": str(pd.Timestamp(row["label_end"]).date()) if pd.notna(row["label_end"]) else None,
                "actual_return_pct": round(float(actual) * 100, 3) if known else None,
                "actual_price": round(close * (1 + float(actual)), 2) if known else None,
                "known": known,
            }
        )
    ev = [(r["actual_return_pct"] / 100, r["predicted_return_pct"] / 100) for r in rows if r["known"]]
    metrics = {}
    if ev:
        yt = np.array([a for a, _ in ev])
        yp = np.array([p for _, p in ev])
        metrics = prediction_metrics(yt, yp)
    return {
        "horizon": horizon,
        "trained_until": str(pd.Timestamp(train_dates.max()).date()) if len(train_dates) else None,
        "n_train_rows": int(len(tr)),
        "window": {"start": str(start.date()), "end": str(end.date())},
        "leakage_free": True,
        "n_predictions": len(rows),
        "n_with_known_outcome": len(ev),
        "metrics": metrics,
        "rows": rows,
    }


def block_bootstrap_ci(values, block=5, n_boot=1000, alpha=0.05, seed=42):
    """Estimate a mean and confidence interval using contiguous sampled blocks."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) < block * 2:
        return {"mean": float(np.mean(v)) if len(v) else None, "lo": None, "hi": None}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(len(v) / block))
    starts_max = len(v) - block
    means = []
    for _ in range(n_boot):
        idx = rng.integers(0, starts_max + 1, size=n_blocks)
        sample = np.concatenate([v[s : s + block] for s in idx])[: len(v)]
        means.append(sample.mean())
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": round(float(np.mean(v)), 5), "lo": round(float(lo), 5), "hi": round(float(hi), 5)}


def paired_date_bootstrap(diff_by_date, n_boot=1000, alpha=0.05, seed=42):
    """Bootstrap paired differences using date blocks to retain local time dependence."""
    dates = sorted(diff_by_date.keys())
    per_date = [np.asarray(diff_by_date[d], float) for d in dates]
    per_date = [a[np.isfinite(a)] for a in per_date]
    all_obs = np.concatenate(per_date) if per_date else np.array([])
    if len(dates) < 42 or len(all_obs) == 0:
        return {
            "mean_diff": float(all_obs.mean()) if len(all_obs) else None,
            "lo": None,
            "hi": None,
            "n_dates": len(dates),
            "n_obs": int(len(all_obs)),
            "note": "at least two 21-session blocks are required for the paired bootstrap",
        }
    rng = np.random.default_rng(seed)
    D = len(dates)
    means = []
    for _ in range(n_boot):
        block_length = 21
        starts = rng.integers(0, D, size=int(np.ceil(D / block_length)))
        pick = np.concatenate([(start + np.arange(block_length)) % D for start in starts])[:D]
        sample = np.concatenate([per_date[i] for i in pick])
        if len(sample):
            means.append(float(sample.mean()))
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    mean_diff = float(all_obs.mean())
    return {
        "mean_diff": round(mean_diff, 6),
        "lo": round(float(lo), 6),
        "hi": round(float(hi), 6),
        "n_dates": D,
        "n_obs": int(len(all_obs)),
        "significant": bool(lo > 0 or hi < 0),
    }


def paired_metric_ci(preds_model, preds_base, metric="abs_error", n_boot=1000, seed=42):
    """Estimate paired improvement intervals on matching ticker-date observations."""
    if preds_model is None or preds_base is None or preds_model.empty or preds_base.empty:
        return {"error": "missing predictions"}
    a = preds_model.copy()
    a["_d"] = a.index
    b = preds_base.copy()
    b["_d"] = b.index
    m = a.merge(b[["_d", "ticker", "y_pred"]], on=["_d", "ticker"], suffixes=("", "_base"), validate="one_to_one")
    if m.empty:
        return {"error": "no matched (date, ticker) observations"}
    yt = m["y_true"].to_numpy(float)
    if metric == "dir_hit":
        diff = (np.sign(m["y_pred"].to_numpy(float)) == np.sign(yt)).astype(float) - (
            np.sign(m["y_pred_base"].to_numpy(float)) == np.sign(yt)
        ).astype(float)
    else:  # abs_error improvement
        diff = np.abs(m["y_pred_base"].to_numpy(float) - yt) - np.abs(m["y_pred"].to_numpy(float) - yt)
    by_date = {d: diff[(m["_d"] == d).to_numpy()] for d in m["_d"].unique()}
    out = paired_date_bootstrap(by_date, n_boot=n_boot, seed=seed)
    out["metric"] = metric
    return out
