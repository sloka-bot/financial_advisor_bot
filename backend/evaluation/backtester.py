"""
Walk-forward, horizon-matched backtester.

For each ticker a FRESH XGBoost direction classifier is trained strictly on rows
before an out-of-sample split (the last OUT_OF_SAMPLE_FRACTION of the history),
with an H-session embargo so no training label overlaps the test window, then evaluated
on the held-out tail. Signal and outcome are matched to the model's horizon
(H sessions), and trades are NON-overlapping: one H-session trade closes before
the next opens. (A `model` argument is accepted for call-compatibility and
ignored - the backtest always trains its own leakage-free model.)

Strategy:
  - Go long when P(rise over H sessions) > 0.5.
  - Otherwise hold cash. No short selling, no leverage.
  - A round-trip transaction cost is charged when the position changes.

Metrics computed:
  Sharpe ratio            - annualised at the H-session period (Sharpe, 1966).
  Max drawdown            - worst peak-to-trough of the strategy equity curve.
  Calmar ratio            - total return / |max drawdown|.
  Direction accuracy (H)  - fraction of test points where P(up)>0.5 matches the
                            sign of the H-session forward return.
  Information coefficient - Spearman correlation of P(up) with the H-session return.
  Excess return           - strategy total return minus buy-and-hold of the window.
"""

import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from backend.config.settings import (
    PREDICTION_HORIZON,
    RF_ANNUAL,
    TRADING_DAYS,
    TX_COST,
    embargo_for,
)
from backend.portfolio.transaction_costs import transaction_cost

from backend.evaluation.metrics import annualized_sharpe, max_drawdown
logger = logging.getLogger(__name__)

OUT_OF_SAMPLE_FRACTION = 0.25  # Final quarter reserved for evaluation.

# labels/meta/executable columns that must never be model features
_EXCLUDED = {
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
    "close_unadj",
    "adj_close",
    "exec_close",
    "target_return",
    "target_direction",
    "ticker",
    "sent_label",
    "bb_upper",
    "bb_lower",
    "extreme_move_flag",
    "potential_split_flag",
    "penny_stock_flag",
    "corp_action_flag",
    "ohlc_breach_flag",
    "outlier_flag",
    "imputed_flag",
    "suspension_flag",
}


class Backtester:
    """Evaluate horizon-matched trades using chronological model fitting."""

    def run(
        self, ticker: str, df: pd.DataFrame, model=None, capital: float = 10_000, horizon: int = PREDICTION_HORIZON
    ) -> dict:
        """
        Leakage-free single-ticker backtest, horizon-matched.

        For each ticker this:
          * trains a FRESH XGBoost classifier STRICTLY on rows before the split,
            with an H-session embargo so no training label overlaps the test window
            (`model` is ignored and kept only for call-compatibility);
          * predicts P(rise over H sessions) on the test rows;
          * scores direction accuracy and IC against the H-SESSION forward return;
          * executes at the following close and holds H sessions;
          * deducts entry, exit and terminal liquidation costs;
          * compares against gross buy-and-hold over the same execution window.
        """
        if df is None or len(df) < 120:
            return {"error": f"Not enough data for {ticker}"}
        import xgboost as xgb
        from sklearn.preprocessing import RobustScaler

        H = int(horizon)
        if H < 1 or not np.isfinite(capital) or capital <= 0:
            return {"error": "Horizon and capital must be positive"}
        d = df.copy()
        if not isinstance(d.index, pd.DatetimeIndex):
            d.index = pd.to_datetime(d.index, errors="coerce")
        d = d[~d.index.isna()].sort_index()
        d = d[~d.index.duplicated(keep="last")]

        fwd = d["close"].shift(-H) / d["close"] - 1.0  # H-session forward return
        feat_cols = [c for c in d.columns if c not in _EXCLUDED and pd.api.types.is_numeric_dtype(d[c])]
        if not feat_cols:
            return {"error": f"No usable features for {ticker}"}
        Xall = d[feat_cols].to_numpy(dtype=float)
        finite = np.isfinite(Xall).all(axis=1)
        fwd_ok = fwd.notna().to_numpy()

        n = len(d)
        split = int(n * (1 - OUT_OF_SAMPLE_FRACTION))
        split = max(60, min(split, n - H - 5))
        emb = embargo_for(H)  # purge H rows before the split

        train_idx = np.array([i for i in range(0, max(0, split - emb)) if finite[i] and fwd_ok[i]])
        test_idx = np.array([i for i in range(split, n - H) if finite[i] and fwd_ok[i]])
        if len(train_idx) < 50 or len(test_idx) < 10:
            return {
                "error": f"Not enough leakage-free rows for {ticker} (train={len(train_idx)}, test={len(test_idx)})"
            }

        y_tr = (fwd.to_numpy()[train_idx] > 0).astype(int)
        if len(np.unique(y_tr)) < 2:
            return {"error": "Training window contains only one direction class"}
        sc = RobustScaler().fit(Xall[train_idx])  # scaler on TRAIN only
        clf = xgb.XGBClassifier(
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
        clf.fit(sc.transform(Xall[train_idx]), y_tr)

        proba = clf.predict_proba(sc.transform(Xall[test_idx]))[:, 1]
        fwd_te = fwd.to_numpy()[test_idx]

        # prediction-quality metrics vs the MATCHED H-session outcome
        dir_acc = float(np.mean((proba > 0.5) == (fwd_te > 0)))
        ic = float(spearmanr(proba, fwd_te).correlation) if len(test_idx) > 2 else 0.0

        # Keep rebalance spacing on the session grid even when feature rows are missing.
        probability = dict(zip(test_idx, proba))
        rebal = list(range(split, n - H - 1, H))
        if not rebal:
            return {"error": "No complete next-close execution windows"}
        strat_rets, bench_rets = [], []
        prev_long = False
        n_long = 0
        for i in rebal:
            entry, exit_price = float(d["close"].iloc[i + 1]), float(d["close"].iloc[i + H + 1])
            if not np.isfinite([entry, exit_price]).all() or entry <= 0 or exit_price <= 0:
                return {"error": "Execution prices are unavailable for a scheduled period"}
            go_long = probability.get(i, 0.0) > 0.5
            realised = exit_price / entry - 1
            cost = transaction_cost(abs(int(go_long) - int(prev_long)), TX_COST)
            strat_rets.append((1 - cost) * (1 + (realised if go_long else 0.0)) - 1)
            bench_rets.append(realised)
            n_long += int(go_long)
            prev_long = go_long
        if prev_long:
            strat_rets[-1] = (1 + strat_rets[-1]) * (1 - transaction_cost(1.0, TX_COST)) - 1
        strat_rets = np.asarray(strat_rets)
        bench_rets = np.asarray(bench_rets)

        strat_curve = capital * np.cumprod(np.concatenate([[1.0], 1 + strat_rets]))
        bench_curve = capital * np.cumprod(np.concatenate([[1.0], 1 + bench_rets]))
        strat_total = strat_curve[-1] / capital - 1
        bench_total = bench_curve[-1] / capital - 1

        ann = TRADING_DAYS / H  # periods per year at this horizon
        sharpe = annualized_sharpe(strat_rets, ann)
        max_dd = max_drawdown(strat_curve, eps=1e-9)

        entry_date = str(d.index[rebal[0] + 1].date())
        te_dates = [str(d.index[i + H + 1].date()) for i in rebal]
        eq_dates = [entry_date] + te_dates
        equity_curve = [{"date": dt, "value": round(v, 2)} for dt, v in zip(eq_dates, strat_curve.tolist())]
        benchmark_curve = [{"date": dt, "value": round(v, 2)} for dt, v in zip(eq_dates, bench_curve.tolist())]

        return {
            "ticker": ticker,
            "initial_capital": capital,
            "horizon_sessions": H,
            "leakage_free": True,
            "trading_protocol": "next_close_v2",
            "n_missing_signal_periods": sum(i not in probability for i in rebal),
            "trained_until": str(d.index[max(0, split - emb) - 1].date()) if split - emb > 0 else None,
            "test_period": f"{entry_date} - {te_dates[-1]}" if te_dates else "",
            "n_test_periods": len(strat_rets),
            "n_long_periods": n_long,
            "equity_curve": equity_curve,
            "benchmark_curve": benchmark_curve,
            "metrics": {
                "sharpe_ratio": round(sharpe, 4),
                "total_return": round(strat_total, 4),
                "benchmark_total_return": round(bench_total, 4),
                "excess_return": round(strat_total - bench_total, 4),
                "max_drawdown": round(max_dd, 4),
                "calmar_ratio": round(-strat_total / (max_dd or -1e-8), 4),
                "direction_accuracy_h": round(dir_acc, 4),  # vs H-session outcome (matched)
                "information_coefficient": round(ic, 4),
            },
        }

    def run_portfolio(self, tickers: list, master_data: dict, model, capital: float = 10_000) -> dict:
        """
        Run individual backtests for each ticker and return a portfolio summary.
        """
        results = {}
        for ticker in tickers:
            df = master_data.get(ticker)
            if df is not None:
                results[ticker] = self.run(ticker, df, model, capital)

        valid = {t: r for t, r in results.items() if "metrics" in r}
        sharpes = [r["metrics"]["sharpe_ratio"] for r in valid.values()]
        dirs = [r["metrics"]["direction_accuracy_h"] for r in valid.values()]

        best_ticker = max(valid.keys(), key=lambda t: valid[t]["metrics"]["total_return"]) if valid else None

        return {
            "individual_metrics": {t: r["metrics"] for t, r in valid.items()},
            "portfolio_summary": {
                "avg_sharpe": round(np.mean(sharpes), 4) if sharpes else 0,
                "avg_direction_acc": round(np.mean(dirs), 4) if dirs else 0,
                "best_ticker": best_ticker,
                "n_backtested": len(valid),
            },
            # return first ticker's curve for chart display
            "equity_curve": valid[tickers[0]]["equity_curve"] if tickers and tickers[0] in valid else [],
            "benchmark_curve": valid[tickers[0]]["benchmark_curve"] if tickers and tickers[0] in valid else [],
        }
