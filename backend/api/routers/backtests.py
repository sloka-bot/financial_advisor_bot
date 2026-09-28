"""HTTP routes for backtests."""

import logging

import pandas as pd
from fastapi import APIRouter, HTTPException

from backend import runtime
from backend.api.schemas import BacktestPredictRequest, BacktestRangeRequest, BacktestRequest

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/api/backtest")
def run_backtest(req: BacktestRequest):
    """Evaluate the requested ticker while preventing concurrent training."""
    runtime._reject_if_training()
    tickers = req.tickers
    capital = req.capital
    mode = req.portfolio_mode
    if not runtime.xgb_model.is_trained():
        raise HTTPException(400, "XGBoost not trained")
    master_data = {t: df for t in tickers if (df := runtime.fusion.load_master(t)) is not None and not df.empty}
    if not master_data:
        raise HTTPException(404, "No data for these tickers")
    if mode or len(tickers) > 1:
        return runtime.backtester.run_portfolio(tickers, master_data, runtime.xgb_model, capital)
    return runtime.backtester.run(tickers[0], master_data[tickers[0]], runtime.xgb_model, capital)


@router.post("/api/backtest-predict")
def backtest_predict(req: BacktestPredictRequest):
    """Backtest horizon returns using a model fitted before the window with an embargo."""
    ticker = req.ticker
    start_date = req.start_date
    window = req.window

    df = runtime.fusion.load_master(ticker)
    if df is None or df.empty:
        raise HTTPException(404, f"No data for {ticker}")
    df.index = pd.to_datetime(df.index)
    start_ts = pd.Timestamp(start_date)
    fwd = df[df.index >= start_ts].head(window)
    if len(fwd) < 5:
        raise HTTPException(422, "Not enough data after start date - choose an earlier date")
    end_ts = fwd.index[-1]

    from backend.evaluation.experiments import range_backtest

    rb = range_backtest({ticker: df}, start_ts, end_ts, feature_set="both")
    if rb.get("error"):
        raise HTTPException(422, rb["error"])
    return {
        "ticker": ticker,
        "start_date": start_date,
        "horizon": rb.get("horizon"),
        "trained_until": rb.get("trained_until"),
        "leakage_free": True,
        "n_predictions": rb.get("n_predictions"),
        "metrics": rb.get("metrics"),  # MAE, balanced directional accuracy and IC
        "rows": rb.get("rows"),  # predicted and actual horizon return per date
    }


@router.post("/api/backtest-range")
def backtest_range(req: BacktestRangeRequest):
    """Fit before the requested window and compare forecasts with observed horizon returns."""
    from backend.evaluation.experiments import range_backtest

    tickers = req.tickers
    start = req.start_date
    end = req.end_date
    horizon = req.horizon
    feature_set = req.feature_set
    master_data = {t: df for t in tickers if (df := runtime.fusion.load_master(t)) is not None and not df.empty}
    if not master_data:
        raise HTTPException(404, "No data for these tickers")
    return range_backtest(master_data, start, end, horizon=horizon, feature_set=feature_set)
