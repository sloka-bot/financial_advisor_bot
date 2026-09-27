"""HTTP routes for analysis."""

import logging

from fastapi import APIRouter, HTTPException

from backend import runtime
from backend.config import settings
from backend.data.contracts import finite_number, freshness
from backend.prediction.recommender import classify_signal

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/api/stock/{ticker}/history")
def stock_history(ticker: str, days: int = 120):
    """Return chart-ready price, technical and sentiment observations."""
    df = runtime.fusion.load_master(ticker)
    if df is None or (hasattr(df, "empty") and df.empty):
        df = runtime.engineer.load(ticker)
    if df is None or (hasattr(df, "empty") and df.empty):
        raise HTTPException(404, f"No data for {ticker} - run analysis first")
    df = df.tail(days).copy()

    def safe_float(value):
        """Return a finite rounded chart value or None."""
        number = finite_number(value)
        return round(number, 6) if number is not None else None

    rows = []
    for date, row in df.iterrows():
        rows.append(
            {
                "date": str(date.date()) if hasattr(date, "date") else str(date),
                "open": safe_float(row.get("open")),
                "high": safe_float(row.get("high")),
                "low": safe_float(row.get("low")),
                "close": safe_float(row.get("close")),
                "volume": int(finite_number(row.get("volume"), 0)),
                "sma20": safe_float(row.get("sma20")),
                "sma50": safe_float(row.get("sma50")),
                "bb_upper": safe_float(row.get("bb_upper")),
                "bb_lower": safe_float(row.get("bb_lower")),
                "rsi": safe_float(row.get("rsi")),
                "macd": safe_float(row.get("macd")),
                "macd_signal": safe_float(row.get("macd_signal")),
                "macd_hist": safe_float(row.get("macd_hist")),
                "volume_ratio": safe_float(row.get("volume_ratio")),
                "adx": safe_float(row.get("adx")),
                "week52_position": safe_float(row.get("week52_position")),
                "sent_score": safe_float(row.get("sent_score")),
            }
        )
    return {"ticker": ticker, "days": len(rows), "history": rows}


@router.get("/api/analysis/{ticker}")
def full_analysis(ticker: str, risk_profile: str = "moderate", horizon: int = settings.PRIMARY_HORIZON):
    """Combine ticker evidence, model estimates and a risk-profile signal."""
    if horizon not in settings.HORIZONS:
        raise HTTPException(422, f"Unsupported horizon {horizon}; choose one of {settings.HORIZONS}")
    xgb_m, lstm_m = runtime.horizon_models(horizon)
    df = runtime.fusion.load_master(ticker)
    if df is None or (hasattr(df, "empty") and df.empty):
        df = runtime.engineer.load(ticker)
    if df is None or (hasattr(df, "empty") and df.empty):
        raise HTTPException(404, f"No data for {ticker}")

    latest = df.iloc[-1]
    # keep the two model outputs separate and named (never averaged). Inference runs
    # under model_lock so it cannot read a model mid-retrain (training holds the same lock).
    with runtime.model_lock:
        prob_up = xgb_m.predict_proba_up(df) if xgb_m.is_trained() else None  # P(rise) over the horizon
        exp_ret = lstm_m.predict_ticker(df) if lstm_m.is_trained() else None  # est. horizon return
        uncertainty = xgb_m.predict_with_uncertainty(df) if xgb_m.is_trained() else {}

    vol = float(latest.get("volatility", 0.02) or 0.02)
    # signal SCORE (0-100, not a return): from the LSTM return per unit risk if
    # available, else the XGBoost probability edge.
    if exp_ret is not None:
        score = min(100, max(0, 50 + (exp_ret / (vol * (max(horizon, 1) ** 0.5) + 1e-6)) * 1000))  # vol scaled to horizon
    elif prob_up is not None:
        score = min(100, max(0, prob_up * 100))
    else:
        score = 50.0
    mom = float(latest.get("momentum_10d", 0) or 0)
    adx = float(latest.get("adx", 20) or 20)
    # NOTE: this is a per-stock TECHNICAL trend label (ADX+momentum), distinct
    # from the market-level HMM regime shown elsewhere.
    trend = "trending_up" if adx > 25 and mom > 0 else ("trending_down" if adx > 25 and mom < 0 else "ranging")
    confidence = runtime.portfolio_manager.confidence_score(ticker, df, exp_ret if exp_ret is not None else 0.0, prob_up=prob_up)
    # Standardised names: support_score is the numeric 0-100 heuristic; support_factors is its breakdown.
    support_score = confidence.get("overall") if isinstance(confidence, dict) else confidence
    support_factors = confidence.get("factors", {}) if isinstance(confidence, dict) else {}

    return {
        "ticker": ticker,
        # SAME rule AND SAME risk profile as the recommendation queue
        # (recommender.classify_signal), so a stock never shows one label here and a
        # different one in the queue. Passing the profile is what makes that true: a
        # conservative user needs a stronger edge for a BUY on BOTH screens. Without
        # it this screen silently used 'moderate' for everyone. The score below is a
        # display measure, not the label source.
        "signal": classify_signal(prob_up, exp_ret, risk_profile, horizon=horizon),
        "risk_profile": risk_profile,
        # Surface data freshness so the UI can flag a signal computed on stale data
        # (the recommendation queue already refuses stale/unpriced tickers upstream).
        "data_fresh": bool(freshness(df)["fresh"]),
        "as_of": freshness(df)["as_of"],
        "score": round(score, 1),
        "trend": trend,
        "regime": trend,  # backward-compat alias; this is a stock trend, not the market HMM regime
        "close": round(float(latest.get("close", 0) or 0), 2),
        "support_score": support_score,
        "support_factors": support_factors,
        "confidence": support_score,  # backward-compat numeric alias of support_score
        "prediction": {
            "horizon": horizon,
            "probability_up": round(prob_up, 4) if prob_up is not None else None,
            "prob_up_pct": round(prob_up * 100, 1) if prob_up is not None else None,
            "expected_return_decimal": round(exp_ret, 6) if exp_ret is not None else None,
            "expected_return_pct": round(exp_ret * 100, 3) if exp_ret is not None else None,
            "calibrated_confidence": uncertainty.get("confidence"),
            "interval_95": uncertainty.get("interval_95"),
            "calibrated": uncertainty.get("calibrated", False),
        },
        "indicators": {
            "rsi": round(float(latest.get("rsi", 50) or 50), 2),
            "adx": round(adx, 2),
            "macd_hist": round(float(latest.get("macd_hist", 0) or 0), 4),
            "bb_pct": round(float(latest.get("bb_pct", 0.5) or 0.5), 4),
            "atr_pct": round(float(latest.get("atr_pct", 0) or 0), 4),
            "stoch_k": round(float(latest.get("stoch_k", 50) or 50), 2),
            "week52_pos": round(float(latest.get("week52_position", 0.5) or 0.5), 4),
            "volume_ratio": round(float(latest.get("volume_ratio", 1) or 1), 3),
            "momentum_10d": round(mom * 100, 2),
            "volatility": round(vol, 4),
        },
        "sentiment": {
            "score": round(float(latest.get("sent_score", 0) or 0), 4),
            "label": str(latest.get("sent_label", "neutral") or "neutral"),
            "count": int(latest.get("sent_news_count", 0) or 0),
        },
        "top_features": (
            dict(list(xgb_m.feature_importance().items())[:8]) if xgb_m.is_trained() else {}
        ),
    }
