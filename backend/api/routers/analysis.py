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

    def to_chart_value(value):
        """Return a finite rounded chart value or None."""
        number = finite_number(value)
        return round(number, 6) if number is not None else None

    rows = []
    for date, row in df.iterrows():
        rows.append(
            {
                "date": str(date.date()) if hasattr(date, "date") else str(date),
                "open": to_chart_value(row.get("open")),
                "high": to_chart_value(row.get("high")),
                "low": to_chart_value(row.get("low")),
                "close": to_chart_value(row.get("close")),
                "volume": int(finite_number(row.get("volume"), 0)),
                "sma20": to_chart_value(row.get("sma20")),
                "sma50": to_chart_value(row.get("sma50")),
                "bb_upper": to_chart_value(row.get("bb_upper")),
                "bb_lower": to_chart_value(row.get("bb_lower")),
                "rsi": to_chart_value(row.get("rsi")),
                "macd": to_chart_value(row.get("macd")),
                "macd_signal": to_chart_value(row.get("macd_signal")),
                "macd_hist": to_chart_value(row.get("macd_hist")),
                "volume_ratio": to_chart_value(row.get("volume_ratio")),
                "adx": to_chart_value(row.get("adx")),
                "week52_position": to_chart_value(row.get("week52_position")),
                "sent_score": to_chart_value(row.get("sent_score")),
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
    # Lock inference against training and keep classifier and regressor outputs separate.
    with runtime.model_lock:
        prob_up = xgb_m.predict_proba_up(df) if xgb_m.is_trained() else None  # probability of a rise over the horizon
        exp_ret = lstm_m.predict_ticker(df) if lstm_m.is_trained() else None  # estimated horizon return
        uncertainty = xgb_m.predict_with_uncertainty(df) if xgb_m.is_trained() else {}

    vol = finite_number(latest.get("volatility"), 0.02)
    # Scale the return or probability edge by risk to form a 0-100 signal score.
    if exp_ret is not None:
        score = min(
            100, max(0, 50 + (exp_ret / (vol * (max(horizon, 1) ** 0.5) + 1e-6)) * 1000)
        )  # volatility scaled to the horizon
    elif prob_up is not None:
        score = min(100, max(0, prob_up * 100))
    else:
        score = 50.0
    mom = finite_number(latest.get("momentum_10d"), 0)
    adx = finite_number(latest.get("adx"), 20)
    # Compute the stock's technical trend separately from the market HMM regime.
    trend = "trending_up" if adx > 25 and mom > 0 else ("trending_down" if adx > 25 and mom < 0 else "ranging")
    confidence = runtime.portfolio_manager.confidence_score(
        ticker, df, exp_ret if exp_ret is not None else 0.0, prob_up=prob_up
    )
    # support_score is the 0-100 heuristic and support_factors its breakdown.
    support_score = confidence.get("overall") if isinstance(confidence, dict) else confidence
    support_factors = confidence.get("factors", {}) if isinstance(confidence, dict) else {}

    return {
        "ticker": ticker,
        # Use the same profile-dependent signal rule as the recommendation queue.
        "signal": classify_signal(prob_up, exp_ret, risk_profile, horizon=horizon),
        "risk_profile": risk_profile,
        # Expose observation freshness alongside the signal.
        "data_fresh": bool(freshness(df)["fresh"]),
        "as_of": freshness(df)["as_of"],
        "score": round(score, 1),
        "trend": trend,
        "regime": trend,  # alias of trend
        "close": round(finite_number(latest.get("close"), 0), 2),
        "support_score": support_score,
        "support_factors": support_factors,
        "confidence": support_score,  # alias of support_score
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
            "rsi": round(finite_number(latest.get("rsi"), 50), 2),
            "adx": round(adx, 2),
            "macd_hist": round(finite_number(latest.get("macd_hist"), 0), 4),
            "bb_pct": round(finite_number(latest.get("bb_pct"), 0.5), 4),
            "atr_pct": round(finite_number(latest.get("atr_pct"), 0), 4),
            "stoch_k": round(finite_number(latest.get("stoch_k"), 50), 2),
            "week52_pos": round(finite_number(latest.get("week52_position"), 0.5), 4),
            "volume_ratio": round(finite_number(latest.get("volume_ratio"), 1), 3),
            "momentum_10d": round(mom * 100, 2),
            "volatility": round(vol, 4),
        },
        "sentiment": {
            "score": round(finite_number(latest.get("sent_score"), 0), 4),
            "label": str(latest.get("sent_label", "neutral") or "neutral"),
            "count": int(finite_number(latest.get("sent_news_count"), 0)),
        },
        "top_features": (dict(list(xgb_m.feature_importance().items())[:8]) if xgb_m.is_trained() else {}),
    }
