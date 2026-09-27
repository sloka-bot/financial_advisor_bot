"""
recommender.py

Converts ranked stock scores into BUY / HOLD / SELL signals.
The signal thresholds are tiered by risk profile so that a conservative user
only gets a BUY when the composite score is very high.
"""

import logging

from backend.config.settings import PREDICTION_HORIZON
from backend.data.contracts import finite_number

logger = logging.getLogger(__name__)

# SINGLE SOURCE OF TRUTH for a per-stock BUY/HOLD/SELL label, from the ABSOLUTE
# model forecast (never the within-universe percentile). Shared by the
# recommendation queue AND the single-stock analysis screen so the same stock can
# never show two different labels. The composite percentile is used only to ORDER
# and SELECT names, never to flip this label. Profile-aware thresholds keep a
# conservative user needing a stronger edge - in ONE place, not per screen.
# Return dead-band per profile: the LSTM edge over a historical-mean baseline is
# small, so a forecast must clear a minimum magnitude before it counts as bullish
# or bearish. A near-zero prediction is treated as HOLD, not BUY/SELL. These are
# conservative floors and should be revisited with a sensitivity study on
# development data rather than presented as optimised values.
SIGNAL_THRESHOLDS = {
    "conservative": {"buy_prob": 0.60, "sell_prob": 0.40, "buy_ret": 0.010, "sell_ret": -0.010},
    "moderate": {"buy_prob": 0.55, "sell_prob": 0.45, "buy_ret": 0.003, "sell_ret": -0.003},
    "aggressive": {"buy_prob": 0.52, "sell_prob": 0.48, "buy_ret": 0.002, "sell_ret": -0.002},
}


def classify_signal(prob_up, exp_ret, risk_profile: str = "moderate", horizon: int | None = None) -> str:
    """Return 'BUY' | 'HOLD' | 'SELL' from the absolute forecast for one stock.

    A high percentile rank alone must never force a BUY when the stock's own
    forecast is not bullish, and an outright-negative forecast forces SELL - so the
    whole universe can legitimately end up HOLD/SELL (an all-cash outcome). When no
    model forecast is available at all, returns 'HOLD' (never a fabricated BUY).
    """
    prob_up, exp_ret = finite_number(prob_up), finite_number(exp_ret)
    t = SIGNAL_THRESHOLDS.get(str(risk_profile).lower(), SIGNAL_THRESHOLDS["moderate"])
    if prob_up is None and exp_ret is None:
        return "HOLD"
    # Scale the RETURN dead-band to the forecast horizon. The thresholds are set for
    # the 21-day horizon, so a 1- or 5-day expected return (proportionally smaller)
    # would almost never clear an unscaled 21-day band. Scale linearly by
    # horizon/PRIMARY; probability thresholds are already horizon-independent. A
    # development-data-tuned band per horizon would be a stronger choice (future work).
    h = int(horizon) if horizon else PREDICTION_HORIZON
    scale = max(h, 1) / max(PREDICTION_HORIZON, 1)
    buy_ret, sell_ret = t["buy_ret"] * scale, t["sell_ret"] * scale
    bull = (exp_ret is not None and exp_ret > buy_ret) or (prob_up is not None and prob_up >= t["buy_prob"])
    bear = (exp_ret is not None and exp_ret < sell_ret) or (prob_up is not None and prob_up <= t["sell_prob"])
    if bull and not bear:
        return "BUY"
    if bear and not bull:
        return "SELL"
    return "HOLD"


class RecommendationEngine:
    """Classify ranked stocks using the configured risk-profile thresholds."""

    def recommend(self, ranked, risk_profile="moderate", top_n=5):
        """Apply the risk-profile thresholds and return a summary of all signals."""
        profile = risk_profile.lower()

        # Fall back to moderate if an unrecognised profile is passed in
        if profile not in SIGNAL_THRESHOLDS:
            logger.warning(f"Unknown risk profile '{risk_profile}', using moderate")
            profile = "moderate"

        signals = []
        for _, row in ranked.iterrows():
            score = row["composite_score"]
            prob_up = finite_number(row.get("prob_up"))
            exp_ret = finite_number(row.get("expected_return"))

            # ONE shared rule (identical to the single-stock analysis screen): the
            # label comes from the absolute forecast. The percentile `score` only
            # ORDERS and selects top picks below - it never flips this label.
            signal = classify_signal(prob_up, exp_ret, profile)
            signals.append(
                {
                    "ticker": row["ticker"],
                    "signal": signal,
                    "composite_score": round(float(score), 2),  # ranking score, NOT a return
                    "rank": int(row["rank"]),
                    # XGBoost and LSTM outputs kept separate and named:
                    "prob_up_pct": round(float(prob_up) * 100, 1) if prob_up is not None else None,
                    "expected_return_pct": round(float(exp_ret) * 100, 2) if exp_ret is not None else None,
                    "horizon": row.get("horizon"),
                    # alias of expected_return_pct for existing consumers:
                    "predicted_return": round(exp_ret * 100, 2) if exp_ret is not None else None,
                    "as_of": row.get("as_of"),
                    "sentiment_label": row.get("sent_label", "neutral"),
                    "sentiment_score": round(float(row.get("sentiment_raw", 0.0)), 3),
                    "momentum": round(float(row.get("momentum_raw", 0.0)) * 100, 2),
                    "rsi": round(float(row.get("rsi", 50.0)), 1),
                    "current_price": round(float(row.get("close", 0.0)), 2),
                    "risk_profile": profile,
                }
            )

        # Top BUY picks only, ranked by composite score (HOLD/SELL are excluded so
        # "top picks" cannot contain a stock the system is not recommending to buy).
        top_picks = sorted(
            [s for s in signals if s["signal"] == "BUY"], key=lambda x: x["composite_score"], reverse=True
        )[:top_n]
        buys = [s for s in signals if s["signal"] == "BUY"]
        sells = [s for s in signals if s["signal"] == "SELL"]

        return {
            "all_signals": signals,
            "top_picks": top_picks,
            "summary": {
                "total": len(signals),
                "buys": len(buys),
                "sells": len(sells),
                "holds": len(signals) - len(buys) - len(sells),
                "profile": profile,
            },
        }
