"""Classify forecasts using profile-specific thresholds and rank eligible buy candidates."""

import logging

from backend.config.settings import PREDICTION_HORIZON
from backend.data.contracts import finite_number

logger = logging.getLogger(__name__)

# Apply shared profile thresholds to forecasts; composite ranks only order candidates.
SIGNAL_THRESHOLDS = {
    "conservative": {"buy_prob": 0.60, "sell_prob": 0.40, "buy_ret": 0.010, "sell_ret": -0.010},
    "moderate": {"buy_prob": 0.55, "sell_prob": 0.45, "buy_ret": 0.003, "sell_ret": -0.003},
    "aggressive": {"buy_prob": 0.52, "sell_prob": 0.48, "buy_ret": 0.002, "sell_ret": -0.002},
}


def classify_signal(prob_up, exp_ret, risk_profile: str = "moderate", horizon: int | None = None) -> str:
    """Classify absolute forecasts using profile thresholds; missing forecasts produce HOLD."""
    prob_up, exp_ret = finite_number(prob_up), finite_number(exp_ret)
    t = SIGNAL_THRESHOLDS.get(str(risk_profile).lower(), SIGNAL_THRESHOLDS["moderate"])
    if prob_up is None and exp_ret is None:
        return "HOLD"
    # Scale return thresholds by horizon while retaining probability thresholds.
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

        # Use moderate for an unrecognised profile.
        if profile not in SIGNAL_THRESHOLDS:
            logger.warning(f"Unknown risk profile '{risk_profile}', using moderate")
            profile = "moderate"

        signals = []
        for _, row in ranked.iterrows():
            score = row["composite_score"]
            prob_up = finite_number(row.get("prob_up"))
            exp_ret = finite_number(row.get("expected_return"))

            # Assign action labels from forecasts before ordering by composite rank.
            signal = classify_signal(prob_up, exp_ret, profile)
            signals.append(
                {
                    "ticker": row["ticker"],
                    "signal": signal,
                    "composite_score": round(float(score), 2),  # ranking score, not a return
                    "rank": int(row["rank"]),
                    # XGBoost and LSTM outputs kept as separate fields.
                    "prob_up_pct": round(float(prob_up) * 100, 1) if prob_up is not None else None,
                    "expected_return_pct": round(float(exp_ret) * 100, 2) if exp_ret is not None else None,
                    "horizon": row.get("horizon"),
                    # alias of expected_return_pct
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

        # Select the highest-ranked BUY signals as top picks.
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
