import logging

logger = logging.getLogger(__name__)

THRESHOLDS = {
    "conservative": {"buy": 72, "sell": 45},
    "moderate":     {"buy": 65, "sell": 40},
    "aggressive":   {"buy": 55, "sell": 30},
}


class RecommendationEngine:

    def recommend(self, ranked, risk_profile="moderate", top_n=5):
        profile = risk_profile.lower()
        if profile not in THRESHOLDS:
            logger.warning(f"Unknown risk profile '{risk_profile}', using moderate")
            profile = "moderate"

        buy_th  = THRESHOLDS[profile]["buy"]
        sell_th = THRESHOLDS[profile]["sell"]

        signals = []
        for _, row in ranked.iterrows():
            score = row["composite_score"]
            if score >= buy_th:
                signal = "BUY"
            elif score < sell_th:
                signal = "SELL"
            else:
                signal = "HOLD"

            signals.append({
                "ticker":           row["ticker"],
                "signal":           signal,
                "composite_score":  round(float(score), 2),
                "rank":             int(row["rank"]),
                "predicted_return": round(float(row["predicted_return"]) * 100, 2),
                "sentiment_label":  row.get("sent_label", "neutral"),
                "sentiment_score":  round(float(row.get("sentiment_raw", 0.0)), 3),
                "momentum":         round(float(row.get("momentum_raw", 0.0)) * 100, 2),
                "rsi":              round(float(row.get("rsi", 50.0)), 1),
                "current_price":    round(float(row.get("close", 0.0)), 2),
                "risk_profile":     profile,
            })

        top_picks = sorted(signals, key=lambda x: x["composite_score"], reverse=True)[:top_n]
        buys      = [s for s in signals if s["signal"] == "BUY"]
        holds     = [s for s in signals if s["signal"] == "HOLD"]
        sells     = [s for s in signals if s["signal"] == "SELL"]

        logger.info(f"Signals ({profile}): BUY={len(buys)} HOLD={len(holds)} SELL={len(sells)}")

        return {
            "top_picks":   top_picks,
            "all_signals": signals,
            "summary": {
                "buy_count":    len(buys),
                "hold_count":   len(holds),
                "sell_count":   len(sells),
                "risk_profile": profile,
                "thresholds":   THRESHOLDS[profile],
            },
        }
