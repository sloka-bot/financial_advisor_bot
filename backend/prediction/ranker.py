"""
ranker.py

Combines the ML model signal, FinBERT sentiment, and technical indicators
into a single composite score for each stock. The composite score only RANKS and
orders names; the BUY / HOLD / SELL label is set from the absolute model outputs
(XGBoost probability and LSTM expected return) in the recommendation engine, not
from this score.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

PROFILE_WEIGHTS = {
    "conservative": [0.25, 0.35, 0.15, 0.25],
    "moderate": [0.40, 0.30, 0.20, 0.10],
    "aggressive": [0.55, 0.25, 0.20, 0.00],
}

MIN_NEWS_FOR_RELIABLE_SENTIMENT = 3


class StockRanker:
    # Score every ticker by combining the ML signal, sentiment, and momentum into a composite
    """Rank prediction evidence with risk-profile-specific component weights."""

    def rank(self, predictions: dict, master_data: dict, risk_profile: str) -> pd.DataFrame:
        """Combine model, sentiment and technical evidence into a ranked table."""
        weights = PROFILE_WEIGHTS.get(risk_profile, PROFILE_WEIGHTS["moderate"])
        rows = []

        for ticker, pr in predictions.items():
            df = master_data.get(ticker)
            if df is None or df.empty:
                continue

            latest = df.iloc[-1]

            vol = float(latest.get("volatility", 0.02) or 0.02)
            rsi = float(latest.get("rsi", 50.0) or 50.0)
            momentum = float(latest.get("momentum_10d", 0.0) or 0.0)
            sent_raw = float(latest.get("sent_score", 0.0) or 0.0)
            n_news = int(latest.get("sent_news_count", 0) or 0)
            sent_label = str(latest.get("sent_label", "neutral") or "neutral")
            close = float(latest.get("close", 0.0) or 0.0)

            # prediction is a named dict {prob_up, expected_return, horizon, ...};
            # tolerate a bare number for backward compatibility.
            if isinstance(pr, dict):
                exp_ret = pr.get("expected_return")
                prob_up = pr.get("prob_up")
                horizon = pr.get("horizon")
            else:
                exp_ret, prob_up, horizon = (float(pr) if pr is not None else None), None, None

            # factor 1: risk-adjusted SIGNAL (a score input, not a return). Prefer
            # the LSTM's estimated return per unit risk; if only the XGBoost
            # probability exists, use its directional edge (prob-0.5) per unit risk.
            if exp_ret is not None:
                risk_adj = exp_ret / (vol + 1e-9)
            elif prob_up is not None:
                risk_adj = (prob_up - 0.5) / (vol + 1e-9)
            else:
                risk_adj = 0.0

            # factor 2: sentiment quality - discount light coverage
            coverage_weight = min(1.0, n_news / MIN_NEWS_FOR_RELIABLE_SENTIMENT)
            sent_quality = sent_raw * coverage_weight

            # factor 3: momentum filtered by RSI headroom
            rsi_room = (100 - rsi) / 100
            mom_quality = momentum * rsi_room

            # factor 4: stability - negative vol rewards steady stocks
            stability = -vol

            rows.append(
                {
                    "ticker": ticker,
                    # both model outputs kept SEPARATE and named (never merged):
                    "prob_up": round(float(prob_up), 4) if prob_up is not None else None,
                    "expected_return": round(float(exp_ret), 6) if exp_ret is not None else None,
                    "horizon": horizon,
                    # `predicted_return` retained as an alias of the LSTM expected
                    # return (decimal) so existing consumers keep working; it is a
                    # return estimate, not the ranking score.
                    "predicted_return": float(exp_ret) if exp_ret is not None else None,
                    "as_of": pr.get("as_of") if isinstance(pr, dict) else None,
                    "volatility": round(vol, 6),
                    "sentiment": sent_label,
                    "sent_label": sent_label,
                    "sentiment_raw": round(sent_raw, 4),
                    "momentum_raw": round(momentum, 6),
                    "rsi": round(rsi, 2),
                    "close": round(close, 4),
                    "_f1": risk_adj,
                    "_f2": sent_quality,
                    "_f3": mom_quality,
                    "_f4": stability,
                }
            )

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # z-score normalise each factor and clip at ±3σ. A single row gives a NaN
        # std (which is truthy, so `std or 1e-8` would not substitute the floor), so
        # guard explicitly against a non-finite or near-zero std before dividing.
        for col in ["_f1", "_f2", "_f3", "_f4"]:
            mu = df[col].mean()
            std = df[col].std()
            if not np.isfinite(std) or std < 1e-8:
                std = 1e-8
            df[col] = ((df[col] - mu) / std).clip(-3, 3)

        df["raw_score"] = (
            df["_f1"] * weights[0] + df["_f2"] * weights[1] + df["_f3"] * weights[2] + df["_f4"] * weights[3]
        )

        n = len(df)
        df["composite_score"] = df["raw_score"].rank(method="average") / n * 100

        df = df.drop(columns=["_f1", "_f2", "_f3", "_f4", "raw_score"])
        df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)

        # add integer rank (1 = best) so recommender can reference it
        df["rank"] = df.index + 1

        logger.info(f"Ranked {len(df)} tickers - profile: {risk_profile}")
        return df
