"""Generate checked Ollama explanations with SHAP and structured-text fallbacks."""

import logging
import pickle
import re
from pathlib import Path

import requests

from backend.config import settings
from backend.explain import explainer_modes as em

logger = logging.getLogger(__name__)
MODELS_DIR = Path("models")

# Try Ollama, then SHAP-based text, then the signal template.

# Read Ollama configuration from shared service settings.
OLLAMA_URL = f"{settings.OLLAMA_URL}/api/generate"
OLLAMA_TAGS = f"{settings.OLLAMA_URL}/api/tags"
OLLAMA_MODEL = settings.OLLAMA_MODEL
OLLAMA_TIMEOUT = 30


class Explainer:
    """Explain recommendation evidence using checked text or a local template."""

    def explain(self, rec: dict, feature_row=None) -> str:
        """Return a checked Ollama explanation, falling back to the data template."""
        template = self._template(rec, feature_row)
        ollama_text = self._ask_ollama(rec)
        return ollama_text if ollama_text else template

    def explain_batch(self, recommendations: list, master_data: dict = None) -> list:
        """Attach explanation text to each recommendation in place."""
        for rec in recommendations:
            ticker = rec["ticker"]
            frow = master_data.get(ticker) if master_data else None
            if frow is not None and not frow.empty:
                frow = frow.iloc[[-1]]
            rec["explanation"] = self.explain(rec, frow)
        return recommendations

    def ollama_available(self) -> bool:
        """Check whether the configured local Ollama service responds."""
        try:
            return requests.get(OLLAMA_TAGS, timeout=2).status_code == 200
        except Exception:
            return False

    # Ask the local Ollama model and return its reply.
    def _ask_ollama(self, rec: dict) -> str | None:
        ticker = rec.get("ticker", "")
        signal = rec.get("signal", "HOLD")
        score = rec.get("composite_score", 50)
        sent = rec.get("sentiment_label", "neutral")
        mom = rec.get("momentum", 0)
        rsi = rec.get("rsi", 50)
        exp_pct = rec.get("expected_return_pct")
        prob_pct = rec.get("prob_up_pct")
        horizon = int(rec.get("horizon") or settings.PRIMARY_HORIZON)

        # Describe each available forecast with its model and trading horizon.
        if exp_pct is not None:
            forecast = f"LSTM expected return over ~{horizon} trading days: {float(exp_pct):+.2f}%"
        elif prob_pct is not None:
            forecast = f"XGBoost probability of a rise over ~{horizon} trading days: {float(prob_pct):.0f}%"
        else:
            forecast = "No model return estimate available"

        prompt = (
            "You are a professional financial advisor. Explain this recommendation in 2-3 sentences.\n"
            'Be specific about the numbers. Do not start with "I". No bullet points.\n\n'
            f"{ticker}: {signal} (score {score:.0f}/100)\n"
            f"{forecast}\n"
            f"10-day momentum: {mom:+.2f}%  RSI: {rsi:.1f}  Sentiment: {sent}\n\n"
            f"Explain why {ticker} is rated {signal}:"
        )

        try:
            r = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 150},
                },
                timeout=OLLAMA_TIMEOUT,
            )
            if r.status_code == 200:
                text = r.json().get("response", "").strip()
                if len(text) > 20:
                    ok, unsupported_values = self._validate_numeric_claims(text, prompt)
                    faithful, fissues = self._validate_claim_context(text, rec)
                    if ok and faithful:
                        return text
                    logger.warning(
                        f"Ollama explanation for {ticker} rejected "
                        f"(unsupported_numbers={unsupported_values}, faithfulness={fissues}); "
                        "using template instead"
                    )
        except requests.exceptions.ConnectionError:
            logger.debug("Ollama not running")
        except requests.exceptions.Timeout:
            logger.debug("Ollama timed out")
        except Exception as e:
            logger.debug(f"Ollama error: {e}")
        return None

    @staticmethod
    def _validate_numeric_claims(text, prompt, tol=0.011):
        """Return numeric support status and unsupported values, without checking field or unit."""
        allowed = [float(x) for x in re.findall(r"-?\d+\.?\d*", prompt)]
        bad = []
        for n in em._extract_claim_numbers(text):
            if not any(abs(n - a) <= tol + abs(a) * 0.02 for a in allowed):
                bad.append(n)
        return (len(bad) == 0, bad)

    @staticmethod
    def _validate_claim_context(text, rec):
        """Check guarantee wording, dollar values and horizon claims; percentages remain unchecked."""
        issues = []
        low = text.lower()
        for pat in em.UNSUPPORTED_CLAIM_PATTERNS:
            if re.search(pat, low):
                issues.append(f"banned phrase /{pat}/")
        price = rec.get("current_price")
        if price is not None:
            for v in re.findall(r"\$\s*(-?\d+\.?\d*)", text):
                if abs(float(v) - float(price)) > 0.011 + abs(float(price)) * 0.02:
                    issues.append(f"$ amount not the current price: {v}")
        h = int(rec.get("horizon") or settings.PRIMARY_HORIZON)
        for m in re.findall(r"(\d+)\s*(?:trading\s*)?days?\b", low):
            if int(m) != h:
                issues.append(f"wrong horizon {m}d vs {h}d")
        if h > 1 and re.search(r"\btomorrow\b|\bnext[\s-]?day\b", low):
            issues.append("implies a 1-day horizon")
        return (len(issues) == 0, issues)

    def _shap_drivers(self, feature_row) -> dict | None:
        # Attribute the current XGBoost prediction to its input features.
        try:
            import numpy as np
            import pandas as pd
            import shap

            mp = MODELS_DIR / "xgboost" / "model.pkl"
            sp = MODELS_DIR / "xgboost" / "scaler.pkl"
            fp = MODELS_DIR / "xgboost" / "feat_cols.pkl"
            if not (mp.exists() and sp.exists() and fp.exists()):
                logger.info("SHAP drivers skipped: model/scaler/feat_cols not all present")
                return None

            with open(mp, "rb") as f:
                model = pickle.load(f)
            with open(sp, "rb") as f:
                scaler = pickle.load(f)
            with open(fp, "rb") as f:
                feat_cols = pickle.load(f)

            row = feature_row if isinstance(feature_row, pd.DataFrame) else pd.DataFrame([feature_row])
            # Match the saved feature order before applying the scaler and SHAP explainer.
            X_df = row.reindex(columns=feat_cols, fill_value=0.0)
            X_df = X_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
            X = scaler.transform(X_df.values)
            sv = shap.TreeExplainer(model).shap_values(X)
            sv = np.asarray(sv[0] if isinstance(sv, list) else sv).reshape(-1)[: len(feat_cols)]
            top3 = sorted(zip(feat_cols, sv), key=lambda x: abs(x[1]), reverse=True)[:3]
            return {n: round(float(v), 4) for n, v in top3}
        except Exception as e:
            logger.warning(f"SHAP driver computation failed (explanation continues without it): {e}")
            return None

    # Build a template explanation from the signal values.
    def _template(self, rec: dict, feature_row=None) -> str:
        ticker = rec.get("ticker", "")
        signal = rec.get("signal", "HOLD")
        score = float(rec.get("composite_score", 50))
        sent = rec.get("sentiment_label", "neutral")
        sent_sc = float(rec.get("sentiment_score", 0))
        mom = float(rec.get("momentum", 0))
        rsi = float(rec.get("rsi", 50))
        price = float(rec.get("current_price", 0))
        vol = float(rec.get("volatility", 0))

        parts = []

        if signal == "BUY":
            parts.append(f"{ticker} scores {score:.0f}/100 - top {100 - score:.0f}% of the universe. Rated BUY.")
        elif signal == "SELL":
            parts.append(f"{ticker} scores {score:.0f}/100 - bottom {score:.0f}% of the universe. Rated SELL.")
        else:
            parts.append(f"{ticker} scores {score:.0f}/100. HOLD until a clearer signal.")

        # Report each model's horizon and retain unavailable estimates as missing.
        exp_pct = rec.get("expected_return_pct")  # LSTM return estimate (%), may be None
        prob_pct = rec.get("prob_up_pct")  # XGBoost P(rise) (%), may be None
        horizon = int(rec.get("horizon") or settings.PRIMARY_HORIZON)
        hz = f"~{horizon} trading days"
        if exp_pct is not None:
            direction = "gain" if exp_pct > 0 else ("decline" if exp_pct < 0 else "flat move")
            parts.append(f"LSTM estimates a {abs(float(exp_pct)):.2f}% {direction} over {hz}.")
        elif prob_pct is not None:
            parts.append(f"XGBoost puts the probability of a rise over {hz} at {float(prob_pct):.0f}%.")
        else:
            parts.append("No model return estimate is available for this stock yet.")

        if vol > 0:
            parts.append(f"Annualised volatility: {vol * (252**0.5) * 100:.1f}%.")

        shap = self._shap_drivers(feature_row) if feature_row is not None else None
        if shap:
            drivers = ", ".join(f"{n} ({'+' if v > 0 else ''}{v})" for n, v in shap.items())
            parts.append(f"Top model drivers: {drivers}.")

        if sent == "positive":
            parts.append(f"FinBERT sentiment positive ({sent_sc:+.3f}) - news flow supportive.")
        elif sent == "negative":
            parts.append(f"FinBERT sentiment negative ({sent_sc:+.3f}) - weighs on outlook.")
        else:
            parts.append("Sentiment neutral.")

        if abs(mom) < 0.5:
            parts.append("10-day momentum flat.")
        elif mom > 0:
            rsi_note = "RSI overbought - may be fading" if rsi >= 70 else "RSI has room to run"
            parts.append(f"Momentum +{mom:.2f}%; {rsi_note}.")
        else:
            rsi_note = "RSI oversold - watch for reversal" if rsi <= 30 else "RSI confirms weakness"
            parts.append(f"Momentum {mom:.2f}%; {rsi_note}.")

        if price > 0:
            parts.append(f"Price: ${price:.2f}.")

        return " ".join(parts)
