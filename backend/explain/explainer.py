"""
explainer.py

Generates natural language explanations for each recommendation.
Tries three sources in order: the local Ollama LLM (best quality),
a SHAP-driven template that references the most important model features,
and a fallback template built from the signal data.

Ollama output is not trusted blindly: before it is used, it is checked so
that every number it cites was one we actually put in the prompt. If it
introduces any other figure (an invented price, probability or return) the
text is discarded and the deterministic template is used instead. This gates
the LIVE explanation path, not just the offline verifier. The check covers
numeric VALUES only - it does not verify that a number was attached to the
correct field or unit.

The SHAP explanations reference Lundberg and Lee (2017) for feature
attribution methodology.
"""

import logging
import pickle
import re
from pathlib import Path

import requests

from backend.config import settings
from backend.explain import explainer_modes as em

logger = logging.getLogger(__name__)
MODELS_DIR = Path("models")

# tries Ollama first (local LLM) - SHAP template - plain template
# each level gracefully catches failures so the API never stalls

# All Ollama endpoints/model come from the central settings (single source of
# truth) - so the chat, the explanation generator and the health check can never
# disagree about which server/model is in use.
OLLAMA_URL = f"{settings.OLLAMA_URL}/api/generate"
OLLAMA_TAGS = f"{settings.OLLAMA_URL}/api/tags"
OLLAMA_MODEL = settings.OLLAMA_MODEL
OLLAMA_TIMEOUT = 8


class Explainer:
    # Generate a natural language explanation for one recommendation
    """Explain recommendation evidence using grounded text or a local template."""

    def explain(self, rec: dict, feature_row=None) -> str:
        """Return a grounded Ollama explanation, falling back to the data template."""
        template = self._template(rec, feature_row)
        ollama_text = self._ask_ollama(rec)
        return ollama_text if ollama_text else template

    # Run explain() over every recommendation in the list
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

    # Call the local Ollama LLM and return the response text
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

        # Describe the models accurately and by horizon; never invent a return when
        # the estimate is missing, and never call a 21-session view "tomorrow".
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
                    ok, invented = self._numbers_ok(text, prompt)
                    faithful, fissues = self._faithful_ok(text, rec)
                    if ok and faithful:
                        return text
                    logger.warning(
                        f"Ollama explanation for {ticker} rejected "
                        f"(unsupported_numbers={invented}, faithfulness={fissues}); "
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
    def _numbers_ok(text, prompt, tol=0.011):
        """True iff every quantitative claim in `text` matches a number that was
        supplied to the model in `prompt`. Returns (ok, [unsupported_numbers]).

        The model may rephrase freely but cannot introduce a figure that was not
        among the inputs. Provenance (ISO dates) and version tokens are stripped
        before the comparison. This checks numeric VALUES only; it does not check
        that a number was attached to the right field, unit or horizon.
        """
        allowed = [float(x) for x in re.findall(r"-?\d+\.?\d*", prompt)]
        bad = []
        for n in em._claim_numbers(text):
            if not any(abs(n - a) <= tol + abs(a) * 0.02 for a in allowed):
                bad.append(n)
        return (len(bad) == 0, bad)

    @staticmethod
    def _faithful_ok(text, rec):
        """Field/claim checks the numeric-provenance guard cannot do. Returns
        (ok, [issues]). Rejects:
          * guarantee / risk-free language (carries no number, so _numbers_ok
            passes it);
          * a $-amount that is not the current price (so a confidence value shown
            as "$61" is caught even though 61 is a legitimate input number);
          * a stated "N (trading) day(s)" horizon that is not the model's, and
            "tomorrow"/"next-day" wording when the horizon is longer than a day.
        Percent figures are NOT unit-checked here because the prompt legitimately
        contains momentum% and RSI alongside the forecast.
        """
        issues = []
        low = text.lower()
        for pat in em.BANNED_PATTERNS:
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
        # SHAP TreeExplainer on the trained XGBoost - shows which features
        # pushed this specific prediction up or down (local feature attribution)
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
            # Use the EXACT training feature set and ORDER (feat_cols.pkl), filling
            # any absent column with 0. Selecting all numeric row columns (which
            # include prices and target columns dropped during training) fed the
            # scaler/model the wrong shape and order - so the attribution was
            # meaningless and usually raised, then got silently swallowed.
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

    # Build a fallback explanation from the raw signal numbers
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

        # Forecast wording from explicit model + horizon + availability fields:
        # the two models are reported separately, the horizon is ~21 sessions, and
        # a missing estimate stays 'unavailable' rather than being shown as 0%.
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
