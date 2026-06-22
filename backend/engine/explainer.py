import logging
import pickle
from pathlib import Path

import requests

logger     = logging.getLogger(__name__)
MODELS_DIR = Path('models')

# tries Ollama first (local LLM) → SHAP template → plain template
# each level gracefully catches failures so the API never stalls
OLLAMA_URL     = 'http://localhost:11434/api/generate'
OLLAMA_MODEL   = 'llama3.2'
OLLAMA_TIMEOUT = 8


class Explainer:

    def explain(self, rec: dict, feature_row=None) -> str:
        template    = self._template(rec, feature_row)
        ollama_text = self._ask_ollama(rec)
        return ollama_text if ollama_text else template

    def explain_batch(self, recommendations: list, master_data: dict = None) -> list:
        for rec in recommendations:
            ticker = rec['ticker']
            frow   = master_data.get(ticker) if master_data else None
            if frow is not None and not frow.empty:
                frow = frow.iloc[[-1]]
            rec['explanation'] = self.explain(rec, frow)
        return recommendations

    def ollama_available(self) -> bool:
        try:
            return requests.get('http://localhost:11434/api/tags', timeout=2).status_code == 200
        except Exception:
            return False

    def _ask_ollama(self, rec: dict) -> str | None:
        ticker = rec.get('ticker', '')
        signal = rec.get('signal', 'HOLD')
        score  = rec.get('composite_score', 50)
        ret    = rec.get('predicted_return', 0)
        sent   = rec.get('sentiment_label', 'neutral')
        mom    = rec.get('momentum', 0)
        rsi    = rec.get('rsi', 50)

        prompt = (
            f'You are a professional financial advisor. Explain this recommendation in 2-3 sentences.\n'
            f'Be specific about the numbers. Do not start with "I". No bullet points.\n\n'
            f'{ticker}: {signal} (score {score:.0f}/100)\n'
            f'Model predicted return: {ret:+.3f}%\n'
            f'10-day momentum: {mom:+.2f}%  RSI: {rsi:.1f}  Sentiment: {sent}\n\n'
            f'Explain why {ticker} is rated {signal}:'
        )

        try:
            r = requests.post(
                OLLAMA_URL,
                json={'model': OLLAMA_MODEL, 'prompt': prompt, 'stream': False,
                      'options': {'temperature': 0.3, 'num_predict': 150}},
                timeout=OLLAMA_TIMEOUT,
            )
            if r.status_code == 200:
                text = r.json().get('response', '').strip()
                if len(text) > 20:
                    return text
        except requests.exceptions.ConnectionError:
            logger.debug('Ollama not running')
        except requests.exceptions.Timeout:
            logger.debug('Ollama timed out')
        except Exception as e:
            logger.debug(f'Ollama error: {e}')
        return None

    def _shap_drivers(self, feature_row) -> dict | None:
        # SHAP TreeExplainer on the trained XGBoost — shows which features
        # pushed this specific prediction up or down (local feature attribution)
        try:
            import shap
            import numpy as np
            import pandas as pd

            mp = MODELS_DIR / 'xgboost' / 'model.pkl'
            sp = MODELS_DIR / 'xgboost' / 'scaler.pkl'
            if not mp.exists():
                return None

            with open(mp, 'rb') as f: model  = pickle.load(f)
            with open(sp, 'rb') as f: scaler = pickle.load(f)

            row  = feature_row if isinstance(feature_row, pd.DataFrame) else pd.DataFrame([feature_row])
            cols = row.select_dtypes(include='number').columns.tolist()
            X    = scaler.transform(row[cols])
            sv   = shap.TreeExplainer(model).shap_values(X)[0]
            top3 = sorted(zip(cols, sv), key=lambda x: abs(x[1]), reverse=True)[:3]
            return {n: round(float(v), 4) for n, v in top3}
        except Exception as e:
            logger.debug(f'SHAP unavailable: {e}')
            return None

    def _template(self, rec: dict, feature_row=None) -> str:
        ticker  = rec.get('ticker', '')
        signal  = rec.get('signal', 'HOLD')
        score   = float(rec.get('composite_score', 50))
        ret     = float(rec.get('predicted_return', 0))
        sent    = rec.get('sentiment_label', 'neutral')
        sent_sc = float(rec.get('sentiment_score', 0))
        mom     = float(rec.get('momentum', 0))
        rsi     = float(rec.get('rsi', 50))
        price   = float(rec.get('current_price', 0))
        vol     = float(rec.get('volatility', 0))

        parts = []

        if signal == 'BUY':
            parts.append(f'{ticker} scores {score:.0f}/100 — top {100-score:.0f}% of the universe. Rated BUY.')
        elif signal == 'SELL':
            parts.append(f'{ticker} scores {score:.0f}/100 — bottom {score:.0f}% of the universe. Rated SELL.')
        else:
            parts.append(f'{ticker} scores {score:.0f}/100. HOLD until a clearer signal.')

        direction = 'gain' if ret > 0 else 'decline'
        parts.append(f'Ensemble model forecasts a {abs(ret):.3f}% {direction} tomorrow.')

        if vol > 0:
            parts.append(f'Annualised volatility: {vol*(252**0.5)*100:.1f}%.')

        shap = self._shap_drivers(feature_row) if feature_row is not None else None
        if shap:
            drivers = ', '.join(f'{n} ({"+" if v>0 else ""}{v})' for n, v in shap.items())
            parts.append(f'Top model drivers: {drivers}.')

        if sent == 'positive':
            parts.append(f'FinBERT sentiment positive ({sent_sc:+.3f}) — news flow supportive.')
        elif sent == 'negative':
            parts.append(f'FinBERT sentiment negative ({sent_sc:+.3f}) — weighs on outlook.')
        else:
            parts.append('Sentiment neutral.')

        if abs(mom) < 0.5:
            parts.append('10-day momentum flat.')
        elif mom > 0:
            rsi_note = 'RSI overbought — may be fading' if rsi >= 70 else 'RSI has room to run'
            parts.append(f'Momentum +{mom:.2f}%; {rsi_note}.')
        else:
            rsi_note = 'RSI oversold — watch for reversal' if rsi <= 30 else 'RSI confirms weakness'
            parts.append(f'Momentum {mom:.2f}%; {rsi_note}.')

        if price > 0:
            parts.append(f'Price: ${price:.2f}.')

        return ' '.join(parts)
