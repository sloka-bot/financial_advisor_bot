"""Estimate regimes from a mean-price market proxy using an HMM or labelled heuristic fallback."""

import logging

import numpy as np
import pandas as pd

from backend.config.settings import TRADING_DAYS  # annualisation factor

logger = logging.getLogger(__name__)

N_STATES = 3  # bull, bear, sideways
MIN_OBS = 60  # minimum observations for an HMM fit
VOL_WINDOW = 10  # rolling volatility window

BAND_ANN = 0.05  # annual return band treated as sideways


class RegimeDetector:
    """Gaussian-HMM market-regime detector with a heuristic fallback."""

    def __init__(self, n_states: int = N_STATES):
        self.n_states = n_states

    def detect(self, master_data: dict, strict: bool = False) -> dict:
        """Detect the regime; strict mode reports HMM failures as unavailable."""
        if not master_data:
            return {"regime": "unknown", "confidence": 0, "method": "none", "metrics": {}}

        feats, mkt_index = self._market_features(master_data)
        if feats is not None and len(feats) >= MIN_OBS:
            hmm_result = self._hmm_detect(feats)
            if hmm_result is not None:
                return hmm_result
            logger.info("Regime: HMM unavailable/failed - using heuristic fallback")
        else:
            logger.info("Regime: too little market history for HMM - using heuristic fallback")

        if strict:
            return {
                "regime": "unknown",
                "confidence": 0,
                "method": "hmm_unavailable",
                "valid": False,
                "metrics": {},
                "error": "HMM required in strict mode but unavailable/failed; heuristic not used",
            }
        return self._heuristic_detect(master_data)

    # Gaussian HMM estimation.
    def _market_features(self, master_data: dict):
        """Build return and rolling-volatility features from the universe's mean close."""
        closes = {}
        for ticker, df in master_data.items():
            if df is None or getattr(df, "empty", True) or "close" not in df.columns:
                continue
            s = pd.to_numeric(df["close"], errors="coerce")
            if not isinstance(s.index, pd.DatetimeIndex):
                try:
                    s.index = pd.to_datetime(s.index)
                except Exception:
                    continue
            closes[ticker] = s
        if not closes:
            return None, None

        mkt = pd.DataFrame(closes).mean(axis=1).sort_index()  # mean-price market proxy
        ret = mkt.pct_change()
        vol = ret.rolling(VOL_WINDOW).std()
        feat_df = pd.DataFrame({"ret": ret, "vol": vol}).dropna()
        if feat_df.empty:
            return None, None
        return feat_df.values.astype(float), feat_df.index

    def _hmm_detect(self, feats: np.ndarray):
        """Fit an HMM and classify recent posterior-weighted return, or return None on failure."""
        try:
            from hmmlearn.hmm import GaussianHMM
        except Exception as exc:  # hmmlearn not installed
            logger.warning(f"hmmlearn unavailable ({exc})")
            return None

        try:
            # Standardise return and volatility features before fitting the HMM.
            mu, sigma = feats.mean(axis=0), feats.std(axis=0) + 1e-9
            feats_z = (feats - mu) / sigma

            model = GaussianHMM(n_components=self.n_states, covariance_type="diag", n_iter=200, random_state=42)
            model.fit(feats_z)
            states = model.predict(feats_z)
            posteriors = model.predict_proba(feats_z)

            # Label states by absolute annual-return bands rather than relative state rank.
            mean_ret_daily = np.array(
                [feats[states == s, 0].mean() if np.any(states == s) else 0.0 for s in range(self.n_states)]
            )
            ann_ret = mean_ret_daily * TRADING_DAYS

            def _label(a):
                return "bull" if a > BAND_ANN else ("bear" if a < -BAND_ANN else "sideways")

            label_for = {s: _label(ann_ret[s]) for s in range(self.n_states)}

            # Classify the recent window's posterior-weighted annual return.
            k = min(len(states), 42)
            post_share = posteriors[-k:].mean(axis=0)  # mean posterior per state
            recent_exp_ann = float(np.dot(post_share, ann_ret))
            regime = _label(recent_exp_ann)
            current_state = int(np.argmax(post_share))
            # Confidence is the recent posterior mass in states with the label.
            conf_mass = sum(float(post_share[s]) for s in range(self.n_states) if label_for[s] == regime)
            confidence = int(round(conf_mass * 100))

            # Occupancy aggregated by regime label.
            occ = {}
            for s in range(self.n_states):
                occ[label_for[s]] = round(occ.get(label_for[s], 0.0) + float(np.mean(states == s)), 3)
            metrics = {
                "method_detail": f"GaussianHMM({self.n_states} states, diag)",
                "converged": bool(model.monitor_.converged),
                "current_state": current_state,
                "recent_exp_ann_pct": round(recent_exp_ann * 100, 2),
                "state_ann_return_pct": {f"state{s}": round(float(ann_ret[s]) * 100, 2) for s in range(self.n_states)},
                "state_labels": {f"state{s}": label_for[s] for s in range(self.n_states)},
                "regime_occupancy": occ,
                "n_observations": int(len(feats)),
            }
            logger.info(f"Regime[HMM]: {regime} ({confidence}%)  converged={metrics['converged']}  occ={occ}")
            return {"regime": regime, "confidence": confidence, "method": "hmm", "metrics": metrics}
        except Exception as exc:  # non-convergence or singular covariance
            logger.warning(f"HMM regime fit failed ({exc})")
            return None

    # Breadth and momentum fallback.
    def _heuristic_detect(self, master_data: dict) -> dict:
        breadth_scores, adx_scores, momentum_scores, rsi_scores = [], [], [], []

        for ticker, df in master_data.items():
            if df is None or df.empty:
                continue
            latest = df.iloc[-1]
            close = float(latest.get("close", 0) or 0)
            sma200 = float(latest.get("sma200", 0) or 0)
            adx = float(latest.get("adx", 0) or 0)
            mom = float(latest.get("momentum_10d", 0) or 0)
            rsi = float(latest.get("rsi", 50) or 50)

            if sma200 > 0:
                breadth_scores.append(1 if close > sma200 else 0)
            if adx > 0:
                adx_scores.append(adx)
            momentum_scores.append(mom)
            rsi_scores.append(rsi)

        if not breadth_scores:
            return {"regime": "unknown", "confidence": 0, "method": "heuristic", "metrics": {}}

        pct_above_sma200 = float(np.mean(breadth_scores))
        avg_adx = float(np.mean(adx_scores)) if adx_scores else 20.0
        avg_momentum = float(np.mean(momentum_scores)) if momentum_scores else 0.0
        avg_rsi = float(np.mean(rsi_scores)) if rsi_scores else 50.0

        # Require sufficient ADX trend strength for a directional regime label.
        trending = avg_adx >= 22.0

        if trending and pct_above_sma200 > 0.60 and avg_momentum > 0 and avg_rsi > 50:
            regime = "bull"
            confidence = min(100, int((pct_above_sma200 - 0.60) * 250 + 50))
        elif trending and pct_above_sma200 < 0.40 and avg_momentum < 0 and avg_rsi < 50:
            regime = "bear"
            confidence = min(100, int((0.40 - pct_above_sma200) * 250 + 50))
        else:
            regime = "sideways"
            confidence = int(50 - abs(pct_above_sma200 - 0.50) * 100)

        metrics = {
            "pct_above_sma200": round(pct_above_sma200 * 100, 1),
            "avg_adx": round(avg_adx, 1),
            "avg_momentum_pct": round(avg_momentum * 100, 2),
            "avg_rsi": round(avg_rsi, 1),
            "n_stocks": len(breadth_scores),
        }
        logger.info(f"Regime[heuristic]: {regime} ({confidence}%)  breadth={pct_above_sma200:.1%}  ADX={avg_adx:.1f}")
        return {"regime": regime, "confidence": confidence, "method": "heuristic", "metrics": metrics}
