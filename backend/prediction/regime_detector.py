"""
regime_detector.py

Detects the current market regime (bull / bear / sideways) used to gate the
portfolio constructor's conservatism.

The primary detector is a Gaussian Hidden Markov Model (hmmlearn) fitted on the
market's own return and volatility history - the same model family used in the
offline experiment harness (run_experiments.py `_hmm_select`), so the live app
and the experiments now speak about "regime" in the same terms. The HMM learns
latent states from the data rather than firing on hand-tuned thresholds; each
latent state is then labelled bull / bear / sideways by its mean return, and the
regime for "now" is the state the model assigns to the most recent observation,
with the state's posterior probability as the confidence.

If hmmlearn is unavailable, or there is too little history to fit a stable HMM,
the detector falls back to the previous cross-sectional heuristic (breadth +
ADX trend strength + momentum) so the app never loses regime information. The
returned `method` field records which path produced the answer.

References:
  Hamilton (1989) - regime-switching models of the business cycle.
  Rabiner (1989) - hidden Markov models.
"""

import logging

import numpy as np
import pandas as pd

from backend.config.settings import TRADING_DAYS  # annualisation factor (single source)

logger = logging.getLogger(__name__)

N_STATES = 3  # bull / bear / sideways
MIN_OBS = 60  # minimum market observations to attempt an HMM fit
VOL_WINDOW = 10  # rolling window for the volatility feature

BAND_ANN = 0.05  # ±5%/yr band: |annualised state return| inside this = sideways


class RegimeDetector:
    """Gaussian-HMM market-regime detector with a heuristic fallback."""

    def __init__(self, n_states: int = N_STATES):
        self.n_states = n_states

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    def detect(self, master_data: dict, strict: bool = False) -> dict:
        """Detect the market regime.

        The heuristic fallback keeps the LIVE app robust when hmmlearn is missing
        or the fit fails. For EXPERIMENTS that must be HMM, pass strict=True: the
        method then returns an explicit invalid result (method='hmm_unavailable',
        valid=False) instead of silently reporting heuristic numbers as if they
        were HMM - two runs of "the same" system must not quietly evaluate
        different algorithms.
        """
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

    # ------------------------------------------------------------------ #
    # Primary path: Gaussian HMM on market return + volatility
    # ------------------------------------------------------------------ #
    def _market_features(self, master_data: dict):
        """Build the market's [return, rolling-vol] feature matrix from the mean
        close across the universe. Returns (feats ndarray, aligned index) or
        (None, None) if a time series cannot be formed."""
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

        mkt = pd.DataFrame(closes).mean(axis=1).sort_index()  # equal-weight market index
        ret = mkt.pct_change()
        vol = ret.rolling(VOL_WINDOW).std()
        feat_df = pd.DataFrame({"ret": ret, "vol": vol}).dropna()
        if feat_df.empty:
            return None, None
        return feat_df.values.astype(float), feat_df.index

    def _hmm_detect(self, feats: np.ndarray):
        """Fit a GaussianHMM, label states by mean return, and read off the regime
        for the latest observation. Returns None on any failure so the caller can
        fall back to the heuristic."""
        try:
            from hmmlearn.hmm import GaussianHMM
        except Exception as exc:  # library not installed
            logger.warning(f"hmmlearn unavailable ({exc})")
            return None

        try:
            # Standardise features before fitting: return (~1e-2) and rolling vol
            # sit on similar-but-not-equal scales, and on raw values the EM fit
            # routinely fails to converge and collapses to two states (verified).
            # Standardising makes the fit converge and separate three states.
            mu, sigma = feats.mean(axis=0), feats.std(axis=0) + 1e-9
            feats_z = (feats - mu) / sigma

            model = GaussianHMM(n_components=self.n_states, covariance_type="diag", n_iter=200, random_state=42)
            model.fit(feats_z)
            states = model.predict(feats_z)
            posteriors = model.predict_proba(feats_z)

            # Label each latent state by its ABSOLUTE annualised mean return, not
            # by rank. Ranking is wrong on a single-regime market: a pure bear
            # market still has a "least-negative" state that a rank rule would call
            # bull. Anchoring to an absolute band (±BAND_ANN per year) means an
            # all-down market's states are all labelled bear, an all-up market's
            # all bull, and a switching market's states split correctly.
            mean_ret_daily = np.array(
                [feats[states == s, 0].mean() if np.any(states == s) else 0.0 for s in range(self.n_states)]
            )
            ann_ret = mean_ret_daily * TRADING_DAYS

            def _label(a):
                return "bull" if a > BAND_ANN else ("bear" if a < -BAND_ANN else "sideways")

            label_for = {s: _label(ann_ret[s]) for s in range(self.n_states)}

            # "Current" regime over the recent window (~2 months). Reading a single
            # sub-state is fragile - a bear market has bounce days the HMM isolates
            # as a positive sub-state, and the last weeks can sit there. Instead take
            # the POSTERIOR-WEIGHTED expected annualised return across states over the
            # window and band-classify that: it aggregates the whole recent mixture,
            # so a mostly-down window reads bear even with an occasional bounce state.
            k = min(len(states), 42)
            post_share = posteriors[-k:].mean(axis=0)  # avg posterior per state (sums to 1)
            recent_exp_ann = float(np.dot(post_share, ann_ret))
            regime = _label(recent_exp_ann)
            current_state = int(np.argmax(post_share))
            # confidence = share of recent posterior mass in states carrying the label
            conf_mass = sum(float(post_share[s]) for s in range(self.n_states) if label_for[s] == regime)
            confidence = int(round(conf_mass * 100))

            # occupancy aggregated by regime label (states can share a label)
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
        except Exception as exc:  # non-convergence, singular cov, etc.
            logger.warning(f"HMM regime fit failed ({exc})")
            return None

    # ------------------------------------------------------------------ #
    # Fallback path: cross-sectional heuristic (breadth + ADX + momentum)
    # ------------------------------------------------------------------ #
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

        # Trend strength (ADX) gates a directional call: a market is only labelled
        # bull/bear when it is actually trending (ADX >= 22), else sideways.
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
