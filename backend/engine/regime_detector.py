import logging
import numpy as np

logger = logging.getLogger(__name__)


class RegimeDetector:
    """
    Classifies the current market into Bull, Bear, or Sideways using
    three cross-sectional signals derived from the tracked universe:

      1. Breadth: fraction of stocks trading above their 200-day SMA.
         > 60% = broad-based uptrend (Bull signal)
         < 40% = broad-based downtrend (Bear signal)

      2. Trend strength: median ADX across the universe.
         > 25 = a trending regime (directional)
         < 20 = a ranging regime (non-directional)

      3. Momentum: median 10-day return across the universe.
         Positive = recent upward bias, negative = downward bias

    These three signals together classify into one of four regimes.
    The confidence score reflects how strongly the signals agree.
    """

    def detect(self, master_data: dict) -> dict:
        if not master_data:
            return {'regime': 'unknown', 'confidence': 0, 'metrics': {}}

        breadth_scores = []
        adx_scores     = []
        momentum_scores = []
        rsi_scores     = []

        for ticker, df in master_data.items():
            if df is None or df.empty:
                continue
            latest = df.iloc[-1]

            close  = float(latest.get('close', 0)   or 0)
            sma200 = float(latest.get('sma200', 0)  or 0)
            adx    = float(latest.get('adx', 0)     or 0)
            mom    = float(latest.get('momentum_10d', 0) or 0)
            rsi    = float(latest.get('rsi', 50)    or 50)

            if sma200 > 0:
                breadth_scores.append(1 if close > sma200 else 0)
            if adx > 0:
                adx_scores.append(adx)
            momentum_scores.append(mom)
            rsi_scores.append(rsi)

        if not breadth_scores:
            return {'regime': 'unknown', 'confidence': 0, 'metrics': {}}

        pct_above_sma200 = float(np.mean(breadth_scores))
        avg_adx          = float(np.mean(adx_scores))    if adx_scores     else 20.0
        avg_momentum     = float(np.mean(momentum_scores)) if momentum_scores else 0.0
        avg_rsi          = float(np.mean(rsi_scores))    if rsi_scores     else 50.0

        # classify regime using breadth + trend strength + momentum
        is_trending = avg_adx > 22

        if pct_above_sma200 > 0.60 and avg_momentum > 0 and avg_rsi > 50:
            regime     = 'bull'
            confidence = min(100, int((pct_above_sma200 - 0.60) * 250 + 50))
        elif pct_above_sma200 < 0.40 and avg_momentum < 0 and avg_rsi < 50:
            regime     = 'bear'
            confidence = min(100, int((0.40 - pct_above_sma200) * 250 + 50))
        else:
            regime     = 'sideways'
            # confidence = how far from the extremes
            confidence = int(50 - abs(pct_above_sma200 - 0.50) * 100)

        metrics = {
            'pct_above_sma200': round(pct_above_sma200 * 100, 1),
            'avg_adx':          round(avg_adx, 1),
            'avg_momentum_pct': round(avg_momentum * 100, 2),
            'avg_rsi':          round(avg_rsi, 1),
            'n_stocks':         len(breadth_scores),
        }

        logger.info(f'Regime: {regime} ({confidence}%)  breadth={pct_above_sma200:.1%}  ADX={avg_adx:.1f}')
        return {'regime': regime, 'confidence': confidence, 'metrics': metrics}
