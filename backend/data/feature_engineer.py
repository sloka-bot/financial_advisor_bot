"""
Technical indicator pipeline.

Each indicator group is justified by the quantitative finance literature:
  - Trend: identifies the dominant price direction (Murphy, 1999)
  - Momentum: measures rate of price change; underpins Jegadeesh & Titman (1993)
  - Volatility: ATR and Bollinger Bands measure compression/expansion cycles
  - Volume: OBV and Acc/Dist detect institutional participation behind moves
  - Cycle/Strength: ADX and CCI distinguish trending from ranging conditions

Feature selection is defined in backend/evaluation/experiments.py. Dataset
columns also contain prices, labels and quality flags, which are not predictors.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import ta

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Build technical predictors and forward targets from cleaned prices."""

    def __init__(self, processed_dir="data/processed", features_dir="data/features"):
        self.processed_dir = Path(processed_dir)
        self.features_dir = Path(features_dir)
        self.features_dir.mkdir(parents=True, exist_ok=True)

    # - public API -

    def generate(self, ticker: str, df: pd.DataFrame = None, save: bool = True):
        """Calculate features and labels, returning None for insufficient history."""
        if df is None:
            df = self._load_cleaned(ticker)
        if df is None or df.empty:
            return None
        if len(df) < 200:
            logger.warning("%s: fewer than 200 observations; skipping technical indicators", ticker)
            return None

        df = df.copy()
        close = df["close"]
        high = df["high"]
        low = df["low"]
        vol = df["volume"]

        # - TREND -
        # SMA crossovers define regime: price above both SMAs = uptrend.
        # The 200-day SMA is the institutional benchmark for long-term direction.
        df["sma20"] = ta.trend.sma_indicator(close, 20)
        df["sma50"] = ta.trend.sma_indicator(close, 50)
        df["sma200"] = ta.trend.sma_indicator(close, 200)
        df["ema12"] = ta.trend.ema_indicator(close, 12)
        df["ema26"] = ta.trend.ema_indicator(close, 26)

        # Slope of the 20-day SMA: captures whether the trend is accelerating.
        # A flattening SMA often precedes a reversal even while price is above it.
        df["sma20_slope"] = df["sma20"].diff(5) / (df["sma20"].shift(5) + 1e-9)

        # Golden cross (SMA50 crosses above SMA200): one of the most studied
        # trend-following signals; binary flag avoids ordinal scale assumptions.
        df["golden_cross"] = (df["sma50"] > df["sma200"]).astype(int)

        # - MACD -
        # Gerald Appel (1979): difference between fast/slow EMA captures momentum
        # direction. The histogram (MACD minus signal) shows momentum's own trend
        # and fires divergence signals before price reverses.
        macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff()

        # - MOMENTUM -
        # RSI (Wilder, 1978): normalises the average gain vs average loss over 14
        # days into [0,100]. Readings above 70 indicate overbought conditions where
        # mean reversion risk outweighs continuation probability.
        df["rsi"] = ta.momentum.rsi(close, window=14)

        # Stochastic %K: compares current close to its 14-day high-low range.
        # More reactive than RSI to short-term price exhaustion.
        stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()  # 3-day smoothed %K

        # Williams %R (Williams, 1973): inverted stochastic; measures proximity to
        # the recent high. Useful for catching reversals at momentum extremes.
        df["williams_r"] = ta.momentum.williams_r(high, low, close, lbp=14)

        # Cross-sectional momentum (Jegadeesh & Titman): raw multi-period returns
        # are the core ranking factor. 10-day and 21-day capture different horizons.
        df["momentum_10d"] = close.pct_change(10)
        df["momentum_21d"] = close.pct_change(21)
        df["roc_10"] = ta.momentum.roc(close, window=10)

        # - VOLATILITY -
        # ATR (Wilder, 1978): rolling mean of the True Range, where True Range is the
        # maximum of (high - low), |high - prev_close| and |low - prev_close|. Used as
        # a volatility scale for position sizing.
        df["atr"] = ta.volatility.average_true_range(high, low, close, window=14)
        df["atr_pct"] = df["atr"] / (close + 1e-9)  # normalise by price for comparability

        # Bollinger Bands (Bollinger, 1992): price channels at ±2 standard deviations
        # from the 20-day mean. %B pinpoints where price sits in the band [0=lower, 1=upper].
        # Bandwidth captures the volatility regime (squeeze vs expansion).
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_pct"] = bb.bollinger_pband()
        df["bb_bandwidth"] = bb.bollinger_wband()

        # Historical vol is the fundamental risk input for the scoring engine.
        df["daily_return"] = close.pct_change()
        df["volatility"] = df["daily_return"].rolling(20).std()

        # Volatility ratio (short-run vs long-run): > 1.5 signals a volatility
        # spike; the model can down-weight signals that fire in noisy regimes.
        long_vol = df["daily_return"].rolling(60).std()
        df["vol_ratio"] = (df["volatility"] / long_vol.replace(0, np.nan)).fillna(1.0)

        # - VOLUME -
        # On-Balance Volume (Granville, 1963): running total that adds volume on up
        # days and subtracts it on down days, summarising volume flow direction.
        df["obv"] = ta.volume.on_balance_volume(close, vol)
        df["obv_momentum"] = df["obv"].pct_change(10)

        # Volume ratio: today's volume vs 20-day average.
        # > 2.0 often accompanies institutional block trades and breakouts.
        df["volume_sma20"] = vol.rolling(20).mean()
        df["volume_ratio"] = vol / (df["volume_sma20"] + 1)

        # Accumulation/Distribution (Chaikin): weights volume by the close's position
        # within the day's range. Separates buying pressure from selling pressure.
        df["acc_dist"] = ta.volume.acc_dist_index(high, low, close, vol)

        # - MULTI-PERIOD RETURNS -
        # Multi-horizon returns let the model distinguish short-term bounce-backs
        # (weekly) from medium-term trend followers (quarterly).
        df["weekly_return"] = close.pct_change(5)
        df["monthly_return"] = close.pct_change(21)
        df["quarterly_return"] = close.pct_change(63)

        # Second derivative of returns: positive acceleration on positive momentum
        # signals a strengthening trend rather than an exhausting one.
        df["return_accel"] = df["daily_return"].diff(5)

        # - PRICE POSITION FEATURES -
        # Distance from moving averages is more informative than the raw price level
        # because it generalises across stocks of different absolute values.
        df["close_to_sma20"] = close / (df["sma20"] + 1e-9) - 1
        df["close_to_sma50"] = close / (df["sma50"] + 1e-9) - 1

        # 52-week position [0, 1]: 0 = at annual low, 1 = at annual high.
        # Research shows stocks near their 52-week high have positive momentum
        # (George & Hwang, 2004), controlling for this prevents the model from
        # confusing "expensive" with "overpriced".
        h250 = high.rolling(250, min_periods=50).max()
        l250 = low.rolling(250, min_periods=50).min()
        df["week52_position"] = (close - l250) / (h250 - l250 + 1e-9)

        # Candle body ratio: proportion of the day's range that is directional.
        # Small bodies (doji candles) = indecision; large bodies = commitment.
        if "open" in df.columns:
            body = (close - df["open"]).abs()
            shadow = (high - low).replace(0, np.nan)
            df["candle_body_ratio"] = body / shadow
        else:
            df["candle_body_ratio"] = 0.5

        # - CYCLE AND TREND STRENGTH -
        # CCI (Lambert, 1980): distance of typical price from its moving average,
        # expressed in multiples of mean deviation. Positive = above average, not
        # overbought in itself - strength of trend matters more than direction.
        df["cci"] = ta.trend.cci(high, low, close, window=20)

        # ADX (Wilder, 1978): trend-strength measure independent of direction
        # (higher = stronger trend). Provided to the model as an input feature.
        df["adx"] = ta.trend.adx(high, low, close, window=14)

        # Drop rows where SMA200 has not yet converged (first ~200 bars).
        # Including uninitialised indicator values would teach the model
        # spurious patterns from the early history of each stock.
        # Clip any inf/-inf that technical indicators can produce
        # (e.g. RSI when high == low, Williams %R with zero range).
        # We clip rather than drop so we do not lose entire tickers
        # whose rolling windows (like week52) are still warming up.
        df.replace([float("inf"), float("-inf")], float("nan"), inplace=True)

        # Only drop rows where the SMA200 warmup is incomplete.
        # NaN in other columns is handled by the XGBoost/LSTM masks,
        # which filter with np.isfinite() before training.
        df = df.dropna(subset=["sma200"])

        logger.info(f"{ticker}: {len(df)} rows · {len(df.columns)} dataset columns")

        if save:
            df.to_csv(self.features_dir / f"{ticker}.csv")

        return df

    def generate_universe(self, tickers: list, progress_cb=None) -> dict:
        """Generate feature files for each ticker and record failures."""
        results = {"success": [], "failed": []}
        n = len(tickers)
        logger.info(f"Generating features for {n} tickers...")
        for i, ticker in enumerate(tickers, 1):
            df = self.generate(ticker)
            (results["success"] if df is not None and not df.empty else results["failed"]).append(ticker)
            if i % 10 == 0 or i == n:
                logger.info(f"  [{i}/{n}]  ok={len(results['success'])}  failed={len(results['failed'])}")
            if progress_cb is not None:
                progress_cb(i, n)
        return results

    def load(self, ticker: str):
        """Read cached technical features, returning None when the file is absent."""
        path = self.features_dir / f"{ticker}.csv"
        return pd.read_csv(path, index_col=0, parse_dates=True) if path.exists() else None


    def _load_cleaned(self, ticker: str):
        path = self.processed_dir / f"{ticker}.csv"
        if not path.exists():
            logger.warning(f"{ticker}: no cleaned data at {path}")
            return None
        return pd.read_csv(path, index_col=0, parse_dates=True)
