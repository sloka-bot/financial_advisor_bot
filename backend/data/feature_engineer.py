"""Generate technical indicators while keeping prices, labels and quality flags separate."""

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

    # Public API.

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

        # Calculate moving averages for short- and long-term trend features.
        df["sma20"] = ta.trend.sma_indicator(close, 20)
        df["sma50"] = ta.trend.sma_indicator(close, 50)
        df["sma200"] = ta.trend.sma_indicator(close, 200)
        df["ema12"] = ta.trend.ema_indicator(close, 12)
        df["ema26"] = ta.trend.ema_indicator(close, 26)

        # Measure the change in the 20-session moving average.
        df["sma20_slope"] = df["sma20"].diff(5) / (df["sma20"].shift(5) + 1e-9)

        # Record whether the 50-session average is above the 200-session average.
        df["golden_cross"] = (df["sma50"] > df["sma200"]).astype(int)

        # Calculate MACD and its signal difference from fast and slow moving averages.
        macd = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_hist"] = macd.macd_diff()

        # Calculate RSI from relative gains and losses over 14 sessions.
        df["rsi"] = ta.momentum.rsi(close, window=14)

        # Locate the close within the recent high-low range.
        stoch = ta.momentum.StochasticOscillator(high, low, close, window=14, smooth_window=3)
        df["stoch_k"] = stoch.stoch()
        df["stoch_d"] = stoch.stoch_signal()  # 3-day smoothed %K

        # Calculate Williams percent R from proximity to the recent high.
        df["williams_r"] = ta.momentum.williams_r(high, low, close, lbp=14)

        # Measure returns over multiple momentum horizons.
        df["momentum_10d"] = close.pct_change(10)
        df["momentum_21d"] = close.pct_change(21)
        df["roc_10"] = ta.momentum.roc(close, window=10)

        # Calculate average true range as a price-volatility feature.
        df["atr"] = ta.volatility.average_true_range(high, low, close, window=14)
        df["atr_pct"] = df["atr"] / (close + 1e-9)  # normalise by price for comparability

        # Locate price within two-standard-deviation bands around the 20-session mean.
        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_pct"] = bb.bollinger_pband()
        df["bb_bandwidth"] = bb.bollinger_wband()

        # Historical volatility is the main risk input for scoring.
        df["daily_return"] = close.pct_change()
        df["volatility"] = df["daily_return"].rolling(20).std()

        # Compare short-term volatility with longer-term volatility.
        long_vol = df["daily_return"].rolling(60).std()
        df["vol_ratio"] = (df["volatility"] / long_vol.replace(0, np.nan)).fillna(1.0)

        # Accumulate signed volume according to the daily price direction.
        df["obv"] = ta.volume.on_balance_volume(close, vol)
        df["obv_momentum"] = df["obv"].pct_change(10)

        # Scale current volume by its 20-session average.
        df["volume_sma20"] = vol.rolling(20).mean()
        df["volume_ratio"] = vol / (df["volume_sma20"] + 1)

        # Weight volume by the close's position within the daily range.
        df["acc_dist"] = ta.volume.acc_dist_index(high, low, close, vol)

        # Calculate returns over weekly, monthly and quarterly windows.
        df["weekly_return"] = close.pct_change(5)
        df["monthly_return"] = close.pct_change(21)
        df["quarterly_return"] = close.pct_change(63)

        # Measure changes in momentum across successive observations.
        df["return_accel"] = df["daily_return"].diff(5)

        # Express price distance from moving averages on a comparable scale.
        df["close_to_sma20"] = close / (df["sma20"] + 1e-9) - 1
        df["close_to_sma50"] = close / (df["sma50"] + 1e-9) - 1

        # Locate the close within its 52-week price range.
        h250 = high.rolling(250, min_periods=50).max()
        l250 = low.rolling(250, min_periods=50).min()
        df["week52_position"] = (close - l250) / (h250 - l250 + 1e-9)

        # Scale the candle body by the daily high-low range.
        if "open" in df.columns:
            body = (close - df["open"]).abs()
            shadow = (high - low).replace(0, np.nan)
            df["candle_body_ratio"] = body / shadow
        else:
            df["candle_body_ratio"] = 0.5

        # Measure typical-price deviation using the commodity channel index.
        df["cci"] = ta.trend.cci(high, low, close, window=20)

        # Calculate directional trend strength using ADX.
        df["adx"] = ta.trend.adx(high, low, close, window=14)

        # Replace non-finite indicator values before filtering the warm-up period.
        df.replace([float("inf"), float("-inf")], float("nan"), inplace=True)

        # Drop incomplete SMA200 rows; model training handles other missing features.
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
