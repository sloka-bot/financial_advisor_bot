import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "open", "high", "low", "close", "volume",
    "sma20", "sma50", "ema20",
    "macd", "macd_signal", "macd_hist",
    "rsi", "atr", "momentum", "volatility",
    "daily_return", "weekly_return", "monthly_return",
    "volume_ratio", "close_to_sma20", "close_to_sma50",
    "sent_score", "sent_weighted",
    "sent_pos", "sent_neg", "sent_neu", "sent_news_count",
]


class FeatureFusion:

    def __init__(self, features_dir="data/features", sentiment_dir="data/sentiment"):
        self.features_dir = Path(features_dir)
        self.sentiment_dir = Path(sentiment_dir)

    def fuse_ticker(self, ticker, save=True):
        tech_path = self.features_dir / f"{ticker}.csv"
        if not tech_path.exists():
            logger.warning(f"{ticker}: no technical features found")
            return None

        df = pd.read_csv(tech_path, index_col=0, parse_dates=True).copy()

        sent_path = self.sentiment_dir / f"{ticker.replace('/', '_')}.csv"
        if sent_path.exists():
            sent = pd.read_csv(sent_path, index_col=0, parse_dates=True).rename(columns={
                "sentiment_score": "sent_score",
                "sentiment_label": "sent_label",
                "pos_count":       "sent_pos",
                "neg_count":       "sent_neg",
                "neu_count":       "sent_neu",
                "news_count":      "sent_news_count",
                "weighted_score":  "sent_weighted",
            })
            df = df.join(sent[["sent_score", "sent_weighted", "sent_pos", "sent_neg",
                                "sent_neu", "sent_news_count", "sent_label"]], how="left")
            # news doesn't come every trading day — carry last value forward
            numeric = ["sent_score", "sent_weighted", "sent_pos", "sent_neg", "sent_neu", "sent_news_count"]
            df[numeric] = df[numeric].ffill(limit=5)

        # fill any gaps with neutral defaults
        for col in ["sent_score", "sent_weighted", "sent_pos", "sent_neg", "sent_neu", "sent_news_count"]:
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = df[col].fillna(0.0)
        if "sent_label" not in df.columns:
            df["sent_label"] = "neutral"
        else:
            df["sent_label"] = df["sent_label"].fillna("neutral")

        # target: next day's return
        df["target_return"]    = df["daily_return"].shift(-1)
        df["target_direction"] = (df["target_return"] > 0).astype(int)
        df = df.dropna(subset=["target_return"])

        if save:
            df.to_csv(self.features_dir / f"{ticker}_master.csv")
            logger.info(f"  {ticker}: master saved ({len(df)} rows)")

        return df

    def fuse_universe(self, tickers):
        results = {"success": [], "failed": []}
        logger.info(f"Fusing features for {len(tickers)} tickers...")

        for ticker in tickers:
            df = self.fuse_ticker(ticker)
            if df is not None and not df.empty:
                results["success"].append(ticker)
            else:
                results["failed"].append(ticker)

        logger.info(f"  Done: {len(results['success'])} ok, {len(results['failed'])} failed")
        return results

    def load_master(self, ticker):
        path = self.features_dir / f"{ticker}_master.csv"
        if not path.exists():
            return None
        return pd.read_csv(path, index_col=0, parse_dates=True)

    def load_all(self, tickers):
        frames = []
        for t in tickers:
            df = self.load_master(t)
            if df is not None and not df.empty:
                df["ticker"] = t
                frames.append(df)
        if not frames:
            logger.error("No master files found — run fuse_universe first")
            return None
        combined = pd.concat(frames, ignore_index=False)
        logger.info(f"Combined: {len(combined)} rows from {len(frames)} tickers")
        return combined

    @staticmethod
    def feature_cols():
        return FEATURE_COLS
