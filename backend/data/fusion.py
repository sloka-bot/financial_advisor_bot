"""
fusion.py

Merges the technical feature matrix with the daily FinBERT sentiment signal to
produce the per-ticker master dataset the models train on.

The sentiment columns are already named `sent_*` by the analyzer, so there is no
rename step here. News does not arrive
every trading day, so a recent score is carried forward for a few sessions, but
`sent_no_news` records whether that day actually had fresh news - a no-news day
stays distinguishable from a neutral-news day. The forward-return
label is defined once here and dropped where it cannot be computed.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from backend.config.settings import PREDICTION_HORIZON

logger = logging.getLogger(__name__)

# Sentiment SCORES decay slowly and may be carried forward a few sessions.
# Sentiment COUNTS describe how much FRESH news arrived THAT day and must NOT be
# carried forward - a carried day has no fresh news, so its counts stay 0. The
# ranker reads sent_news_count as coverage, so carrying it would fake coverage.
SENT_SCORES = ["sent_score", "sent_weighted", "sent_disagreement"]
SENT_COUNTS = ["sent_pos", "sent_neg", "sent_neu", "sent_news_count"]
SENT_NUMERIC = SENT_SCORES + SENT_COUNTS
FFILL_LIMIT = 5  # carry a recent score forward at most a trading week


def _verify_label_horizon(df: pd.DataFrame, horizon: int) -> None:
    """Fail loudly if a forward-return label does not use the close exactly
    `horizon` sessions ahead of its anchor row. Guards against an off-by-one or a
    silently misaligned target when sessions are missing (weekends, holidays)."""
    closes = df["close"].to_numpy()
    dates = np.asarray(df.index)
    tgt = df["target_return"].to_numpy()
    end = df["target_end_date"].to_numpy()
    n = len(df)
    labelled = np.flatnonzero(~np.isnan(tgt))
    labelled = labelled[labelled + horizon < n]
    if labelled.size == 0:
        return
    if not np.array_equal(end[labelled], dates[labelled + horizon]):
        raise ValueError("target_end_date is not H sessions ahead of the label anchor")
    expected = closes[labelled + horizon] / closes[labelled] - 1.0
    got = tgt[labelled]
    fin = np.isfinite(expected) & np.isfinite(got)
    if fin.any() and not np.allclose(expected[fin], got[fin], atol=1e-9, rtol=1e-6):
        raise ValueError("target_return does not equal close[t+H]/close[t]-1")


class FeatureFusion:
    """Align technical features and available sentiment on decision dates."""

    def __init__(self, features_dir="data/features", sentiment_dir="data/sentiment"):
        self.features_dir = Path(features_dir)
        self.sentiment_dir = Path(sentiment_dir)

    def fuse_ticker(self, ticker, save=True):
        """Merge ticker features with sentiment and persist the master dataset."""
        tech_path = self.features_dir / f"{ticker}.csv"
        if not tech_path.exists():
            logger.warning(f"{ticker}: no technical features found")
            return None
        df = pd.read_csv(tech_path, index_col=0, parse_dates=True).sort_index()

        paths = [
            self.sentiment_dir / f"{ticker}.csv",
            self.sentiment_dir / "historical" / f"{ticker}.csv",
            self.sentiment_dir / "live" / f"{ticker}.csv",
        ]
        sources = []
        for path in paths:
            if not path.exists():
                continue
            try:
                source = pd.read_csv(path, index_col=0)
                source.index = pd.to_datetime(source.index, errors="coerce")
                source = source.loc[source.index.notna()]
                if not source.empty:
                    sources.append(source)
            except (pd.errors.EmptyDataError, pd.errors.ParserError, ValueError) as exc:
                logger.warning("%s: skipping invalid sentiment file %s: %s", ticker, path, exc)
        if sources:
            sent = pd.concat(sources).sort_index()
            sent = sent[~sent.index.duplicated(keep="last")]
            # Move news on non-trading days to the next observed price session.
            positions = df.index.searchsorted(sent.index)
            valid = positions < len(df.index)
            sent = sent.loc[valid].copy()
            sent.index = df.index[positions[valid]]
            aggregations = {c: ("sum" if c in SENT_COUNTS else "mean") for c in sent.columns if c in SENT_NUMERIC}
            if not aggregations:
                logger.warning("%s: sentiment has no recognized numeric columns; using no-news defaults", ticker)
                sent = pd.DataFrame(index=pd.DatetimeIndex([]))
            else:
                for column in aggregations:
                    sent[column] = pd.to_numeric(sent[column], errors="coerce")
                sent = sent.groupby(level=0).agg(aggregations)
            sent["sent_label"] = np.where(
                sent.get("sent_score", 0) > 0,
                "positive",
                np.where(sent.get("sent_score", 0) < 0, "negative", "neutral"),
            )
            keep = [c for c in SENT_NUMERIC + ["sent_label"] if c in sent.columns]
            df = df.join(sent[keep], how="left")

            # a day has FRESH news iff it had a sentiment row with a positive count
            had_news = (
                df["sent_news_count"].fillna(0) > 0 if "sent_news_count" in df else pd.Series(False, index=df.index)
            )
            # carry only the SCORE columns forward (sentiment decays slowly);
            # counts stay 0 on carried/no-news days so coverage is never faked
            score_cols = [c for c in SENT_SCORES if c in df.columns]
            df[score_cols] = df[score_cols].ffill(limit=FFILL_LIMIT)
            count_cols = [c for c in SENT_COUNTS if c in df.columns]
            df[count_cols] = df[count_cols].fillna(0.0)
            # carry sent_label forward WITH the score (same limit) so the label and
            # the numeric score never disagree - a carried +0.7 score would otherwise
            # show label "neutral". sent_no_news still flags that the news is carried,
            # not fresh, so the UI can say "carried positive; no new articles today".
            if "sent_label" in df.columns:
                df["sent_label"] = df["sent_label"].ffill(limit=FFILL_LIMIT)
            # no-news flag reflects THIS day, not the carried-forward score
            df["sent_no_news"] = (~had_news).astype(int)
        else:
            df["sent_no_news"] = 1

        # neutral defaults for anything still missing (beyond the carry window)
        for col in SENT_NUMERIC:
            df[col] = df[col].fillna(0.0) if col in df.columns else 0.0
        df["sent_label"] = df["sent_label"].fillna("neutral") if "sent_label" in df.columns else "neutral"

        # Forward-return label over the horizon (strictly future). The newest `h`
        # rows have no observed future yet: target_return is NaN there, and
        # target_direction stays NaN too, not 0, so an unobserved future is not
        # labelled "down" (`nan > 0` is False).
        h = PREDICTION_HORIZON
        df["target_return"] = df["close"].shift(-h) / df["close"] - 1.0
        df["target_direction"] = np.where(df["target_return"].notna(), (df["target_return"] > 0).astype(float), np.nan)
        # Record the session each label matures on and verify the horizon alignment.
        df["target_end_date"] = pd.Series(df.index, index=df.index).shift(-h)
        _verify_label_horizon(df, h)
        # Keep these newest, label-less rows: they carry the most recent FEATURES
        # that live inference needs. Training callers must EXCLUDE rows with a
        # missing target (they do: build_panel recomputes and drops NaN fwd_ret;
        # the model trainers mask on target.notna()); they must never fill it.
        df.replace([float("inf"), float("-inf")], float("nan"), inplace=True)

        if save:
            df.to_csv(self.features_dir / f"{ticker}_master.csv")
            logger.info(f"  {ticker}: master saved ({len(df)} rows)")
        return df

    def fuse_universe(self, tickers, progress_cb=None):
        """Fuse each ticker dataset and report successful and failed inputs."""
        results = {"success": [], "failed": []}
        n = len(tickers)
        logger.info(f"Fusing features for {n} tickers...")
        for i, ticker in enumerate(tickers, 1):
            df = self.fuse_ticker(ticker)
            results["success" if (df is not None and not df.empty) else "failed"].append(ticker)
            if progress_cb is not None:
                progress_cb(i, n)
        logger.info(f"  Done: {len(results['success'])} ok, {len(results['failed'])} failed")
        return results

    def load_master(self, ticker):
        """Read a ticker master file, returning None when it is absent."""
        path = self.features_dir / f"{ticker}_master.csv"
        return pd.read_csv(path, index_col=0, parse_dates=True) if path.exists() else None

    def load_all(self, tickers):
        """Combine the available master datasets with ticker identifiers."""
        frames = []
        for t in tickers:
            df = self.load_master(t)
            if df is not None and not df.empty:
                df["ticker"] = t
                frames.append(df)
        if not frames:
            logger.error("No master files found - run fuse_universe first")
            return None
        combined = pd.concat(frames, ignore_index=False)
        logger.info(f"Combined: {len(combined)} rows from {len(frames)} tickers")
        return combined
