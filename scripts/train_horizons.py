"""
train_horizons.py

Train and deploy SIDE models for the short horizons (1 and 5 trading sessions),
so the Stock Predictor can offer a 1 / 5 / 21-day forecast. The 21-day models are
the primary deployed artifacts (models/xgboost, models/lstm) and are trained by
retrain_models.py / retrain_lstm.py - this script never touches them or the shared
registry.

Each horizon H gets its own directory:
    models/xgboost/h{H}/   models/lstm/h{H}/
The forward-return label is recomputed PER TICKER at horizon H before training, so
an H-day model learns an H-day target (not the 21-day label baked into the masters).

Run (from the project root):
    ./venv/bin/python scripts/train_horizons.py --limit 5 --epochs 2     # fast smoke test
    caffeinate -i ./venv/bin/python scripts/train_horizons.py            # full 1- and 5-day models
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.data.fusion import FeatureFusion
from backend.infra.reproducibility import set_seed
from backend.prediction.lstm_model import LSTMForecaster
from backend.prediction.xgboost_model import XGBoostForecaster
from backend.universe.universe_builder import UniverseBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_horizons")

FEATURES_DIR = Path("data/features")


def relabel(df, horizon):
    """Recompute the forward-return label at `horizon` sessions, per ticker, so no
    label crosses the seam between two tickers."""
    out = df.copy()
    fwd_close = out.groupby("ticker")["close"].shift(-horizon)
    out["target_return"] = fwd_close / out["close"] - 1.0
    out["target_direction"] = np.where(
        out["target_return"].notna(), (out["target_return"] > 0).astype(float), np.nan
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 5], help="short horizons to train (21 is the primary, trained elsewhere)")
    ap.add_argument("--epochs", type=int, default=50, help="LSTM epochs max (early stopping ends it sooner)")
    ap.add_argument("--batch-size", type=int, default=256, help="LSTM batch size")
    ap.add_argument("--patience", type=int, default=10, help="LSTM early-stop patience")
    ap.add_argument("--limit", type=int, default=None, help="first N masters only (smoke test)")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--end", default="2023-12-31", help="news-window end; matches the report and eval scripts")
    ap.add_argument("--no-strict-universe", action="store_true")
    args = ap.parse_args()

    set_seed(42)

    tickers = sorted(p.stem.replace("_master", "") for p in FEATURES_DIR.glob("*_master.csv"))
    if args.limit:
        tickers = tickers[: args.limit]
    if not tickers:
        logger.error("No master datasets in %s", FEATURES_DIR)
        sys.exit(1)

    logger.info("Loading %d master datasets...", len(tickers))
    combined = FeatureFusion().load_all(tickers)
    if combined is None or combined.empty:
        logger.error("No usable master data loaded")
        sys.exit(1)

    import pandas as pd

    combined = combined.loc[
        (combined.index >= pd.Timestamp(args.start)) & (combined.index <= pd.Timestamp(args.end))
    ]
    if combined.empty:
        logger.error("No rows in [%s, %s]", args.start, args.end)
        sys.exit(1)

    if not args.no_strict_universe:
        combined = UniverseBuilder().filter_eligible_rows(combined, strict=True)
    logger.info("Base panel: %d rows across %d tickers", len(combined), combined["ticker"].nunique())

    for horizon in args.horizons:
        logger.info("=" * 70)
        logger.info("HORIZON %d", horizon)
        panel = relabel(combined, horizon)
        n_lbl = int(panel["target_return"].notna().sum())
        logger.info("Relabelled at H=%d: %d labelled rows", horizon, n_lbl)

        xgb_dir = f"models/xgboost/h{horizon}"
        lstm_dir = f"models/lstm/h{horizon}"

        logger.info("Training XGBoost (H=%d) -> %s", horizon, xgb_dir)
        xgb_res = XGBoostForecaster(models_dir=xgb_dir, horizon=horizon).train(panel, register=False)
        logger.info(
            "  XGBoost H=%d: cv_auc_mean=%s cv_direction_acc=%s deployed=%s",
            horizon, xgb_res.get("cv_auc_mean"), xgb_res.get("cv_direction_acc"), xgb_res.get("deployed"),
        )

        logger.info("Training LSTM (H=%d, epochs=%d, batch=%d) -> %s", horizon, args.epochs, args.batch_size, lstm_dir)
        lstm_res = LSTMForecaster(models_dir=lstm_dir, horizon=horizon).train(
            panel, epochs=args.epochs, batch_size=args.batch_size, patience=args.patience
        )
        logger.info(
            "  LSTM H=%d: %s", horizon,
            {k: lstm_res.get(k) for k in ("best_val_loss", "best_dir_acc", "best_epoch", "total_epochs")},
        )

    logger.info("Done. Side models written under models/xgboost/h*/ and models/lstm/h*/.")


if __name__ == "__main__":
    main()
