"""
retrain_lstm.py

Fast, LSTM-ONLY retrain. Reuses exactly the same master datasets and strict
point-in-time S&P-500 universe filter as retrain_models.py, but trains only the
LSTM (XGBoost is left as already deployed) with a larger batch and fewer epochs.

Why this is safe to shorten: on this data the LSTM's validation loss plateaus
after the first epoch (daily-return forecasting is near-random), so a short run
reaches the same model far faster than 50 epochs. A larger batch means far fewer
gradient steps per epoch, and a smaller early-stop patience ends the run as soon
as the (flat) validation loss stops improving.

Run (from the project root):
    caffeinate -i ./venv/bin/python scripts/retrain_lstm.py
    ./venv/bin/python scripts/retrain_lstm.py --epochs 8 --batch-size 256 --patience 3
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.data.fusion import FeatureFusion
from backend.infra.reproducibility import set_seed
from backend.prediction.lstm_model import LSTMForecaster
from backend.universe.universe_builder import UniverseBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("retrain_lstm")

FEATURES_DIR = Path("data/features")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50, help="LSTM training epochs max (early stopping ends it well before this)")
    ap.add_argument("--batch-size", type=int, default=256, help="larger batch = far fewer steps/epoch = faster")
    ap.add_argument("--patience", type=int, default=10, help="early-stop patience on validation loss")
    ap.add_argument("--limit", type=int, default=None, help="train on first N masters only (smoke test)")
    ap.add_argument(
        "--no-strict-universe",
        action="store_true",
        help="skip the strict point-in-time membership filter (NOT for the reported run)",
    )
    ap.add_argument("--start", default="2010-01-01", help="earliest master date to train on")
    ap.add_argument(
        "--end",
        default="2023-12-31",
        help="latest master date to train on (FNSPID news-window end; matches the report "
             "and the --end default of the evaluation scripts)",
    )
    args = ap.parse_args()

    set_seed(42)

    tickers = sorted(p.stem.replace("_master", "") for p in FEATURES_DIR.glob("*_master.csv"))
    if args.limit:
        tickers = tickers[: args.limit]
    if not tickers:
        logger.error("No master datasets in %s - run the fusion step first", FEATURES_DIR)
        sys.exit(1)
    logger.info("Loading %d master datasets...", len(tickers))

    combined = FeatureFusion().load_all(tickers)
    if combined is None or combined.empty:
        logger.error("No usable master data loaded")
        sys.exit(1)

    # Cap the training window to the news-covered period so the deployed model,
    # the evaluation scripts (which default to the same --end) and the report all
    # describe one 2010-2023 window. Training past the FNSPID news end would learn
    # on newsless rows the report does not describe.
    import pandas as _pd
    combined = combined.loc[
        (combined.index >= _pd.Timestamp(args.start)) & (combined.index <= _pd.Timestamp(args.end))
    ]
    if combined.empty:
        logger.error("No rows in [%s, %s]", args.start, args.end)
        sys.exit(1)
    logger.info("Date-capped to [%s, %s]: %d rows", args.start, args.end, len(combined))

    if not args.no_strict_universe:
        combined = UniverseBuilder().filter_eligible_rows(combined, strict=True)
    logger.info("Training on %d rows across %d tickers", len(combined), combined["ticker"].nunique())

    logger.info(
        "Training LSTM (epochs=%d, batch_size=%d, patience=%d)...",
        args.epochs, args.batch_size, args.patience,
    )
    res = LSTMForecaster().train(
        combined, epochs=args.epochs, batch_size=args.batch_size, patience=args.patience
    )
    logger.info(
        "LSTM done: %s",
        {k: res.get(k) for k in ("best_val_loss", "best_dir_acc", "best_epoch", "total_epochs")},
    )
    logger.info("Saved to models/lstm/ (model.pt, meta.pkl, training_history.json).")


if __name__ == "__main__":
    main()
