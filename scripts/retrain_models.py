"""Retrain the deployed XGBoost and LSTM models on the existing master datasets."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.data.fusion import FeatureFusion
from backend.infra.reproducibility import set_seed
from backend.prediction.lstm_model import LSTMForecaster
from backend.prediction.xgboost_model import XGBoostForecaster
from backend.universe.universe_builder import UniverseBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("retrain")

FEATURES_DIR = Path("data/features")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="train on first N masters only (smoke test)")
    ap.add_argument("--epochs", type=int, default=50, help="LSTM training epochs")
    ap.add_argument(
        "--no-strict-universe",
        action="store_true",
        help="skip the strict point-in-time membership filter (NOT for the reported run)",
    )
    ap.add_argument(
        "--respect-gate",
        action="store_true",
        help="on a full run, obey the registry AUC deploy-gate instead of force-deploying "
        "the freshly trained full model (default: force-deploy the canonical model)",
    )
    ap.add_argument("--start", default="2010-01-01", help="earliest master date to train on")
    ap.add_argument(
        "--end",
        default="2023-12-31",
        help="latest master date to train on (FNSPID news-window end; matches the report "
        "and the --end default of the evaluation scripts)",
    )
    ap.add_argument(
        "--skip-lstm",
        action="store_true",
        help="train/deploy XGBoost only; run scripts/retrain_lstm.py for a fast LSTM",
    )
    ap.add_argument(
        "--lstm-batch-size", type=int, default=256, help="LSTM batch size (larger = fewer steps per epoch = faster)"
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

    # Cap training at the end of the news-covered 2010-2023 window.
    import pandas as _pd

    combined = combined.loc[(combined.index >= _pd.Timestamp(args.start)) & (combined.index <= _pd.Timestamp(args.end))]
    if combined.empty:
        logger.error("No rows in [%s, %s]", args.start, args.end)
        sys.exit(1)
    logger.info("Date-capped to [%s, %s]: %d rows", args.start, args.end, len(combined))

    # Filter rows by point-in-time membership.
    if not args.no_strict_universe:
        combined = UniverseBuilder().filter_eligible_rows(combined, strict=True)
    logger.info("Training on %d rows across %d tickers", len(combined), combined["ticker"].nunique())

    smoke = bool(args.limit)
    if smoke:
        logger.warning(
            "SMOKE/DRY RUN (--limit %d): training for a sanity check only - the production "
            "registry and deployed model artifacts are NOT written.",
            args.limit,
        )
    logger.info("Training XGBoost...")
    xgb_res = XGBoostForecaster().train(combined, persist=not smoke, force_deploy=(not smoke and not args.respect_gate))
    logger.info(
        "XGBoost: cv_auc_mean=%s cv_direction_acc=%s deployed=%s",
        xgb_res.get("cv_auc_mean"),
        xgb_res.get("cv_direction_acc"),
        xgb_res.get("deployed"),
    )

    if args.skip_lstm:
        logger.info("Skipping LSTM (--skip-lstm); run scripts/retrain_lstm.py for the sequence model.")
    else:
        logger.info("Training LSTM (%d epochs, batch=%d)...", args.epochs, args.lstm_batch_size)
        lstm_res = LSTMForecaster().train(
            combined, epochs=args.epochs, batch_size=args.lstm_batch_size, persist=not smoke
        )
        logger.info("LSTM: %s", {k: lstm_res.get(k) for k in ("best_val_loss", "best_dir_acc", "val_mae")})

    logger.info("Done. Artifacts written under models/ (model files, manifest, registry).")


if __name__ == "__main__":
    main()
