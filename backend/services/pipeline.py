"""Data and training orchestration with explicit service dependencies."""

import logging

import pandas as pd

from backend.api.schemas import RunRequest
from backend.config import settings

logger = logging.getLogger(__name__)


def run_training_pipeline(
    req: RunRequest,
    *,
    universe_builder,
    downloader,
    cleaner,
    engineer,
    news_collector,
    sentiment,
    fusion,
    xgb_model,
    lstm_model,
    model_lock,
    update_pipeline,
    _stage_progress,
    user_store,
    queue_recommendations,
    drift_monitor,
):
    """Prepare historical-member data, train models and refresh current-member recommendations."""
    from backend.infra.reproducibility import set_seed

    set_seed(42)
    try:
        already_trained = xgb_model.is_trained() and lstm_model.is_trained()

        update_pipeline("fetching", "Resolving point-in-time S&P 500 universe", 3)
        _today = pd.Timestamp.today().strftime("%Y-%m-%d")
        pit_available = True
        try:
            all_tickers = universe_builder.eligible_between(settings.HISTORY_START, _today)
        except Exception as e:
            # Allow a labelled current-member fallback only in the live pipeline.
            pit_available = False
            logger.warning(
                f"Point-in-time universe unavailable ({e}); live app falling back to "
                "CURRENT members (survivorship-biased - not for evaluation)"
            )
            all_tickers = universe_builder.members()
        tickers = all_tickers[:10] if req.scope == "sample" else all_tickers
        logger.info(
            f"Scope: {req.scope} - point-in-time universe "
            f"{settings.HISTORY_START}..{_today}: training on {len(tickers)} of {len(all_tickers)} tickers"
        )

        if already_trained:
            update_pipeline("downloading", f"Updating {len(tickers)} stocks with latest prices", 10)
            dl_r = downloader.download_universe_incremental(
                tickers, progress_cb=_stage_progress("downloading", "Updating prices", 10, 22)
            )
        else:
            update_pipeline("downloading", f"Downloading full history for {len(tickers)} stocks", 10)
            dl_r = downloader.download_universe_incremental(
                tickers, progress_cb=_stage_progress("downloading", "Downloading history", 10, 22)
            )

        downloaded = dl_r.get("success", []) + dl_r.get("skipped", [])

        update_pipeline("cleaning", "Cleaning and validating OHLCV data", 22)
        cl_r = cleaner.clean_universe(downloaded, progress_cb=_stage_progress("cleaning", "Cleaning OHLCV", 22, 34))
        cleaned = cl_r.get("success", downloaded)

        update_pipeline("features", "Building technical indicators", 34)
        fe_r = engineer.generate_universe(
            cleaned, progress_cb=_stage_progress("features", "Building indicators", 34, 48)
        )
        featured = fe_r.get("success", cleaned)

        update_pipeline("news", "Collecting news headlines", 48)
        news_collector.get_universe_news(featured, progress_cb=_stage_progress("news", "Collecting news", 48, 60))

        update_pipeline("sentiment", "Running FinBERT sentiment analysis", 60)
        sent_r = sentiment.analyze_universe(
            featured, progress_cb=_stage_progress("sentiment", "FinBERT sentiment", 60, 72)
        )
        # Exclude failed sentiment runs from fusion to avoid reusing stale scores.
        sent_failed = set(sent_r.get("failed", [])) if isinstance(sent_r, dict) else set()
        fuse_inputs = [t for t in featured if t not in sent_failed]
        if sent_failed:
            logger.warning(
                "Excluding %d ticker(s) with failed sentiment from fusion: %s",
                len(sent_failed),
                sorted(sent_failed),
            )

        update_pipeline("fusion", "Merging price features and sentiment scores", 72)
        fused_r = fusion.fuse_universe(fuse_inputs, progress_cb=_stage_progress("fusion", "Merging features", 72, 80))
        fused = fused_r.get("success", featured)

        combined = fusion.load_all(fused)
        if combined is None or (hasattr(combined, "empty") and combined.empty):
            update_pipeline("error", "No fused data available - check logs", error="no_data")
            return

        # Filter training rows by historical membership when membership records are available.
        combined = universe_builder.filter_eligible_rows(combined, strict=pit_available)

        if already_trained:
            update_pipeline("xgboost", "Retraining XGBoost on the latest data (leakage-safe walk-forward)", 80)
            with model_lock:
                xgb_result = xgb_model.update(combined)
            if xgb_result.get("mode") == "skipped":
                logger.warning(f"XGBoost update skipped: {xgb_result.get('reason')}")

            update_pipeline("lstm", "Low-learning-rate LSTM re-fit over full history (10 epochs)", 90)
            with model_lock:
                lstm_result = lstm_model.update(combined, epochs=10)
            if lstm_result.get("mode") == "skipped":
                logger.warning(f"LSTM update skipped: {lstm_result.get('reason')}")
        else:
            update_pipeline("xgboost", "Training XGBoost from scratch (walk-forward CV + early stopping)", 80)
            with model_lock:
                xgb_model.train(combined)

            update_pipeline("lstm", "Training LSTM from scratch (50 epochs, early stopping)", 90)

            # Publish epoch progress during LSTM training.
            def _lstm_progress(epoch, total, val_loss, dir_acc):
                pct = 90 + int(9 * epoch / max(1, total))
                update_pipeline(
                    "lstm", f"Training LSTM - epoch {epoch}/{total} (val_loss={val_loss:.4f})", min(99, pct)
                )

            with model_lock:
                lstm_model.train(combined, epochs=50, batch_size=256, progress_cb=_lstm_progress)

        update_pipeline("recommendations", "Preparing recommendations", 99, tickers=fused)

        if req.user_id and user_store.get(req.user_id):
            # Use current membership for live recommendations.
            current_members = set(universe_builder.members())
            live_universe = [t for t in fused if t in current_members] or fused
            queue_recommendations(req.user_id, live_universe, req.risk_profile, req.budget, pipeline_internal=True)
        drift_monitor.evaluate_past_recommendations({t: fusion.load_master(t) for t in fused})
        update_pipeline("done", "Pipeline complete", 100, tickers=fused)

    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        update_pipeline("error", str(e), error=str(e))
