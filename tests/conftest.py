"""Redirect model artifact paths to a temporary directory for the test session."""

import tempfile
from pathlib import Path

from backend.infra import model_registry
from backend.prediction import lstm_model, xgboost_model

_TMP = Path(tempfile.mkdtemp(prefix="fab_test_models_"))

# Redirect module-level model paths.
xgboost_model.MODELS_DIR = _TMP / "xgboost"
xgboost_model.MODELS_DIR.mkdir(parents=True, exist_ok=True)

lstm_model.MODELS_DIR = _TMP / "lstm"
lstm_model.MODELS_DIR.mkdir(parents=True, exist_ok=True)

model_registry.REGISTRY_PATH = _TMP / "registry.json"
