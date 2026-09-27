"""
conftest.py

Test isolation for model artefacts.

The model classes write to fixed relative paths - XGBoost to models/xgboost/,
LSTM to models/lstm/, and the registry to models/registry.json. Running the
test suite from the project root therefore RETRAINED and OVERWROTE the deployed
production model with a tiny synthetic one (the 200-row / AUC-0.5 artefact seen
in models/xgboost/training_results.json).

This conftest redirects all three locations to a throwaway temp directory for
the whole test session, before any test imports/instantiates the classes, so
tests can never touch the real trained models or registry again.
"""

import tempfile
from pathlib import Path

from backend.infra import model_registry
from backend.prediction import lstm_model, xgboost_model

_TMP = Path(tempfile.mkdtemp(prefix="fab_test_models_"))

# redirect the module-level path globals the classes read at call time
xgboost_model.MODELS_DIR = _TMP / "xgboost"
xgboost_model.MODELS_DIR.mkdir(parents=True, exist_ok=True)

lstm_model.MODELS_DIR = _TMP / "lstm"
lstm_model.MODELS_DIR.mkdir(parents=True, exist_ok=True)

model_registry.REGISTRY_PATH = _TMP / "registry.json"
