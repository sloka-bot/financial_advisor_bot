"""
model_registry.py

Tracks every trained version of each model and controls which version is
deployed for inference. Only promotes a new model if it is not meaningfully
worse than the current best on walk-forward AUC.
"""

import hashlib
import json
import os
import logging
import platform
from datetime import datetime
from importlib import metadata as _md
from pathlib import Path

logger = logging.getLogger(__name__)
REGISTRY_PATH = Path("models/registry.json")



def _pkg_versions() -> dict:
    """Record the library versions used to train an artifact, without importing
    the heavy libraries themselves."""
    versions = {"python": platform.python_version()}
    for name in ("scikit-learn", "xgboost", "torch", "pandas", "numpy", "scipy"):
        try:
            versions[name] = _md.version(name)
        except Exception:
            versions[name] = None
    return versions


CRITICAL_PKGS = ("numpy", "scipy", "scikit-learn", "xgboost", "torch")


def feature_schema_hash(feature_names) -> str:
    """Stable 16-hex-char hash of the ordered feature list."""
    return hashlib.sha256("|".join(map(str, feature_names)).encode()).hexdigest()[:16]


def artifact_fingerprint(feature_names, data_version=None) -> dict:
    """Compatibility fingerprint written next to a saved model artifact: the
    library versions it was trained under, its ordered-feature schema hash, and
    an optional training-data version. Verified on load so an artifact pickled
    under an incompatible environment or a different feature schema is rejected
    rather than used silently."""
    try:
        from backend.config.settings import FEATURE_PIPELINE_VERSION
    except Exception:
        FEATURE_PIPELINE_VERSION = None
    return {
        "package_versions": _pkg_versions(),
        "feature_schema_hash": feature_schema_hash(feature_names),
        "n_features": len(list(feature_names)),
        "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
        "data_version": data_version,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }


def check_artifact_compatibility(fingerprint, feature_names):
    """Return (ok, issues). An artifact is rejected (ok=False) on a MAJOR-version
    change of a critical numerical/ML library or a feature-schema hash mismatch,
    because pickled estimators and tensors are not portable across those. A minor
    or patch version change is reported as a non-fatal warning."""
    issues = []
    fatal = False
    cur = _pkg_versions()
    saved = (fingerprint or {}).get("package_versions", {}) or {}
    for pkg in CRITICAL_PKGS:
        s, c = saved.get(pkg), cur.get(pkg)
        if s and c and s != c:
            if str(s).split(".")[0] != str(c).split(".")[0]:
                issues.append(f"{pkg} major version changed {s} -> {c}")
                fatal = True
            else:
                issues.append(f"{pkg} version changed {s} -> {c}")
    saved_hash = (fingerprint or {}).get("feature_schema_hash")
    if saved_hash and saved_hash != feature_schema_hash(feature_names):
        issues.append("feature-schema hash does not match current features")
        fatal = True
    try:
        from backend.config.settings import FEATURE_PIPELINE_VERSION as _fpv
    except Exception:
        _fpv = None
    saved_fpv = (fingerprint or {}).get("feature_pipeline_version")
    if saved_fpv is not None and _fpv is not None and saved_fpv != _fpv:
        issues.append(f"feature-pipeline version changed {saved_fpv} -> {_fpv}")
        fatal = True
    return (not fatal), issues


class ModelRegistry:
    """Persist model versions, validation scores and deployment decisions."""

    def __init__(self):
        # Ensure the models directory exists before trying to write the registry
        REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load()

    # Write a new training run to the registry and decide its status
    def register(self, model_type: str, metrics: dict, features: list, train_end_year: int | None = None, data_version: str | None = None, force_best: bool = False) -> int:
        """Add a new training run to the registry and return its version number."""
        self._data = self._load()
        records = self._data.setdefault(model_type, [])
        version = (records[-1]["version"] + 1) if records else 1
        existing = self._best(model_type)
        # force_best: a canonical regeneration run (retrain_models.py) deploys the
        # freshly trained full-corpus model as best regardless of a noisy AUC gate.
        status = "best" if force_best else self._decide_status(existing, metrics)

        entry = {
            "version": version,
            "trained_at": datetime.now().isoformat(),
            "train_end_year": train_end_year or datetime.now().year,
            "status": status,
            **metrics,
            "n_features": len(features),
            "feature_names": list(features),
            "feature_schema_hash": feature_schema_hash(features),
            "feature_preview": list(features)[:10],
            "package_versions": _pkg_versions(),
            "data_version": data_version,
        }
        records.append(entry)

        # Archive the previous best so the history is preserved
        if status == "best":
            for previous in records[:-1]:
                if previous.get("status") in ("best", "deployed"):
                    previous["status"] = "archived"

        self._save()
        _auc = metrics.get("auc")
        _auc_str = f"{_auc:.4f}" if _auc is not None else "n/a"
        logger.info(f"Registry: {model_type} v{version} registered as {status}  AUC={_auc_str}")
        return version

    # Return True unless the new model is meaningfully worse than the current best
    def should_deploy(self, model_type: str, new_metrics: dict) -> bool:
        """Deploy the new model for inference unless it is meaningfully worse than
        the BEST-EVER model (not merely the last-deployed one). Comparing to the
        best-ever AUC stops the baseline ratcheting downward, where each retrain
        within 0.005 of the last could quietly lower the bar for the next."""
        self._data = self._load()
        best = self._best(model_type)
        if not best:
            return True
        # A candidate with no validated walk-forward estimate must not replace a
        # model that has one.
        if new_metrics.get("cv_ran") is False or new_metrics.get("auc") is None:
            logger.info(f"Registry: {model_type} candidate has no walk-forward AUC - not deploying over existing best")
            return False
        best_auc = best.get("auc")
        if best_auc is None:
            return True  # any validated model beats an unvalidated incumbent
        new_auc = new_metrics.get("auc", 0.5)
        deploy = new_auc >= best_auc - 0.005
        logger.info(f"Registry: {model_type} deploy={deploy}  new_auc={new_auc}  best_ever_auc={best_auc}")
        return deploy

    def get_best(self, model_type: str) -> dict | None:
        """Reload the registry and return the selected version for a model family."""
        self._data = self._load()
        return self._best(model_type)

    def list_all(self, model_type: str | None = None) -> dict:
        """Reload and return all recorded model versions."""
        self._data = self._load()
        if model_type:
            return {model_type: self._data.get(model_type, [])}
        return dict(self._data)

    def _best(self, model_type: str) -> dict | None:
        # The best-ever record BY AUC, not the most recently deployed one, so the
        # promotion baseline can never drift downward across retrains.
        records = self._data.get(model_type, [])
        if not records:
            return None
        # None AUCs (unvalidated runs) rank lowest, so a real model is always preferred
        return max(records, key=lambda r: ((r.get("auc") if r.get("auc") is not None else -1.0), r.get("version", 0)))

    def _decide_status(self, existing: dict | None, new_metrics: dict) -> str:
        # 'best' only when at least as good as the best-ever (so the recorded best
        # is always the true maximum); 'deployed' when within tolerance and serving
        # inference; 'rejected' otherwise.
        if not existing:
            return "best"
        # None AUC = unvalidated run; rank it lowest so a validated model always wins
        # and comparisons never touch None.
        _raw_new = new_metrics.get("auc")
        _raw_best = existing.get("auc")
        new_auc = _raw_new if _raw_new is not None else -1.0
        best_auc = _raw_best if _raw_best is not None else -1.0
        if new_auc >= best_auc:
            return "best"
        return "deployed" if new_auc >= best_auc - 0.005 else "rejected"

    # Read the registry JSON or return an empty dict if not yet created
    def _load(self) -> dict:
        if REGISTRY_PATH.exists():
            try:
                return json.loads(REGISTRY_PATH.read_text())
            except Exception as e:
                # Preserve a corrupt registry instead of silently discarding it, so a
                # parse failure never wipes the deployment history.
                backup = REGISTRY_PATH.with_suffix(".corrupt")
                try:
                    REGISTRY_PATH.replace(backup)
                    logger.error("Registry unreadable (%s); moved to %s and starting empty", e, backup)
                except Exception:
                    logger.error("Registry unreadable (%s) and could not be preserved", e)
                return {}
        return {}

    # Write the registry back to disk after every change, atomically (temp file +
    # fsync + rename) so a crash mid-write cannot corrupt the registry.
    def _save(self):
        tmp = REGISTRY_PATH.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(self._data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, REGISTRY_PATH)
