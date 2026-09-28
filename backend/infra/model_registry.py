"""Record model versions and select deployment candidates using validation scores."""

import hashlib
import json
import logging
import os
import platform
from datetime import datetime
from importlib import metadata as _md
from pathlib import Path

logger = logging.getLogger(__name__)
REGISTRY_PATH = Path("models/registry.json")


def _package_versions() -> dict:
    """Read installed library versions without importing numerical runtimes."""
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
    """Record model library versions, ordered feature schema and optional data version."""
    try:
        from backend.config.settings import FEATURE_PIPELINE_VERSION
    except Exception:
        FEATURE_PIPELINE_VERSION = None
    return {
        "package_versions": _package_versions(),
        "feature_schema_hash": feature_schema_hash(feature_names),
        "n_features": len(list(feature_names)),
        "feature_pipeline_version": FEATURE_PIPELINE_VERSION,
        "data_version": data_version,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }


def check_artifact_compatibility(fingerprint, feature_names):
    """Reject major-version or feature-schema mismatches and report minor-version changes."""
    issues = []
    fatal = False
    cur = _package_versions()
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
        # Create the models directory.
        REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._data = self._load()

    def register(
        self,
        model_type: str,
        metrics: dict,
        features: list,
        train_end_year: int | None = None,
        data_version: str | None = None,
        force_best: bool = False,
    ) -> int:
        """Add a new training run to the registry and return its version number."""
        self._data = self._load()
        records = self._data.setdefault(model_type, [])
        version = (records[-1]["version"] + 1) if records else 1
        existing = self._best(model_type)
        # Explicit regeneration can override the validation-based deployment decision.
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
            "package_versions": _package_versions(),
            "data_version": data_version,
        }
        records.append(entry)

        # Archive the previous best version.
        if status == "best":
            for previous in records[:-1]:
                if previous.get("status") in ("best", "deployed"):
                    previous["status"] = "archived"

        self._save()
        _auc = metrics.get("auc")
        _auc_str = f"{_auc:.4f}" if _auc is not None else "n/a"
        logger.info(f"Registry: {model_type} v{version} registered as {status}  AUC={_auc_str}")
        return version

    def should_deploy(self, model_type: str, new_metrics: dict) -> bool:
        """Compare a candidate's validation score with the best recorded score and tolerance."""
        self._data = self._load()
        best = self._best(model_type)
        if not best:
            return True
        # Preserve validated models when a candidate lacks a validation score.
        if new_metrics.get("cv_ran") is False or new_metrics.get("auc") is None:
            logger.info(f"Registry: {model_type} candidate has no walk-forward AUC - not deploying over existing best")
            return False
        best_auc = best.get("auc")
        if best_auc is None:
            return True  # a validated model beats an unvalidated incumbent
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
        # Use the highest recorded AUC as the deployment reference.
        records = self._data.get(model_type, [])
        if not records:
            return None
        # Unvalidated runs rank lowest.
        return max(records, key=lambda r: ((r.get("auc") if r.get("auc") is not None else -1.0), r.get("version", 0)))

    def _decide_status(self, existing: dict | None, new_metrics: dict) -> str:
        # Distinguish new best scores, deployments within tolerance and rejected candidates.
        if not existing:
            return "best"
        # Rank missing validation scores below measured scores.
        _raw_new = new_metrics.get("auc")
        _raw_best = existing.get("auc")
        new_auc = _raw_new if _raw_new is not None else -1.0
        best_auc = _raw_best if _raw_best is not None else -1.0
        if new_auc >= best_auc:
            return "best"
        return "deployed" if new_auc >= best_auc - 0.005 else "rejected"

    # Read the registry JSON, or an empty dict when absent.
    def _load(self) -> dict:
        if REGISTRY_PATH.exists():
            try:
                return json.loads(REGISTRY_PATH.read_text())
            except Exception as e:
                # Preserve a corrupt registry rather than discarding deployment history.
                backup = REGISTRY_PATH.with_suffix(".corrupt")
                try:
                    REGISTRY_PATH.replace(backup)
                    logger.error("Registry unreadable (%s); moved to %s and starting empty", e, backup)
                except Exception:
                    logger.error("Registry unreadable (%s) and could not be preserved", e)
                return {}
        return {}

    # Persist the registry through an atomic file replacement.
    def _save(self):
        tmp = REGISTRY_PATH.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(self._data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, REGISTRY_PATH)
