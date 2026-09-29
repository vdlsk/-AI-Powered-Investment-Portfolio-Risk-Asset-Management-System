"""
Model registry / persistence.

README §46-47: never save only the estimator. Every saved artifact bundles
the full inference pipeline (preprocessing + model) together with the
metadata needed to reproduce the experiment: dataset/feature version,
hyperparameters, random seed, and the exact train/validation/test date
ranges used.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parents[2] / "models" / "production"


@dataclass
class ExperimentRecord:
    experiment_id: str
    model_name: str
    feature_version: str
    hyperparameters: dict[str, Any]
    random_seed: int
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str
    metrics: dict[str, float] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


def _artifact_paths(experiment_id: str) -> tuple[Path, Path]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    return (
        MODELS_DIR / f"{experiment_id}.joblib",
        MODELS_DIR / f"{experiment_id}.json",
    )


def save_model(pipeline, record: ExperimentRecord) -> Path:
    """Persist the full sklearn Pipeline (preprocessing + model) and its
    ExperimentRecord side by side under the same experiment_id.
    """
    model_path, meta_path = _artifact_paths(record.experiment_id)
    joblib.dump(pipeline, model_path)
    with open(meta_path, "w") as f:
        json.dump(asdict(record), f, indent=2, default=str)
    logger.info("Saved model artifact to %s and metadata to %s", model_path, meta_path)
    return model_path


def load_model(experiment_id: str):
    """Return (pipeline, ExperimentRecord) for a previously saved experiment."""
    model_path, meta_path = _artifact_paths(experiment_id)
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"No saved model found for experiment_id='{experiment_id}' in {MODELS_DIR}"
        )
    pipeline = joblib.load(model_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return pipeline, ExperimentRecord(**meta)


def list_experiments() -> list[str]:
    if not MODELS_DIR.exists():
        return []
    return sorted(p.stem for p in MODELS_DIR.glob("*.json"))
