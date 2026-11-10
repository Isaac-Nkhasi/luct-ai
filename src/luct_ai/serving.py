"""Serve the trained laptop price model (Phase 10).

The heavy lifting (feature engineering) is reused from :mod:`luct_ai.modeling`,
so a served prediction goes through exactly the same transformation as training.
Incoming text values are normalised with the same standardisers the cleaner uses,
so callers can send ``"i5"`` or ``"lenovo"`` and still hit the right category.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import joblib
import pandas as pd

from . import cleaning, modeling

MODEL_FILENAME = "laptop_price_model.joblib"
VERSION_FILENAME = "model_version.json"

#: Feature order the model was trained with.
FEATURE_COLUMNS = list(modeling.DEFAULT_FEATURES)

#: Text features are standardised the same way as in Phase 8.
_STANDARDISERS = {
    "brand": cleaning.standardise_brand,
    "processor": cleaning.standardise_processor,
    "storage_type": cleaning.standardise_storage_type,
    "condition": cleaning.standardise_condition,
}


def default_models_dir() -> Path:
    """``<repo>/models`` by default; override with ``LUCT_MODELS_DIR``."""
    override = os.environ.get("LUCT_MODELS_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "models"


@dataclass(frozen=True)
class ModelBundle:
    """The loaded estimator plus the manifest that describes it."""

    model: Any
    version: dict[str, Any]
    path: Path


def load_bundle(models_dir: str | Path | None = None) -> ModelBundle:
    """Load the joblib model and its version manifest from ``models_dir``."""
    directory = Path(models_dir) if models_dir is not None else default_models_dir()
    model_path = directory / MODEL_FILENAME
    if not model_path.exists():
        raise FileNotFoundError(
            f"No model at {model_path}. Train one first: "
            "python -m luct_ai.pipeline all --input <cleaned csv>"
        )
    version_path = directory / VERSION_FILENAME
    version: dict[str, Any] = {}
    if version_path.exists():
        version = json.loads(version_path.read_text(encoding="utf-8"))
    # ``save_model_bundle`` writes ``{"model": <estimator>, "name": <str>}``.
    payload = joblib.load(model_path)
    estimator = payload["model"] if isinstance(payload, dict) else payload
    return ModelBundle(model=estimator, version=version, path=model_path)


def build_feature_row(payload: Mapping[str, Any]) -> pd.DataFrame:
    """Turn one API payload into the single-row frame the model expects."""
    row: dict[str, Any] = {column: payload.get(column) for column in FEATURE_COLUMNS}
    for column, standardise in _STANDARDISERS.items():
        row[column] = standardise(payload.get(column)) or None
    frame = pd.DataFrame([row], columns=FEATURE_COLUMNS)
    return modeling.feature_frame(frame)


def predict_price(bundle: ModelBundle, payload: Mapping[str, Any]) -> float:
    """Predict an asking price in LSL. Never returns a negative value."""
    features = build_feature_row(payload)
    prediction = float(bundle.model.predict(features)[0])
    return max(prediction, 0.0)
