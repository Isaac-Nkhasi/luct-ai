"""Phase 9 — train and compare the laptop asking-price models.

Three approaches are compared on the same grouped split:

1. ``median_baseline`` — predicts the training median (a benchmark).
2. ``linear_regression`` — a simple, interpretable model.
3. ``random_forest`` — a model that can capture non-linear relationships.

The test split is only ever touched for the final comparison. Models are tuned
inside the training data with grouped cross-validation, then refit on
train+validation before scoring on the held-out test set.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import evaluate
from .schema import GROUP_KEY_COLUMNS

TARGET_COLUMN = "price_lsl"

#: Laptop-identity columns used to keep groups together across splits/folds.
GROUP_COLUMNS = list(GROUP_KEY_COLUMNS)

NUMERIC_FEATURES = ["ram_gb", "storage_gb", "age_years", "battery_health"]
CATEGORICAL_FEATURES = ["brand", "processor", "storage_type", "condition"]

#: ``model`` is deliberately excluded by default: it is very high-cardinality and
#: would need target encoding to be used safely. Pass ``include_model=True`` to
#: experiment, but watch for leakage.
DEFAULT_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# Feature frame
# --------------------------------------------------------------------------- #


def feature_frame(frame: pd.DataFrame, *, include_model: bool = False) -> pd.DataFrame:
    """Select and order the model input columns from a cleaned dataset."""
    columns = list(DEFAULT_FEATURES)
    if include_model and "model" in frame.columns:
        columns.append("model")
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"Cleaned dataset is missing feature column(s): {', '.join(missing)}")
    out = frame[columns].copy()
    # The cleaned CSV may be read back as pandas ``string`` dtype, which turns
    # missing values into ``pd.NA``. sklearn's imputers cannot handle ``pd.NA``
    # (it raises "boolean value of NA is ambiguous"), so convert the numeric
    # features to floats and the categorical features to object with ``np.nan``.
    for column in NUMERIC_FEATURES:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    for column in columns:
        if column in NUMERIC_FEATURES:
            continue
        out[column] = out[column].astype("object").fillna(np.nan)
    return out


def usable_mask(frame: pd.DataFrame, target: str = TARGET_COLUMN) -> pd.Series:
    """Rows that carry a finite target and therefore can be trained/scored."""
    if target not in frame.columns:
        raise ValueError(f"Cleaned dataset has no '{target}' column; run Phase 8 first.")
    return pd.to_numeric(frame[target], errors="coerce").notna()


# --------------------------------------------------------------------------- #
# Pipelines
# --------------------------------------------------------------------------- #


def build_preprocessor(
    numeric: Sequence[str] = NUMERIC_FEATURES,
    categorical: Sequence[str] = CATEGORICAL_FEATURES,
) -> ColumnTransformer:
    """Median-imputation + scaling for numbers; mode-imputation + one-hot for categories."""
    numeric_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("num", numeric_pipe, list(numeric)), ("cat", categorical_pipe, list(categorical))]
    )


def build_estimators(random_state: int = RANDOM_STATE) -> dict[str, Pipeline]:
    """Return the three untuned estimator pipelines, keyed by model name."""
    pre = build_preprocessor()
    return {
        "median_baseline": Pipeline(
            [("pre", pre), ("model", DummyRegressor(strategy="median"))]
        ),
        "linear_regression": Pipeline(
            [("pre", pre), ("model", LinearRegression())]
        ),
        "random_forest": Pipeline(
            [("pre", pre),
             # n_jobs=1: the dataset is small, and joblib worker processes are
             # expensive to spawn (and slow to import) on some machines.
             ("model", RandomForestRegressor(random_state=random_state, n_jobs=1))]
        ),
    }


def param_grids() -> dict[str, dict]:
    """Hyper-parameter grids. The baseline has nothing meaningful to tune."""
    return {
        "median_baseline": {},
        "linear_regression": {"model__fit_intercept": [True, False]},
        "random_forest": {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 8, 16, 24],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": [1.0, "sqrt"],
        },
    }


def _group_keys(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    present = [c for c in columns if c in frame.columns]
    if not present:
        return pd.Series(frame.index.astype(str), index=frame.index)
    return frame[present].astype("string").fillna("NA").agg("|".join, axis=1)


def group_keys(frame: pd.DataFrame, columns: Sequence[str] = GROUP_COLUMNS) -> pd.Series:
    """Public helper: the laptop-identity key used for leakage-free grouping."""
    return _group_keys(frame, columns)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #


@dataclass
class TrainResult:
    models: dict[str, Pipeline]
    val_predictions: dict[str, np.ndarray]
    test_predictions: dict[str, np.ndarray]
    test_truth: np.ndarray
    test_bands: pd.Series
    comparison_val: pd.DataFrame
    comparison_test: pd.DataFrame
    feature_importances: pd.Series | None
    best_model_name: str
    metadata: dict = field(default_factory=dict)

    @property
    def best_model(self) -> Pipeline:
        return self.models[self.best_model_name]


def _tune(
    estimator: Pipeline,
    grid: Mapping,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    random_state: int,
) -> Pipeline:
    """Tune one estimator on the training partition with grouped CV.

    Falls back to an untuned fit when there are not enough groups, so small
    samples still produce a usable model instead of an exception.
    """
    n_groups = int(groups.nunique())
    if not grid or n_groups < 2:
        estimator.fit(X, y)
        return estimator
    cv = GroupKFold(n_splits=min(4, n_groups))
    # n_jobs=1 keeps behaviour deterministic across platforms (the grid is small).
    search = GridSearchCV(
        estimator, grid, scoring="neg_mean_absolute_error", cv=cv, n_jobs=1, refit=True
    )
    search.fit(X, y, groups=groups)
    return search.best_estimator_


def train_models(
    cleaned: pd.DataFrame,
    *,
    target: str = TARGET_COLUMN,
    include_model: bool = False,
    random_state: int = RANDOM_STATE,
    tune: bool = True,
) -> TrainResult:
    """Tune, refit and compare the three models on the grouped split.

    ``cleaned`` must carry the ``split`` column produced by Phase 8.
    """
    if "split" not in cleaned.columns:
        raise ValueError("Dataset has no 'split' column; run Phase 8 cleaning first.")

    usable = cleaned[usable_mask(cleaned, target)].copy()
    X = feature_frame(usable, include_model=include_model)
    y = pd.to_numeric(usable[target], errors="coerce")
    groups = _group_keys(usable, ["brand", "model", "processor", "ram_gb", "storage_gb"])

    split = usable["split"].astype("string")
    train_idx = split == "train"
    val_idx = split == "val"
    test_idx = split == "test"

    references = build_estimators(random_state)
    grids = param_grids() if tune else {name: {} for name in references}

    models: dict[str, Pipeline] = {}
    for name, estimator in references.items():
        models[name] = _tune(
            estimator, grids.get(name, {}),
            X[train_idx], y[train_idx], groups[train_idx], random_state,
        )

    def _predict_all(frame_idx: pd.Series) -> dict[str, np.ndarray]:
        return {name: model.predict(X[frame_idx]) for name, model in models.items()}

    val_predictions = _predict_all(val_idx) if val_idx.any() else {}
    test_predictions = _predict_all(test_idx) if test_idx.any() else {}

    comparison_val = (
        evaluate.comparison_table({n: (y[val_idx], p) for n, p in val_predictions.items()})
        if val_predictions else pd.DataFrame(columns=["model", "n", "mae", "rmse", "r2"])
    )
    comparison_test = (
        evaluate.comparison_table({n: (y[test_idx], p) for n, p in test_predictions.items()})
        if test_predictions else pd.DataFrame(columns=["model", "n", "mae", "rmse", "r2"])
    )

    # Pick the best model on validation MAE (falling back to training MAE).
    if not comparison_val.empty:
        best_model_name = str(comparison_val.sort_values("mae").iloc[0]["model"])
    else:
        scores = {
            name: evaluate.regression_metrics(y[train_idx], model.predict(X[train_idx]))["mae"]
            for name, model in models.items()
        }
        best_model_name = min(scores, key=lambda k: scores[k] if np.isfinite(scores[k]) else np.inf)

    feature_importances = _feature_importances(models.get("random_forest"), X)

    metadata = {
        "trained_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "best_model": best_model_name,
        "target": target,
        "features": list(X.columns),
        "include_model": include_model,
        "n_train": int(train_idx.sum()),
        "n_val": int(val_idx.sum()),
        "n_test": int(test_idx.sum()),
        "random_state": random_state,
        "sklearn_version": sklearn.__version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "validation_metrics": comparison_val.to_dict(orient="records"),
        "test_metrics": comparison_test.to_dict(orient="records"),
    }

    return TrainResult(
        models=models,
        val_predictions=val_predictions,
        test_predictions=test_predictions,
        test_truth=np.asarray(y[test_idx], dtype="float64"),
        test_bands=usable.loc[test_idx, "price_band"] if "price_band" in usable.columns else pd.Series(dtype=object),
        comparison_val=comparison_val,
        comparison_test=comparison_test,
        feature_importances=feature_importances,
        best_model_name=best_model_name,
        metadata=metadata,
    )


def _feature_importances(model: Pipeline | None, X: pd.DataFrame) -> pd.Series | None:
    if model is None:
        return None
    try:
        importances = model.named_steps["model"].feature_importances_
        names = model.named_steps["pre"].get_feature_names_out()
    except (AttributeError, KeyError):
        return None
    return pd.Series(importances, index=[str(n) for n in names]).sort_values(ascending=False)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def dataset_fingerprint(frame: pd.DataFrame) -> str:
    """Stable SHA-256 over the cleaned dataset, used to version a model."""
    hasher = hashlib.sha256()
    hasher.update("|".join(map(str, frame.columns)).encode())
    for row in frame.astype("string").fillna("").itertuples(index=False):
        hasher.update("|".join(row).encode())
    return hasher.hexdigest()


def save_model_bundle(
    result: TrainResult,
    output_dir: str | Path,
    *,
    cleaned: pd.DataFrame | None = None,
    extra_metadata: Mapping | None = None,
) -> dict[str, Path]:
    """Persist the best model plus a JSON version/manifest file.

    Returns the paths written: ``{"model": ..., "version": ...}``.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "laptop_price_model.joblib"
    version_path = output_dir / "model_version.json"

    joblib.dump({"model": result.best_model, "name": result.best_model_name}, model_path)

    manifest = dict(result.metadata)
    manifest["model_file"] = model_path.name
    manifest["package_version"] = "0.1.0"
    if cleaned is not None:
        manifest["dataset_fingerprint"] = dataset_fingerprint(cleaned)
        manifest["dataset_rows"] = int(len(cleaned))
    if extra_metadata:
        manifest.update(dict(extra_metadata))
    version_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    return {"model": model_path, "version": version_path}


def load_model_bundle(path: str | Path) -> tuple[Pipeline, str]:
    """Load a saved model bundle; returns ``(pipeline, model_name)``."""
    payload = joblib.load(Path(path))
    return payload["model"], payload["name"]


def predict_one(model: Pipeline, record: Mapping[str, object], *, include_model: bool = False) -> float:
    """Predict a single asking price from a mapping of raw feature values.

    This is the same code path the Phase 10 API will call.
    """
    columns = list(DEFAULT_FEATURES) + (["model"] if include_model else [])
    frame = pd.DataFrame([{column: record.get(column) for column in columns}])
    return float(model.predict(frame)[0])
