"""Unit tests for the Phase 9 modelling helpers.

These tests only prove the code runs and behaves; they do **not** produce any
model performance claim. Real metrics come from the collected dataset.
"""

from __future__ import annotations

import numpy as np
import pytest

from luct_ai import modeling


def test_build_estimators_returns_three_models():
    estimators = modeling.build_estimators()
    assert set(estimators) == {"median_baseline", "linear_regression", "random_forest"}


def test_feature_frame_rejects_missing_columns():
    import pandas as pd

    with pytest.raises(ValueError):
        modeling.feature_frame(pd.DataFrame({"brand": ["Dell"]}))


def test_train_models_compares_all_three(cleaned_frame):
    result = modeling.train_models(cleaned_frame, tune=False)
    assert set(result.models) == {"median_baseline", "linear_regression", "random_forest"}
    assert result.best_model_name in result.models
    # A test split exists, so a comparison table should have three rows.
    assert set(result.comparison_test["model"]) == set(result.models)
    for column in ("mae", "rmse"):
        assert np.isfinite(result.comparison_test[column]).all()


def test_train_models_predicts_reasonable_values(cleaned_frame):
    result = modeling.train_models(cleaned_frame, tune=False)
    predictions = result.test_predictions[result.best_model_name]
    assert predictions.shape[0] == result.test_truth.shape[0]
    # Prices are positive; the median baseline guarantees positive predictions.
    assert (predictions > 0).all()


def test_save_and_load_bundle_roundtrip(tmp_path, cleaned_frame):
    result = modeling.train_models(cleaned_frame, tune=False)
    paths = modeling.save_model_bundle(result, tmp_path)
    assert paths["model"].exists()
    assert paths["version"].exists()

    model, name = modeling.load_model_bundle(paths["model"])
    assert name == result.best_model_name

    record = {
        "brand": "Lenovo", "processor": "Intel Core i5", "ram_gb": 8,
        "storage_gb": 256, "storage_type": "SSD", "age_years": 5,
        "condition": "Good", "battery_health": 80,
    }
    prediction = modeling.predict_one(model, record)
    assert np.isfinite(prediction)
    assert prediction > 0


def test_dataset_fingerprint_is_stable(cleaned_frame):
    assert modeling.dataset_fingerprint(cleaned_frame) == modeling.dataset_fingerprint(cleaned_frame)


def test_tuning_runs(cleaned_frame):
    """Tuning must not raise; it may fall back to an untuned fit on tiny data."""
    result = modeling.train_models(cleaned_frame, tune=True, random_state=0)
    assert result.best_model_name in result.models
