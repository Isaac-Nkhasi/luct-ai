"""Unit tests for the Phase 9 evaluation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from luct_ai import evaluate


def test_regression_metrics_perfect_prediction():
    y = [1000.0, 2000.0, 3000.0]
    metrics = evaluate.regression_metrics(y, y)
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0
    assert metrics["n"] == 3


def test_regression_metrics_known_error():
    y_true = [1000.0, 2000.0]
    y_pred = [1100.0, 1900.0]
    metrics = evaluate.regression_metrics(y_true, y_pred)
    assert metrics["mae"] == 100.0
    assert metrics["rmse"] == 100.0


def test_regression_metrics_ignores_non_finite():
    metrics = evaluate.regression_metrics([1000.0, np.nan], [1000.0, 500.0])
    assert metrics["n"] == 1
    assert metrics["mae"] == 0.0


def test_comparison_table_shape():
    predictions = {
        "a": ([1000.0, 2000.0, 3000.0], [1100.0, 1900.0, 3050.0]),
        "b": ([1000.0, 2000.0, 3000.0], [1000.0, 2000.0, 3000.0]),
    }
    table = evaluate.comparison_table(predictions)
    assert list(table["model"]) == ["a", "b"]
    # The perfect model must win on MAE.
    assert table.sort_values("mae").iloc[0]["model"] == "b"


def test_error_by_band_groups_and_sorts():
    y_true = [1000, 1100, 5000, 5200]
    y_pred = [900, 1000, 5000, 5000]
    bands = ["low", "low", "high", "high"]
    table = evaluate.error_by_band(y_true, y_pred, bands)
    assert set(table["band"]) == {"low", "high"}
    assert table.set_index("band").loc["low", "n"] == 2


def test_learning_curve_returns_empty_on_tiny_data():
    from sklearn.linear_model import LinearRegression

    X = pd.DataFrame({"ram_gb": [4, 8]})
    curve = evaluate.learning_curve_data(LinearRegression(), X, [1000, 2000])
    assert curve.is_empty is True
