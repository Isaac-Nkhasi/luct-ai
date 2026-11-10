"""Tests for the Phase 10 prediction service (``api/main.py``)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app
from luct_ai import serving

VALID = {
    "brand": "Lenovo",
    "processor": "Intel Core i5",
    "ram_gb": 8,
    "storage_gb": 256,
    "storage_type": "SSD",
    "condition": "Good",
    "age_years": 5,
    "battery_health": 80,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:  # `with` runs the startup lifespan
        yield test_client


def test_health_reports_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["error"] is None


def test_predict_returns_positive_price(client):
    response = client.post("/predict", json=VALID)
    assert response.status_code == 200
    price = response.json()["predicted_price_lsl"]
    assert isinstance(price, (int, float))
    assert price > 0


def test_predict_matches_the_model_directly(client):
    expected = serving.predict_price(serving.load_bundle(), VALID)
    response = client.post("/predict", json=VALID)
    assert response.json()["predicted_price_lsl"] == round(expected, 2)


def test_predict_normalises_messy_text(client):
    messy = {**VALID, "brand": "lenovo", "processor": "i5-8250U"}
    clean = client.post("/predict", json=VALID).json()["predicted_price_lsl"]
    response = client.post("/predict", json=messy)
    assert response.status_code == 200
    assert response.json()["predicted_price_lsl"] == clean


def test_predict_rejects_missing_field(client):
    response = client.post("/predict", json={k: v for k, v in VALID.items() if k != "brand"})
    assert response.status_code == 422


def test_predict_rejects_impossible_ram(client):
    response = client.post("/predict", json={**VALID, "ram_gb": 4096})
    assert response.status_code == 422


def test_predict_rejects_negative_price_input(client):
    response = client.post("/predict", json={**VALID, "storage_gb": -5})
    assert response.status_code == 422


def test_openapi_documents_predict(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200
    assert "/predict" in response.json()["paths"]


def test_metrics_returns_the_model_manifest(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    body = response.json()
    assert body["best_model"] == "random_forest"
    assert body["target"] == "price_lsl"
    assert body["split"]["test"] > 0
    assert {m["model"] for m in body["test_metrics"]} == {
        "median_baseline",
        "linear_regression",
        "random_forest",
    }
    for metric in body["test_metrics"]:
        assert metric["mae"] > 0
        assert metric["rmse"] > 0


def test_metrics_matches_the_version_file(client):
    version = serving.load_bundle().version
    body = client.get("/metrics").json()
    assert body["dataset_fingerprint"] == version["dataset_fingerprint"]
    assert body["dataset_rows"] == version["dataset_rows"]
    assert body["test_metrics"] == version["test_metrics"]
