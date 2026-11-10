"""End-to-end test of the Phase 8 + Phase 9 orchestration.

Uses the synthetic test fixture from ``conftest.py`` written to a temporary
directory. It proves the pipeline writes its artefacts; it produces no claim
about real-world model quality.
"""

from __future__ import annotations

import json

from luct_ai import pipeline


def test_run_cleaning_writes_artefacts(tmp_path, raw_frame):
    raw_path = tmp_path / "raw.csv"
    raw_frame.to_csv(raw_path, index=False)

    result = pipeline.run_cleaning(raw_path, output_root=tmp_path)

    assert result.usable_records == len(raw_frame)
    assert (tmp_path / "data" / "processed" / "laptop_prices_clean.csv").exists()
    assert (tmp_path / "reports" / "cleaning_report.md").exists()
    report = json.loads((tmp_path / "reports" / "cleaning_report.json").read_text(encoding="utf-8"))
    assert report["records_after_cleaning"] == len(raw_frame)


def test_run_training_writes_model_and_reports(tmp_path, raw_frame):
    raw_path = tmp_path / "raw.csv"
    raw_frame.to_csv(raw_path, index=False)
    pipeline.run_cleaning(raw_path, output_root=tmp_path)

    cleaned_path = tmp_path / "data" / "processed" / "laptop_prices_clean.csv"
    result = pipeline.run_training(cleaned_path, output_root=tmp_path, tune=False)

    assert result.best_model_name in result.models
    assert (tmp_path / "models" / "laptop_price_model.joblib").exists()
    assert (tmp_path / "models" / "model_version.json").exists()
    assert (tmp_path / "reports" / "model_report.md").exists()

    manifest = json.loads((tmp_path / "models" / "model_version.json").read_text(encoding="utf-8"))
    assert manifest["best_model"] == result.best_model_name
    assert "test_metrics" in manifest
    assert manifest["dataset_rows"] == len(raw_frame)


def test_missing_input_does_not_fabricate(tmp_path):
    import pytest

    raw_path = tmp_path / "raw.csv"
    raw_path.write_text(
        "record_id,brand,model,processor,ram_gb,storage_gb,storage_type,age_years,"
        "condition,battery_health,price,currency,source,collection_date,price_type\n",
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        pipeline.run_cleaning(raw_path, output_root=tmp_path)
