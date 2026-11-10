"""Unit tests for the Phase 8 cleaning helpers."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from luct_ai import cleaning
from luct_ai.schema import DEFAULT_RULES


def test_parse_gb_handles_units():
    assert cleaning.parse_gb("8GB") == 8
    assert cleaning.parse_gb("512 gb") == 512
    assert cleaning.parse_gb("1TB") == 1024
    assert cleaning.parse_gb("256") == 256
    assert math.isnan(cleaning.parse_gb(""))
    assert math.isnan(cleaning.parse_gb("unknown"))


def test_standardise_brand_maps_aliases():
    assert cleaning.standardise_brand("lenovo") == "Lenovo"
    assert cleaning.standardise_brand("IBM/Lenovo") == "Lenovo"
    assert cleaning.standardise_brand("Apple Inc") == "Apple"
    assert cleaning.standardise_brand("Totally Unknown Brand") == "Other"
    assert cleaning.standardise_brand("") == ""


def test_standardise_processor_reduces_to_family():
    assert cleaning.standardise_processor("Intel Core i5-8250U") == "Intel Core i5"
    assert cleaning.standardise_processor("core i7 10th gen") == "Intel Core i7"
    assert cleaning.standardise_processor("AMD Ryzen 5 3500U") == "AMD Ryzen 5"
    assert cleaning.standardise_processor("Apple M1") == "Apple M1"
    assert cleaning.standardise_processor("Celeron N4020") == "Intel Celeron"
    assert cleaning.standardise_processor("Pentium Gold") == "Intel Pentium"


def test_standardise_condition_and_ordinal():
    assert cleaning.standardise_condition("excellent") == "Like New"
    assert cleaning.standardise_condition("USED") == "Good"
    assert cleaning.standardise_condition("") == ""
    assert cleaning.condition_ordinal("New") == 4.0
    assert cleaning.condition_ordinal("Poor") == 0.0


def test_standardise_storage_type():
    assert cleaning.standardise_storage_type("solid state drive") == "SSD"
    assert cleaning.standardise_storage_type("NVMe") == "SSD"
    assert cleaning.standardise_storage_type("hard disk drive") == "HDD"
    assert cleaning.standardise_storage_type("") == "Unknown"


def test_summarise_missingness_counts(raw_frame):
    frame = raw_frame.copy()
    frame.loc[0, "battery_health"] = pd.NA
    report = cleaning.summarise_missingness(frame)
    battery = report.set_index("column").loc["battery_health"]
    assert battery["missing"] == 1
    assert battery["present"] == len(frame) - 1


def test_deduplicate_removes_identical_records(raw_frame):
    duplicated = pd.concat([raw_frame, raw_frame.iloc[[0, 1]]], ignore_index=True)
    kept, removed = cleaning.deduplicate(duplicated)
    assert len(removed) == 2
    assert len(kept) == len(raw_frame)
    assert kept["is_duplicate"].sum() == 0


def test_convert_currency_identity_and_missing_rate(raw_frame):
    frame = raw_frame.copy()
    frame.loc[0, "currency"] = "USD"
    frame.loc[0, "price"] = 100
    rates = {"USD": {"rate": 18.5, "date": "2026-01-15", "source": "test"}}
    converted = cleaning.convert_currency(cleaning.standardise_dataframe(frame), rates)

    # LSL rows pass through unchanged and are tagged 'identity'.
    lsl_row = converted[converted["currency"] == "LSL"].iloc[0]
    assert lsl_row["price_lsl"] == float(lsl_row["price"])
    assert lsl_row["fx_rate_source"] == "identity"

    # USD row is converted with the recorded rate/date.
    usd_row = converted[converted["currency"] == "USD"].iloc[0]
    assert usd_row["price_lsl"] == 100 * 18.5
    assert usd_row["fx_rate_used"] == "18.5"
    assert usd_row["fx_rate_date"] == "2026-01-15"


def test_convert_currency_flags_unconvertible():
    frame = pd.DataFrame(
        [{"record_id": "A1", "brand": "Dell", "price": 500, "currency": "EUR"}]
    )
    converted = cleaning.convert_currency(frame, {})
    assert math.isnan(converted.loc[0, "price_lsl"])
    assert "fx_missing_EUR" in converted.loc[0, "quality_flags"]


def test_split_by_group_has_no_group_leakage(cleaned_frame):
    keys = (
        cleaned_frame[["brand", "model", "processor", "ram_gb", "storage_gb", "storage_type"]]
        .astype("string").fillna("NA").agg("|".join, axis=1)
    )
    per_group = pd.DataFrame({"key": keys, "split": cleaned_frame["split"]})
    assert per_group.groupby("key")["split"].nunique().max() == 1


def test_clean_dataset_end_to_end(raw_frame):
    result = cleaning.clean_dataset(raw_frame)
    assert not result.frame.empty
    assert set(result.frame["split"].unique()) <= {"train", "val", "test"}
    assert "price_lsl" in result.frame.columns
    assert "quality_flags" in result.frame.columns
    report = result.to_report()
    # Every record in the fixture is a legitimate asking price in LSL.
    assert report["records_after_cleaning"] == len(raw_frame)
    assert isinstance(report["notes"], list)


def test_clean_dataset_flags_impossible_specs(raw_frame):
    broken = raw_frame.copy()
    broken.loc[0, "ram_gb"] = 9999  # impossible
    broken.loc[1, "battery_health"] = 150  # impossible
    result = cleaning.clean_dataset(broken)
    flags = ";".join(result.frame["quality_flags"].astype(str))
    assert "implausible_ram" in flags
    assert "implausible_battery_health" in flags


def test_standardise_dataframe_outputs_numeric_dtypes(raw_frame):
    standardised = cleaning.standardise_dataframe(raw_frame, DEFAULT_RULES)
    assert pd.api.types.is_numeric_dtype(standardised["ram_gb"])
    assert pd.api.types.is_numeric_dtype(standardised["price"])
    assert standardised["brand"].iloc[0] == "Lenovo"
    assert standardised["processor"].iloc[0] == "Intel Core i5"
