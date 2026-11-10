"""Pytest configuration and shared test fixtures.

IMPORTANT: the fixtures below are **synthetic test data used only to exercise the
code paths in unit tests**. They are not, and must never be presented as, the
collected laptop dataset. The real dataset is collected separately following
docs/data_collection_plan.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _fixture_rows() -> list[dict]:
    """Build a small, deterministic frame spanning several laptop groups."""
    groups = [
        ("Lenovo", "ThinkPad T480", "Intel Core i5", 8, 256, "SSD", 5, "Good", 80, 4500),
        ("Dell", "Latitude 5490", "Intel Core i5", 8, 256, "SSD", 4, "Good", 85, 4200),
        ("HP", "EliteBook 840", "Intel Core i7", 16, 512, "SSD", 3, "Like New", 90, 7000),
        ("Apple", "MacBook Air M1", "Apple M1", 8, 256, "SSD", 2, "Like New", 95, 9500),
        ("Acer", "Aspire 5", "AMD Ryzen 5", 8, 512, "SSD", 2, "Good", 88, 5200),
        ("Asus", "VivoBook 15", "Intel Core i3", 4, 1000, "HDD", 6, "Fair", 60, 2600),
    ]
    rows = []
    index = 1
    for repeat in range(4):  # 6 groups x 4 = 24 rows, same group repeated
        for (brand, model, cpu, ram, storage, storage_type, age, cond, battery, price) in groups:
            rows.append(
                {
                    "record_id": f"TEST{index:03d}",
                    "brand": brand,
                    "model": model,
                    "processor": cpu,
                    "ram_gb": ram,
                    "storage_gb": storage,
                    "storage_type": storage_type,
                    "age_years": age,
                    "condition": cond,
                    "battery_health": battery,
                    "price": price + repeat * 25,  # tiny spread so prices differ
                    "currency": "LSL",
                    "source": "test fixture",
                    "collection_date": "2026-01-15",
                    "price_type": "asking",
                }
            )
            index += 1
    return rows


@pytest.fixture
def raw_frame() -> pd.DataFrame:
    """A raw-schema frame (tests only) ready for cleaning.clean_dataset."""
    return pd.DataFrame(_fixture_rows())


@pytest.fixture
def cleaned_frame(raw_frame: pd.DataFrame) -> pd.DataFrame:
    """A cleaned frame with a grouped split (tests only)."""
    from luct_ai import cleaning

    from luct_ai.schema import DEFAULT_RULES

    standardised = cleaning.standardise_dataframe(raw_frame, DEFAULT_RULES)
    flagged = cleaning.convert_currency(standardised, {})
    flagged = cleaning.flag_quality(flagged, DEFAULT_RULES)
    flagged = cleaning.add_price_band(flagged)
    kept, _ = cleaning.deduplicate(flagged)
    split_frame, _ = cleaning.split_by_group(kept)
    return split_frame
