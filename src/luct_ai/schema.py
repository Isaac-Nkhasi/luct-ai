"""Canonical schema, allowed values and validation rules for the laptop dataset.

Everything here is declarative so the cleaning notebook, the pipeline and the
tests all share one source of truth. No data rows live in this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Currency
# --------------------------------------------------------------------------- #

#: All prices are converted to this currency for modelling. The marketplace is
#: denominated in Lesotho loti (plural: maloti), so model errors are reported in
#: maloti.
TARGET_CURRENCY = "LSL"

# --------------------------------------------------------------------------- #
# Columns
# --------------------------------------------------------------------------- #

#: Columns a collector must fill in for every record.
REQUIRED_COLUMNS = [
    "record_id",
    "brand",
    "model",
    "processor",
    "ram_gb",
    "storage_gb",
    "storage_type",
    "age_years",
    "condition",
    "battery_health",
    "price",
    "currency",
    "source",
    "collection_date",
    "price_type",
]

#: Optional columns that improve provenance but are not mandatory.
OPTIONAL_COLUMNS = ["notes"]

ALL_COLUMNS = REQUIRED_COLUMNS + OPTIONAL_COLUMNS

#: Columns that must be non-null once a record has been standardised before it
#: can be used for modelling. Others may legitimately be missing.
CRITICAL_COLUMNS = [
    "record_id",
    "brand",
    "ram_gb",
    "storage_gb",
    "condition",
    "price",
    "currency",
    "price_type",
]

#: Columns produced by the cleaning step (never collected by hand).
DERIVED_COLUMNS = [
    "price_lsl",
    "fx_rate_used",
    "fx_rate_date",
    "fx_rate_source",
    "quality_flags",
    "is_duplicate",
    "split",
]

# --------------------------------------------------------------------------- #
# Controlled vocabularies
# --------------------------------------------------------------------------- #

#: Ordered worst-to-best so it can be used as an ordinal feature.
CONDITION_ORDER = ["Poor", "Fair", "Good", "Like New", "New"]

STORAGE_TYPES = ["SSD", "HDD", "eMMC", "SSD+HDD", "Unknown"]

PRICE_TYPES = ["asking", "completed"]

BRANDS = [
    "Acer",
    "Apple",
    "Asus",
    "Dell",
    "Fujitsu",
    "HP",
    "Lenovo",
    "Microsoft",
    "MSI",
    "Other",
    "Samsung",
    "Toshiba",
]

#: Lower-case alias -> canonical brand. Extend as real data arrives rather than
#: guessing; unmapped values are flagged, not silently relabelled.
BRAND_ALIASES = {
    "acer": "Acer",
    "apple": "Apple",
    "apple inc": "Apple",
    "macbook": "Apple",
    "asus": "Asus",
    "asustek": "Asus",
    "dell": "Dell",
    "dell inc": "Dell",
    "fujitsu": "Fujitsu",
    "hewlett packard": "HP",
    "hp": "HP",
    "hpe": "HP",
    "lenovo": "Lenovo",
    "ibm": "Lenovo",
    "ibm/lenovo": "Lenovo",
    "thinkpad": "Lenovo",
    "microsoft": "Microsoft",
    "surface": "Microsoft",
    "msi": "MSI",
    "samsung": "Samsung",
    "toshiba": "Toshiba",
    "dynabook": "Toshiba",
}

#: Processor families used as a categorical feature. Kept coarse on purpose:
#: fine-grained SKU strings are noisy and sparse.
PROCESSOR_FAMILIES = [
    "AMD Athlon",
    "AMD Ryzen 3",
    "AMD Ryzen 5",
    "AMD Ryzen 7",
    "AMD Ryzen 9",
    "Apple M1",
    "Apple M2",
    "Apple M3",
    "Intel Celeron",
    "Intel Core i3",
    "Intel Core i5",
    "Intel Core i7",
    "Intel Core i9",
    "Intel Pentium",
    "Other",
]

# --------------------------------------------------------------------------- #
# Plausibility bounds (used to flag, never to silently drop)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ValidationRules:
    """Bounds and vocabularies used to flag suspicious records."""

    min_ram_gb: int = 1
    max_ram_gb: int = 256
    min_storage_gb: int = 8
    max_storage_gb: int = 20000
    min_age_years: float = 0.0
    max_age_years: float = 25.0
    min_battery_health: int = 0
    max_battery_health: int = 100
    #: Plausible asking-price band in the target currency (LSL).
    min_price_lsl: float = 100.0
    max_price_lsl: float = 100_000.0
    #: A price this many times above the dataset median is flagged (IQR fence is
    #: also applied). Configurable because it depends on the collected sample.
    outlier_multiplier: float = 5.0
    conditions: tuple[str, ...] = tuple(CONDITION_ORDER)
    storage_types: tuple[str, ...] = tuple(STORAGE_TYPES)
    price_types: tuple[str, ...] = tuple(PRICE_TYPES)
    brands: tuple[str, ...] = tuple(BRANDS)
    processor_families: tuple[str, ...] = tuple(PROCESSOR_FAMILIES)
    extra_flag_columns: tuple[str, ...] = field(default_factory=tuple)


DEFAULT_RULES = ValidationRules()

# --------------------------------------------------------------------------- #
# Train / validation / test split
# --------------------------------------------------------------------------- #

#: The dataset must be split by laptop identity so the same physical machine
#: specification never appears in both training and test data.
GROUP_KEY_COLUMNS = ["brand", "model", "processor", "ram_gb", "storage_gb", "storage_type"]

DEFAULT_TEST_SIZE = 0.2
DEFAULT_VAL_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42


def canonical_columns() -> list[str]:
    """Return the collected columns in canonical order."""
    return list(ALL_COLUMNS)
