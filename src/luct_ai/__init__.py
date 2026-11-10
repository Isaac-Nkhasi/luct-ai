"""LUCT Market AI — used-laptop asking-price estimator.

Phase 8 builds and cleans the laptop-price dataset.
Phase 9 trains and evaluates the price models.

The package intentionally contains no dataset rows. Data is collected
separately (see docs/data_collection_plan.md) and must never be invented.
"""

from .schema import ALL_COLUMNS, REQUIRED_COLUMNS, TARGET_CURRENCY  # noqa: F401

__version__ = "0.1.0"
