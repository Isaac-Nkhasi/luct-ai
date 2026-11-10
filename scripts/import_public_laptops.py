"""Import the public laptop-listing dataset into the LUCT Market AI schema.

Provenance
----------
Dataset : ``bitsnaps/laptops_specs`` on Hugging Face
URL     : https://huggingface.co/datasets/bitsnaps/laptops_specs
File    : ``laptops_specs.parquet`` (1.97 MB, 8 835 rows)
Licence : Apache-2.0 (as declared by the uploader on the dataset card).
          The card describes the source only as "Laptop prices from Algerian
          public data sources"; the underlying listings are third-party data.
          Verify the licence is acceptable for your submission before relying
          on it.
Fetched : 2026-11-11

What this script does
---------------------
It maps the public dataset onto the columns defined in
:mod:`luct_ai.schema`, so the normal Phase 8/9 pipeline can run on it:

    python scripts/import_public_laptops.py
    python -m luct_ai.pipeline all --input data/raw/laptop_prices_raw.csv

Honesty rules applied here
--------------------------
* No row is invented. Every output row comes from a real source row.
* Fields the source does not contain are left **empty** (they are not guessed):
  ``model``, ``age_years`` and ``battery_health``.
* Prices are in Algerian dinar (DZD) and are converted to LSL by the normal
  cleaning step, which records the rate, date and source per record. The rate
  lives in ``data/fx_rates.csv``; nothing is hard-coded here.
* Prices that fall outside the schema's plausibility band (100–100 000 LSL)
  are blanked and noted, so a broken scrape cannot poison the model. The row
  is kept, and the reason is recorded in ``notes``.

Known limitations of the source
-------------------------------
* These are **Algerian asking prices**, not Lesotho used-market prices.
* Most listings are **new** retail stock: 2 312 are marked ``neuf`` (new),
  only 219 are ``occasion`` (used), and 6 304 carry no condition at all.
* Condition is coarse: "occasion" carries no grade, so it is mapped to the
  project's ``Good`` bucket (the same target the cleaner already uses for the
  English word "used").
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from luct_ai import cleaning  # noqa: E402
from luct_ai.schema import DEFAULT_RULES, REQUIRED_COLUMNS  # noqa: E402

SOURCE_LABEL = "Public dataset: bitsnaps/laptops_specs (Hugging Face, Apache-2.0)"
SOURCE_CURRENCY = "DZD"
#: Date the dataset was downloaded and ingested (the listings' own dates are
#: not published by the source).
COLLECTION_DATE = "2026-11-11"

#: French listing terms used by the source -> project condition vocabulary.
CONDITION_MAP = {
    "neuf": "New",       # "neuf" = brand new
    "occasion": "Good",  # "occasion" = second-hand; the source gives no grade
}


def derive_storage_type(row: pd.Series) -> str:
    """Collapse the source's boolean storage flags into one vocabulary value."""
    if row.get("is_ssd") and row.get("is_hdd"):
        return "SSD+HDD"
    if row.get("is_ssd") or row.get("is_nvme"):
        return "SSD"
    if row.get("is_hdd"):
        return "HDD"
    if row.get("is_emmc"):
        return "eMMC"
    return "Unknown"


def to_schema(source: pd.DataFrame, dzd_rate: float, rules=DEFAULT_RULES) -> pd.DataFrame:
    """Return a frame with exactly the columns the cleaning step expects."""
    min_dzd = rules.min_price_lsl / dzd_rate
    max_dzd = rules.max_price_lsl / dzd_rate

    prices = pd.to_numeric(source["price_da"], errors="coerce")
    price_is_usable = prices.between(min_dzd, max_dzd)

    titles = source["title"].astype("string").fillna("")
    notes = "external public listing (bitsnaps/laptops_specs); source title: " \
        + titles.str.slice(0, 120)
    notes = notes.where(price_is_usable,
                        notes + "; price_da=" + prices.astype("string")
                        + " rejected (outside " + f"{rules.min_price_lsl:g}"
                        + f"-{rules.max_price_lsl:g} LSL band)")

    out = pd.DataFrame(
        {
            "record_id": [f"PUB{i:05d}" for i in range(1, len(source) + 1)],
            "brand": source["brand"].astype("string"),
            # Not published by the source: left empty rather than guessed.
            "model": pd.NA,
            "processor": source["cpu_raw"].astype("string"),
            "ram_gb": pd.to_numeric(source["ram_gb"], errors="coerce"),
            "storage_gb": pd.to_numeric(source["storage_gb"], errors="coerce"),
            "storage_type": source.apply(derive_storage_type, axis=1),
            "age_years": pd.NA,
            "condition": source["condition"].astype("string").fillna("").str.lower()
                        .map(CONDITION_MAP).fillna(""),
            "battery_health": pd.NA,
            # Blank out prices that fall outside the plausible band (kept, flagged).
            "price": prices.where(price_is_usable),
            "currency": SOURCE_CURRENCY,
            "source": SOURCE_LABEL,
            "collection_date": COLLECTION_DATE,
            "price_type": "asking",
            "notes": notes,
        }
    )
    return out[REQUIRED_COLUMNS + ["notes"]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", default=str(ROOT / "data" / "raw" / "laptops_specs.parquet"))
    parser.add_argument("--output", default=str(ROOT / "data" / "raw" / "laptop_prices_raw.csv"))
    parser.add_argument("--rates", default=str(ROOT / "data" / "fx_rates.csv"))
    args = parser.parse_args(argv)

    rates = cleaning.load_fx_rates(args.rates)
    if SOURCE_CURRENCY not in rates:
        raise SystemExit(
            f"No {SOURCE_CURRENCY}->LSL rate in {args.rates}. Add one (rate, date, "
            "source) before importing; this script will not guess a rate."
        )
    dzd_rate = float(rates[SOURCE_CURRENCY]["rate"])

    source = pd.read_parquet(args.input)
    frame = to_schema(source, dzd_rate)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    blanks = int(frame["price"].isna().sum())
    print(f"source rows          : {len(source)}")
    print(f"written              : {output}")
    print(f"records              : {len(frame)}")
    print(f"via DZD->LSL rate    : {dzd_rate} ({rates[SOURCE_CURRENCY]['date']})")
    print(f"prices blanked       : {blanks} (outside plausible band, kept + noted)")
    print(f"condition breakdown  : {frame['condition'].value_counts(dropna=False).to_dict()}")
    print(f"storage type         : {frame['storage_type'].value_counts().to_dict()}")
    print(f"currencies           : {frame['currency'].value_counts().to_dict()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
