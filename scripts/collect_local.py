"""Validate and merge contributed laptop records into the local dataset.

This is the intake step for Tier B (permitted public listings) and Tier C
(voluntary student contributions) described in ``docs/data_collection_plan.md``.

It is a **gatekeeper, not a transformer**:

* it never invents a row — every output row comes from a contribution file;
* it never converts prices — that is the pipeline's job, so the FX rate stays
  traceable;
* it rejects a row rather than silently "fixing" it, and reports every reason.

Run it after collecting one or more contribution CSVs (the form in
``kit/contribute.html`` produces exactly one such file per contributor):

    python scripts/collect_local.py
    python -m luct_ai.pipeline all --input data/raw/laptop_prices_local.csv

By default it reads every ``*.csv`` in ``data/contributions/`` and writes
``data/raw/laptop_prices_local.csv``. ``record_id`` values are (re)assigned here
so they are unique across the whole local file; any value a contributor typed is
ignored.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from luct_ai import cleaning, schema  # noqa: E402

DEFAULT_CONTRIBUTIONS = ROOT / "data" / "contributions"
DEFAULT_OUTPUT = ROOT / "data" / "raw" / "laptop_prices_local.csv"
DEFAULT_PREFIX = "LAP"

#: Every collected field that must be present for a row to be usable. The
#: data dictionary marks ``battery_health`` and ``notes`` optional, so they are
#: deliberately absent here.
REQUIRED_FOR_INTAKE = [
    "brand",
    "model",
    "processor",
    "ram_gb",
    "storage_gb",
    "storage_type",
    "age_years",
    "condition",
    "price",
    "currency",
    "price_type",
    "source",
    "collection_date",
]

#: Numeric ranges, mirroring ``schema.ValidationRules``.
RANGE_CHECKS = (("ram_gb", 1, 256), ("storage_gb", 8, 20000), ("age_years", 0, 25))


def _text(value: object) -> str:
    """Return a trimmed string, mapping any missing value to ``""``."""
    if pd.isna(value):
        return ""
    return str(value).strip()


def _number(value: object) -> float:
    """Parse a possibly formatted number; ``NaN`` when it cannot be parsed."""
    return float(pd.to_numeric(_text(value).replace(",", ""), errors="coerce"))


def _is_iso_date(value: object) -> bool:
    text = _text(value)
    if not text:
        return False
    try:
        _dt.date.fromisoformat(text)
    except ValueError:
        return False
    return True


def _row_problem(row: pd.Series) -> str | None:
    """Return the first reason a row is unusable, or ``None`` when it is fine."""
    for column in REQUIRED_FOR_INTAKE:
        if not _text(row.get(column)):
            return f"missing {column}"
    if not _number(row.get("price")) > 0:
        return "price is not a number greater than 0"
    for column, low, high in RANGE_CHECKS:
        value = _number(row.get(column))
        if pd.isna(value):
            return f"{column} is not a number"
        if not low <= value <= high:
            return f"{column} outside {low}-{high}"
    if _text(row.get("price_type")).lower() not in schema.PRICE_TYPES:
        return "price_type must be 'asking' or 'completed'"
    if not _is_iso_date(row.get("collection_date")):
        return "collection_date must be YYYY-MM-DD"
    return None


def collect(contributions: Path, output: Path, prefix: str = DEFAULT_PREFIX) -> int:
    files = sorted(Path(contributions).glob("*.csv"))
    if not files:
        raise SystemExit(
            f"No contribution CSVs found in {contributions}.\n"
            "Collect some first (see kit/README.md and kit/contribute.html), then re-run.\n"
            "No records have been invented for you."
        )

    accepted: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    offered = 0

    for path in files:
        try:
            frame = cleaning.read_dataset(path)
        except ValueError as exc:
            rejected.append({"file": path.name, "row": "-", "reason": str(exc)})
            continue

        extra = [c for c in frame.columns if c not in schema.ALL_COLUMNS]
        if extra:
            print(f"  note: {path.name}: ignoring unrecognised column(s): {', '.join(extra)}")

        for line, (_, row) in enumerate(frame.iterrows(), start=2):  # line 1 is the header
            offered += 1
            problem = _row_problem(row)
            if problem:
                rejected.append({"file": path.name, "row": str(line), "reason": problem})
                continue
            record = {
                column: ("" if pd.isna(row.get(column)) else _text(row.get(column)))
                for column in schema.ALL_COLUMNS
                if column != "record_id"
            }
            accepted.append(record)

    # Assign unique ids over the merged set; ignored whatever a contributor typed.
    width = max(5, len(str(len(accepted))))
    for number, record in enumerate(accepted, start=1):
        record["record_id"] = f"{prefix}{number:0{width}d}"

    output = Path(output)
    if accepted:
        frame = pd.DataFrame(accepted)[schema.ALL_COLUMNS]
        cleaning.write_dataset(frame, output)

    # ------------------------------------------------------------------ report
    print(f"contribution files     : {len(files)}")
    print(f"rows offered           : {offered}")
    print(f"rows accepted          : {len(accepted)}")
    print(f"rows rejected          : {len(rejected)}")
    if accepted:
        print(f"written                : {output}")
        print("source breakdown       : "
              + str(frame["source"].value_counts(dropna=False).to_dict()))
        print("price_type breakdown   : "
              + str(frame["price_type"].value_counts(dropna=False).to_dict()))
    else:
        print("nothing written — no usable rows were found.")

    if rejected:
        reasons: dict[str, int] = {}
        for item in rejected:
            key = item["reason"]
            # Group blank-field failures together; keep the rest as-is.
            if key.startswith("missing "):
                key = "missing a required field"
            reasons[key] = reasons.get(key, 0) + 1
        print("rejection reasons      : " + str(reasons))
        print(f"  (first few: "
              + "; ".join(f"{r['file']}:{r['row']} {r['reason']}" for r in rejected[:5]) + ")")

    print(f"\nActual number collected: {len(accepted)} local record(s) "
          f"from {len(files)} file(s). Nothing was invented.")
    return 0 if accepted else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--contributions", default=str(DEFAULT_CONTRIBUTIONS),
                        help="folder containing contribution CSVs")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="merged local dataset to write")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX,
                        help="record_id prefix for the local dataset")
    args = parser.parse_args(argv)
    return collect(Path(args.contributions), Path(args.output), args.prefix)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
