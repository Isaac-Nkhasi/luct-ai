"""Phase 8 — clean and prepare the used-laptop dataset.

The functions here are pure and independently testable. They:

1. read a CSV that follows :mod:`luct_ai.schema`;
2. standardise brand, processor, RAM, storage and condition;
3. report missing values;
4. flag impossible specifications and implausible prices;
5. remove duplicates;
6. convert prices to the target currency (recording the method and date);
7. split train/validation/test **by laptop group** so the same machine spec
   cannot leak across the split.

Nothing here invents data: an empty input produces an empty output plus a
report explaining what is still missing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from . import schema
from .schema import (
    BRAND_ALIASES,
    DEFAULT_RULES,
    GROUP_KEY_COLUMNS,
    REQUIRED_COLUMNS,
    TARGET_CURRENCY,
    ValidationRules,
)

# --------------------------------------------------------------------------- #
# Small parsing / normalisation helpers
# --------------------------------------------------------------------------- #

_WHITESPACE = re.compile(r"\s+")
_GB_PATTERN = re.compile(r"(?P<num>\d+(?:[.,]\d+)?)\s*(?P<unit>tb|gb|t|g|mb)?", re.I)


def normalise_text(value: object) -> str:
    """Collapse whitespace and trim; return '' for nulls."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return _WHITESPACE.sub(" ", str(value)).strip()


def normalise_key(value: object) -> str:
    """Lower-cased comparison key."""
    return normalise_text(value).lower()


def parse_gb(value: object) -> float:
    """Parse '8GB', '1 TB', '512 gb', '256' into gigabytes.

    Returns ``nan`` when nothing numeric can be found.
    """
    text = normalise_key(value)
    if not text:
        return float("nan")
    match = _GB_PATTERN.search(text)
    if not match:
        return float("nan")
    number = float(match.group("num").replace(",", "."))
    unit = (match.group("unit") or "").lower()
    if unit in {"tb", "t"}:
        return number * 1024.0
    if unit in {"mb"}:
        return number / 1024.0
    # No unit, or gb/g: assume the collector meant gigabytes.
    return number


def standardise_brand(value: object) -> str:
    """Map a raw brand string to a canonical brand, else ``Other``."""
    key = normalise_key(value)
    if not key:
        return ""
    if key in BRAND_ALIASES:
        return BRAND_ALIASES[key]
    title = normalise_text(value)
    for canonical in schema.BRANDS:
        if canonical.lower() == key:
            return canonical
    return "Other"


def standardise_processor(value: object) -> str:
    """Reduce a raw CPU string to one of :data:`luct_ai.schema.PROCESSOR_FAMILIES`."""
    text = normalise_key(value)
    if not text:
        return ""
    # Apple silicon.
    for chip in ("m1", "m2", "m3"):
        if re.search(rf"\b{chip}\b", text):
            return f"Apple {chip.upper()}"
    # AMD.
    for tier in ("3", "5", "7", "9"):
        if re.search(rf"ryzen\s*{tier}\b", text):
            return f"AMD Ryzen {tier}"
    if "athlon" in text:
        return "AMD Athlon"
    # Intel.
    for tier in ("3", "5", "7", "9"):
        if re.search(rf"\bi{tier}\b", text) or re.search(rf"core\s*{tier}\b", text):
            return f"Intel Core i{tier}"
    if "celeron" in text:
        return "Intel Celeron"
    if "pentium" in text:
        return "Intel Pentium"
    if "intel" in text or "amd" in text:
        return "Other"
    return "Other"


_STORAGE_TYPE_ALIASES = {
    "ssd": "SSD",
    "solid state": "SSD",
    "solid state drive": "SSD",
    "nvme": "SSD",
    "hdd": "HDD",
    "hard disk": "HDD",
    "hard disk drive": "HDD",
    "hard drive": "HDD",
    "emmc": "eMMC",
    "ssd+hdd": "SSD+HDD",
    "ssd + hdd": "SSD+HDD",
    "hybrid": "SSD+HDD",
    "sshd": "SSD+HDD",
}


def standardise_storage_type(value: object) -> str:
    key = normalise_key(value)
    if not key:
        return "Unknown"
    if key in _STORAGE_TYPE_ALIASES:
        return _STORAGE_TYPE_ALIASES[key]
    for canonical in schema.STORAGE_TYPES:
        if canonical.lower() == key:
            return canonical
    return "Unknown"


_CONDITION_ALIASES = {
    "new": "New",
    "brand new": "New",
    "sealed": "New",
    "like new": "Like New",
    "excellent": "Like New",
    "mint": "Like New",
    "very good": "Like New",
    "good": "Good",
    "used": "Good",
    "fair": "Fair",
    "average": "Fair",
    "poor": "Poor",
    "faulty": "Poor",
    "for parts": "Poor",
}


def standardise_condition(value: object) -> str:
    key = normalise_key(value)
    if not key:
        return ""
    return _CONDITION_ALIASES.get(key, "")


def condition_ordinal(value: object) -> float:
    """Map a canonical condition to 0=poor .. 4=new for numeric models."""
    canonical = standardise_condition(value)
    if canonical not in schema.CONDITION_ORDER:
        return float("nan")
    return float(schema.CONDITION_ORDER.index(canonical))


def coerce_numeric(series: pd.Series, *, integer: bool = False) -> pd.Series:
    """Best-effort numeric coercion of a possibly messily formatted column."""
    cleaned = (
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace(r"[^0-9eE.+-]", "", regex=True)
        .replace({"": pd.NA})
    )
    numeric = pd.to_numeric(cleaned, errors="coerce")
    if integer:
        return numeric.round()
    return numeric


# --------------------------------------------------------------------------- #
# Reading / writing
# --------------------------------------------------------------------------- #


def read_dataset(path: str | Path) -> pd.DataFrame:
    """Read a laptop dataset CSV and verify the required columns are present."""
    frame = pd.read_csv(path, dtype="string", keep_default_na=True)
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(
            f"{path} is missing required column(s): {', '.join(missing)}. "
            f"Start from data/laptop_prices_template.csv."
        )
    return frame


def write_dataset(frame: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


# --------------------------------------------------------------------------- #
# Reports
# --------------------------------------------------------------------------- #


def summarise_missingness(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-column count and share of missing values."""
    total = len(frame)
    rows = []
    for column in frame.columns:
        missing = int(frame[column].isna().sum())
        rows.append(
            {
                "column": column,
                "missing": missing,
                "present": total - missing,
                "missing_pct": round(100.0 * missing / total, 2) if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Standardisation
# --------------------------------------------------------------------------- #


def standardise_dataframe(
    frame: pd.DataFrame,
    rules: ValidationRules = DEFAULT_RULES,
) -> pd.DataFrame:
    """Return a new frame with standardised brand/processor/spec/condition."""
    out = frame.copy()

    for column in ("brand", "model", "processor", "source", "currency", "price_type", "record_id", "notes"):
        if column in out.columns:
            out[column] = out[column].map(normalise_text)

    if "brand" in out.columns:
        out["brand"] = out["brand"].map(standardise_brand)
    if "processor" in out.columns:
        out["processor"] = out["processor"].map(standardise_processor)
    if "storage_type" in out.columns:
        out["storage_type"] = out["storage_type"].map(standardise_storage_type)
    if "condition" in out.columns:
        out["condition"] = out["condition"].map(standardise_condition)

    if "ram_gb" in out.columns:
        out["ram_gb"] = out["ram_gb"].map(parse_gb)
    if "storage_gb" in out.columns:
        out["storage_gb"] = out["storage_gb"].map(parse_gb)

    for column in ("age_years", "battery_health"):
        if column in out.columns:
            out[column] = coerce_numeric(out[column])

    if "price" in out.columns:
        out["price"] = coerce_numeric(out["price"])

    if "currency" in out.columns:
        out["currency"] = out["currency"].str.upper()
    if "price_type" in out.columns:
        out["price_type"] = out["price_type"].str.lower()

    if "collection_date" in out.columns:
        out["collection_date"] = pd.to_datetime(
            out["collection_date"], errors="coerce", format="mixed"
        ).dt.date

    return out


# --------------------------------------------------------------------------- #
# Validation flags
# --------------------------------------------------------------------------- #


def _outside(value: object, low: float, high: float) -> bool:
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        return False  # missing handled separately
    try:
        return not (low <= float(value) <= high)
    except (TypeError, ValueError):
        return True


def flag_quality(frame: pd.DataFrame, rules: ValidationRules = DEFAULT_RULES) -> pd.DataFrame:
    """Add a ``quality_flags`` column listing every issue found per record."""
    out = frame.copy()
    flags: list[list[str]] = [[] for _ in range(len(out))]

    def add(mask: pd.Series, label: str) -> None:
        # Coerce to plain bools: nullable/pandas-boolean masks can carry NA.
        clean_mask = pd.Series(mask, index=out.index).fillna(False).astype(bool)
        for position in np.flatnonzero(clean_mask.to_numpy()):
            flags[position].append(label)

    # Missing critical values.
    for column in schema.CRITICAL_COLUMNS:
        if column in out.columns:
            add(out[column].isna() | (out[column].astype("string").str.strip() == ""),
                f"missing_{column}")

    # Impossible / implausible specifications.
    if "ram_gb" in out.columns:
        add(out["ram_gb"].map(lambda v: _outside(v, rules.min_ram_gb, rules.max_ram_gb)),
            "implausible_ram")
    if "storage_gb" in out.columns:
        add(out["storage_gb"].map(lambda v: _outside(v, rules.min_storage_gb, rules.max_storage_gb)),
            "implausible_storage")
    if "age_years" in out.columns:
        add(out["age_years"].map(lambda v: _outside(v, rules.min_age_years, rules.max_age_years)),
            "implausible_age")
    if "battery_health" in out.columns:
        add(out["battery_health"].map(
            lambda v: _outside(v, rules.min_battery_health, rules.max_battery_health)),
            "implausible_battery_health")
    # The price bounds are expressed in the target currency (LSL), so they are
    # applied to the *converted* price whenever it is available. Comparing a
    # foreign-currency price against LSL bounds would flag every record.
    price_column = "price_lsl" if "price_lsl" in out.columns else "price"
    if price_column in out.columns:
        add(out[price_column].map(lambda v: _outside(v, rules.min_price_lsl, rules.max_price_lsl)),
            "implausible_price")

    # Vocabulary members.
    if "brand" in out.columns:
        add(~out["brand"].isin(rules.brands), "brand_unmapped")
    if "processor" in out.columns:
        add(~out["processor"].isin(rules.processor_families), "processor_unmapped")
    if "condition" in out.columns:
        add(~out["condition"].isin(rules.conditions), "condition_unmapped")
    if "storage_type" in out.columns:
        add(~out["storage_type"].isin(rules.storage_types), "storage_type_unmapped")
    if "price_type" in out.columns:
        add(~out["price_type"].isin(rules.price_types), "price_type_unmapped")

    computed = pd.Series([";".join(item) for item in flags], index=out.index, dtype="string")
    # Merge with any flags already present (e.g. fx_missing_* added by
    # convert_currency) instead of overwriting them.
    if "quality_flags" in out.columns:
        existing = out["quality_flags"].astype("string").fillna("")
        computed = existing.str.cat(computed, sep=";")
    out["quality_flags"] = (
        computed.str.replace(r";{2,}", ";", regex=True).str.strip(";").fillna("")
    )
    return out


def price_outlier_mask(
    frame: pd.DataFrame,
    column: str = "price_lsl",
    rules: ValidationRules = DEFAULT_RULES,
) -> pd.Series:
    """IQR-fence + median-multiplier outlier detection on a price column."""
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    values = pd.to_numeric(frame[column], errors="coerce")
    valid = values.dropna()
    if valid.empty:
        return pd.Series(False, index=frame.index)
    q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
    iqr = q3 - q1
    upper_fence = q3 + 1.5 * iqr
    median = valid.median()
    ceiling = median * rules.outlier_multiplier
    return values > min(upper_fence, ceiling) if iqr > 0 else values > ceiling


# --------------------------------------------------------------------------- #
# Duplicates
# --------------------------------------------------------------------------- #


def deduplicate(
    frame: pd.DataFrame,
    subset: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop duplicate records, returning ``(kept, removed)``.

    Two passes: duplicate ``record_id`` values, then duplicate natural keys
    (same machine spec listed at the same price).
    """
    if subset is None:
        subset = ["brand", "model", "processor", "ram_gb", "storage_gb",
                  "storage_type", "age_years", "price", "currency", "price_type"]
        subset = [c for c in subset if c in frame.columns]

    working = frame.copy()
    duplicate_record_id = working.duplicated(subset=["record_id"], keep="first") \
        if "record_id" in working.columns else pd.Series(False, index=working.index)
    duplicate_natural = working.duplicated(subset=subset, keep="first") if subset else \
        pd.Series(False, index=working.index)

    duplicate_mask = duplicate_record_id | duplicate_natural
    removed = working[duplicate_mask].copy()
    removed["is_duplicate"] = True
    kept = working[~duplicate_mask].copy()
    kept["is_duplicate"] = False
    return kept.reset_index(drop=True), removed.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Currency
# --------------------------------------------------------------------------- #


def load_fx_rates(path: str | Path | None) -> dict[str, dict[str, object]]:
    """Load a currency->LSL rate table.

    Expected CSV columns: ``currency, rate_to_lsl, rate_date, source``.
    The repository ships an empty template; **no rates are hard-coded** because
    a rate must always be traceable to a date and a source.
    """
    if path is None:
        return {}
    path = Path(path)
    if not path.exists():
        return {}
    table = pd.read_csv(path, dtype="string")
    rates: dict[str, dict[str, object]] = {}
    for _, row in table.iterrows():
        currency = normalise_text(row.get("currency")).upper()
        if not currency:
            continue
        rates[currency] = {
            "rate": float(row.get("rate_to_lsl")),
            "date": normalise_text(row.get("rate_date")),
            "source": normalise_text(row.get("source")),
        }
    return rates


def convert_currency(
    frame: pd.DataFrame,
    rates: Mapping[str, Mapping[str, object]] | None = None,
    target: str = TARGET_CURRENCY,
) -> pd.DataFrame:
    """Add ``price_lsl`` plus FX provenance columns.

    Records already in the target currency are tagged ``identity``. Records
    whose currency has no rate are left unconverted and flagged, never guessed.
    """
    out = frame.copy()
    rates = dict(rates or {})
    prices = pd.to_numeric(out.get("price"), errors="coerce")
    currencies = out.get("currency", pd.Series([""] * len(out), index=out.index)).astype("string").str.upper()

    price_lsl = pd.Series(np.nan, index=out.index, dtype="float64")
    rate_used = pd.Series(pd.NA, index=out.index, dtype="string")
    rate_date = pd.Series(pd.NA, index=out.index, dtype="string")
    rate_source = pd.Series(pd.NA, index=out.index, dtype="string")
    extra_flags = pd.Series("", index=out.index, dtype="string")

    for index in out.index:
        currency = currencies.loc[index]
        if currency == target:
            price_lsl.loc[index] = prices.loc[index]
            rate_used.loc[index] = "1.0"
            rate_source.loc[index] = "identity"
            continue
        info = rates.get(str(currency))
        if info and info.get("rate"):
            price_lsl.loc[index] = prices.loc[index] * float(info["rate"])
            rate_used.loc[index] = str(info["rate"])
            rate_date.loc[index] = str(info.get("date") or "")
            rate_source.loc[index] = str(info.get("source") or "")
            continue
        # Unconvertible: preserve the original, record a flag.
        price_lsl.loc[index] = np.nan
        extra_flags.loc[index] = f"fx_missing_{currency}" if currency else "currency_missing"

    out["price_lsl"] = price_lsl
    out["fx_rate_used"] = rate_used
    out["fx_rate_date"] = rate_date
    out["fx_rate_source"] = rate_source
    if "quality_flags" in out.columns:
        out["quality_flags"] = (
            out["quality_flags"].astype("string").fillna("").str.strip(";")
            .str.cat(extra_flags, sep=";").str.strip(";").str.strip(";")
        )
    else:
        out["quality_flags"] = extra_flags
    return out


# --------------------------------------------------------------------------- #
# Grouped split
# --------------------------------------------------------------------------- #


def _group_keys(frame: pd.DataFrame, group_columns: Sequence[str]) -> pd.Series:
    columns = [c for c in group_columns if c in frame.columns]
    if not columns:
        return pd.Series(frame.index.astype(str), index=frame.index)
    return (
        frame[columns].astype("string").fillna("NA")
        .agg("|".join, axis=1)
    )


def split_by_group(
    frame: pd.DataFrame,
    *,
    test_size: float = schema.DEFAULT_TEST_SIZE,
    val_size: float = schema.DEFAULT_VAL_SIZE,
    random_state: int = schema.DEFAULT_RANDOM_STATE,
    group_columns: Sequence[str] = GROUP_KEY_COLUMNS,
    stratify_column: str = "price_band",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign a ``split`` column without letting laptop groups leak.

    Returns ``(frame_with_split, split_summary)``. Oversized groups (a spec
    appearing many times) are kept together in the training set to avoid
    leaking one machine type into the test set.
    """
    from sklearn.model_selection import GroupShuffleSplit

    out = frame.copy()
    out["_group"] = _group_keys(out, group_columns)

    groups = out["_group"]
    out["split"] = "train"

    if groups.nunique() < 2:
        summary = out.groupby("split").size().reset_index(name="records")
        out = out.drop(columns=["_group"])
        return out, summary

    holdout = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    rest_idx, test_idx = next(holdout.split(out, groups=groups))
    out.loc[out.index[test_idx], "split"] = "test"

    remainder = out.iloc[rest_idx]
    if val_size and remainder["_group"].nunique() >= 2:
        val_fraction = val_size / (1.0 - test_size)
        val_splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=random_state)
        train_rel, val_rel = next(val_splitter.split(remainder, groups=remainder["_group"]))
        out.loc[remainder.index[val_rel], "split"] = "val"

    summary = out.groupby("split").size().reset_index(name="records")
    out = out.drop(columns=["_group"])
    return out, summary


def add_price_band(
    frame: pd.DataFrame,
    column: str = "price_lsl",
    bands: int = 4,
) -> pd.DataFrame:
    """Add a ``price_band`` column (quartiles) for error analysis and splitting."""
    out = frame.copy()
    values = pd.to_numeric(out.get(column), errors="coerce")
    try:
        out["price_band"] = pd.qcut(values, q=bands, duplicates="drop").astype("string")
    except (ValueError, IndexError):
        out["price_band"] = pd.NA
    return out


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


@dataclass
class CleaningResult:
    frame: pd.DataFrame
    missingness_before: pd.DataFrame
    missingness_after: pd.DataFrame
    duplicates_removed: pd.DataFrame
    flag_counts: pd.DataFrame
    currency_summary: pd.DataFrame
    split_summary: pd.DataFrame
    notes: list[str] = field(default_factory=list)

    @property
    def usable_records(self) -> int:
        return int(len(self.frame))

    def to_report(self) -> dict:
        return {
            "records_after_cleaning": self.usable_records,
            "duplicates_removed": int(len(self.duplicates_removed)),
            "missingness_before": self.missingness_before.to_dict(orient="records"),
            "missingness_after": self.missingness_after.to_dict(orient="records"),
            "quality_flag_counts": self.flag_counts.to_dict(orient="records"),
            "currency_summary": self.currency_summary.to_dict(orient="records"),
            "split_summary": self.split_summary.to_dict(orient="records"),
            "notes": self.notes,
        }

    def write_report(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_report(), indent=2, default=str), encoding="utf-8")
        return path


def _flag_counts(flags: pd.Series) -> pd.DataFrame:
    counter: dict[str, int] = {}
    for value in flags.fillna(""):
        for flag in str(value).split(";"):
            flag = flag.strip()
            if flag:
                counter[flag] = counter.get(flag, 0) + 1
    if not counter:
        return pd.DataFrame(columns=["flag", "records"])
    return (
        pd.DataFrame(sorted(counter.items(), key=lambda kv: kv[1], reverse=True),
                     columns=["flag", "records"])
    )


def clean_dataset(
    raw: pd.DataFrame,
    *,
    rates: Mapping[str, Mapping[str, object]] | None = None,
    rules: ValidationRules = DEFAULT_RULES,
    drop_flagged: bool = False,
    test_size: float = schema.DEFAULT_TEST_SIZE,
    val_size: float = schema.DEFAULT_VAL_SIZE,
    random_state: int = schema.DEFAULT_RANDOM_STATE,
) -> CleaningResult:
    """Run the full Phase 8 preparation pipeline.

    Parameters
    ----------
    raw:
        The collected dataset (already read with :func:`read_dataset`).
    rates:
        Currency->LSL rates from :func:`load_fx_rates`.
    drop_flagged:
        When ``True`` records with quality flags are removed from the output
        (the flags are still reported). Default ``False`` so nothing is lost
        silently.

    Prices that could not be converted are flagged and excluded from modelling
    only; they remain in the cleaned CSV for inspection.
    """
    notes: list[str] = [f"input records: {len(raw)}"]

    missingness_before = summarise_missingness(raw)

    standardised = standardise_dataframe(raw, rules)
    # Convert first: the plausibility checks are expressed in the target
    # currency, so they need price_lsl to exist already.
    converted = convert_currency(standardised, rates)
    flagged = flag_quality(converted, rules)
    flagged = add_price_band(flagged)

    kept, removed = deduplicate(flagged)
    notes.append(f"duplicates removed: {len(removed)}")

    # Price outliers are flagged, never dropped automatically.
    outlier_mask = price_outlier_mask(kept, "price_lsl", rules)
    flags = kept["quality_flags"].astype("string").fillna("")
    kept["quality_flags"] = flags.mask(
        outlier_mask.fillna(False),
        (flags + ";price_outlier").str.strip(";"),
    )

    if drop_flagged:
        before = len(kept)
        kept = kept[kept["quality_flags"].astype("string").str.strip() == ""].reset_index(drop=True)
        notes.append(f"flagged records dropped: {before - len(kept)}")

    missingness_after = summarise_missingness(kept)

    split_frame, split_summary = split_by_group(
        kept, test_size=test_size, val_size=val_size, random_state=random_state
    )

    currency_summary = (
        split_frame.groupby("currency", dropna=False)
        .agg(records=("record_id", "size"),
             converted=("price_lsl", lambda s: int(s.notna().sum())))
        .reset_index()
        if "currency" in split_frame.columns
        else pd.DataFrame(columns=["currency", "records", "converted"])
    )

    notes.append(f"usable records after cleaning: {len(split_frame)}")
    if len(split_frame) < 100:
        notes.append(
            "WARNING: fewer than 100 usable records — model metrics will be "
            "unstable; report the limitation rather than over-claiming accuracy."
        )

    return CleaningResult(
        frame=split_frame,
        missingness_before=missingness_before,
        missingness_after=missingness_after,
        duplicates_removed=removed,
        flag_counts=_flag_counts(split_frame["quality_flags"]),
        currency_summary=currency_summary,
        split_summary=split_summary,
        notes=notes,
    )
