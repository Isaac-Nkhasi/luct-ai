# Used-laptop dataset — data dictionary

This is the contract for the Phase 8 dataset. The machine-readable source of
truth is [`src/luct_ai/schema.py`](../src/luct_ai/schema.py); this document
explains it in plain language.

- **Target currency:** all prices are converted to **LSL** (Lesotho loti / maloti)
  for modelling, because the marketplace is denominated in maloti.
- **A record = one laptop listing sighting**, not one seller and not one sale.
- Start every new file from [`data/laptop_prices_template.csv`](../data/laptop_prices_template.csv).

## Collected columns

| Column | Type | Required | Allowed values / range | Description | Example |
| --- | --- | --- | --- | --- | --- |
| `record_id` | text | yes | `LAP001`, `LAP002`, … (unique) | Unique identifier for the row. Never reused. | `LAP001` |
| `brand` | text | yes | see **Brands** below | Manufacturer. Standardised during cleaning. | `Lenovo` |
| `model` | text | yes | free text | Model name/number as advertised. | `ThinkPad T480` |
| `processor` | text | yes | see **Processor families** below | CPU, standardised to a family. | `Intel Core i5` |
| `ram_gb` | number | yes | 1–256 GB | Installed memory in GB. | `8` |
| `storage_gb` | number | yes | 8–20000 GB | Drive capacity in GB (`1TB` is fine, it is parsed). | `256` |
| `storage_type` | text | yes | `SSD`, `HDD`, `eMMC`, `SSD+HDD`, `Unknown` | Drive technology, standardised. | `SSD` |
| `age_years` | number | yes | 0–25 | Age in years at collection. | `5` |
| `condition` | text | yes | `New`, `Like New`, `Good`, `Fair`, `Poor` | Cosmetic/functional grade, standardised. | `Good` |
| `battery_health` | number | no | 0–100 (%) | Battery health % where the seller can report it. | `80` |
| `price` | number | yes | > 0, in `currency` | The observed price **as stated**. | `4500` |
| `currency` | text | yes | ISO-4217 code, e.g. `LSL`, `ZAR`, `USD` | Currency of `price`. | `LSL` |
| `source` | text | yes | free text; be specific | Where/how the record was obtained (see collection plan). | `Student contribution` |
| `collection_date` | date | yes | `YYYY-MM-DD` | Date the record was collected. | `2026-02-01` |
| `price_type` | text | yes | `asking`, `completed` | Asking prices and completed-sale prices are **never mixed**. | `asking` |
| `notes` | text | no | free text | Anything that affects interpretation. | `cracked bezel` |

## Columns the pipeline adds (never collected by hand)

| Column | Meaning |
| --- | --- |
| `price_lsl` | `price` converted to LSL using `fx_rate_used`. |
| `fx_rate_used` | Multiplier applied (`1.0` + source `identity` when already LSL). |
| `fx_rate_date` | Date of the FX rate used. |
| `fx_rate_source` | Where the rate came from (e.g. a central-bank page). |
| `price_band` | Quartile band of `price_lsl`, used for error-by-band reporting. |
| `quality_flags` | `;`-separated list of issues found (see glossary below). Empty = clean. |
| `is_duplicate` | `True` for rows removed as duplicates. |
| `split` | `train` / `val` / `test`, assigned by **laptop group** so specs cannot leak. |

## Controlled vocabularies

**Brands:** Acer, Apple, Asus, Dell, Fujitsu, HP, Lenovo, Microsoft, MSI, Other,
Samsung, Toshiba.

**Processor families:** Intel Core i3/i5/i7/i9, Intel Celeron, Intel Pentium,
AMD Ryzen 3/5/7/9, AMD Athlon, Apple M1/M2/M3, Other.

**Storage types:** SSD, HDD, eMMC, SSD+HDD, Unknown.

**Condition (worst → best):** Poor, Fair, Good, Like New, New.

Aliases are mapped automatically (e.g. `excellent` → `Like New`, `i5-8250U` →
`Intel Core i5`, `IBM/Lenovo` → `Lenovo`). Anything that cannot be mapped is
**flagged**, never silently relabelled.

## Quality-flag glossary

| Flag | Meaning |
| --- | --- |
| `missing_<column>` | A critical value was blank. |
| `implausible_ram` / `implausible_storage` | Outside the plausible GB range. |
| `implausible_age` | Outside 0–25 years. |
| `implausible_battery_health` | Outside 0–100 %. |
| `implausible_price` | Outside the plausible LSL band. |
| `brand_unmapped` / `processor_unmapped` / `condition_unmapped` / `storage_type_unmapped` | Value is not in the controlled vocabulary. |
| `price_type_unmapped` | Not `asking` or `completed`. |
| `fx_missing_<CUR>` | No FX rate available; price could not be converted. |
| `currency_missing` | `currency` was blank. |
| `price_outlier` | Statistically far above the rest of the sample (IQR fence / median multiplier). |

## Rules baked into the schema

1. **Asking ≠ completed.** `price_type` keeps them apart; the estimator trains on
   asking prices unless a filter says otherwise.
2. **No invented records.** The pipeline refuses to run — and clearly says so — when
   a file has no rows.
3. **Unconvertible prices are flagged, not guessed.**
4. **Duplicates are removed, not averaged.**
5. **Splits are grouped**, so the same laptop specification never appears in both
   training and test data.
