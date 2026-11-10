# Local collection kit

Everything needed to collect **real** used-laptop records from Lesotho instead of
relying on foreign data. Read `../docs/data_collection_plan.md` for the full rules;
this file is the practical how-to.

**Two rules that matter most**

1. **Never invent records.** The number you report must be the number actually
   collected. The intake script refuses to run on an empty folder rather than
   produce fake rows.
2. **Never mix asking and completed prices.** Every row carries a `price_type`.

## What's in the kit

| File | Purpose |
| --- | --- |
| `contribute.html` | Offline form a contributor fills in. Exports a CSV in the exact schema. |
| `../scripts/collect_local.py` | Validates and merges contribution CSVs into one local dataset. |
| `../data/laptop_prices_template.csv` | Blank header for collecting by hand in a spreadsheet. |
| `../docs/data_dictionary.md` | Every column, allowed value and quality flag. |

## Run a collection drive

1. **Host or share the form.** `contribute.html` is a single file with no backend
   and no internet calls. Either:
   - open it locally and let contributors fill it at a shared laptop, or
   - drop it on any free static host (GitHub Pages, Netlify drop, an LMS file share).

   It collects **laptop information only** — no names, emails, student numbers or
   other personal data. Contributors tick a consent box; nothing is uploaded, they
   download a CSV and send it to you.

2. **Collect the CSVs.** Put every contribution file in `data/contributions/`
   (create the folder if it does not exist). One file per contributor is fine.

3. **Merge and validate.**

   ```powershell
   .venv\Scripts\python scripts/collect_local.py
   ```

   You get an honest count: how many rows were offered, accepted and rejected,
   plus a reason for every rejection. `record_id` values are assigned here.

4. **Clean and model the local data.**

   ```powershell
   .venv\Scripts\python -m luct_ai.pipeline all --input data/raw/laptop_prices_local.csv
   ```

   To model local and public data together, concatenate the two raw files first —
   the cleaner's `source` column keeps the provenance of each row separate.

## What the intake step rejects

A row is refused (with a reason, never silently changed) when it is missing a
required field, has a non-numeric or non-positive price, has RAM/storage/age
outside the plausible range, has a `price_type` other than `asking`/`completed`,
or has a `collection_date` that is not `YYYY-MM-DD`.

Prices are **not** converted here — leave the currency as stated and let the
pipeline record the rate it used. See `../data/fx_rates.csv`.

## Privacy

Store raw contributions under `data/contributions/` and `data/raw/`; both are
git-ignored and hold **no personal data**. Keep it that way so the data-protection
position (Phase 12) stays simple.
