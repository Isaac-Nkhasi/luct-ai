# Data collection plan — used-laptop asking prices

**Goal:** build a real dataset that can train and evaluate a used-laptop
asking-price estimator for the LUCT Market.

**Non-negotiable rules**

1. **Never invent records.** The reported number of records must be the number
   actually collected. The pipeline will not fabricate rows for you.
2. **Keep asking prices separate from completed-sale prices** (`price_type`).
   They answer different questions and must never be mixed.
3. **Do not scrape sites whose terms prohibit it.** Only use sources you are
   permitted to use.
4. **Record the source and the collection date** for every record (`source`,
   `collection_date`).
5. **Voluntary contribution only.** Student contributions are opt-in with a
   clear purpose, and personal details are not collected.

## Where records come from

| Tier | Source | Notes |
| --- | --- | --- |
| A | **Public laptop-price dataset** used for initial model development | Record the dataset name/URL, licence and access date in `source`. Follow its licence. |
| B | **Permitted public listings** | Only where the site's terms allow it. Record the site as `source`. |
| C | **Local student contributions** | The most valuable for local realism. Voluntary, consented, no personal data. |

## What counts as one record

One **laptop listing sighting** with its specification and price. If the same
listing is seen twice, keep one and let the pipeline's duplicate detection catch
it — do not hand-edit numbers to force uniqueness.

## The collection form

Collect exactly the columns in [`docs/data_dictionary.md`](data_dictionary.md).
A ready-to-fill header is in
[`data/laptop_prices_template.csv`](../data/laptop_prices_template.csv).

Practical tips:

- Enter `price` **as stated** with its `currency`; do not pre-convert. The
  pipeline converts to LSL and records the rate.
- If battery health is unknown, leave it blank — it is optional.
- Put anything ambiguous in `notes` rather than guessing.

## Currency handling

1. Record the original `price` + `currency`.
2. Maintain [`data/fx_rates.csv`](../data/fx_rates.csv) with the columns
   `currency, rate_to_lsl, rate_date, source`.
3. The cleaning step writes `price_lsl`, `fx_rate_used`, `fx_rate_date` and
   `fx_rate_source` onto each record, so every conversion is traceable.
4. Rows whose currency has no rate are **flagged** (`fx_missing_<CUR>`) and
   excluded from modelling — never approximated.

> No FX rate is hard-coded in this repository. A rate must always be tied to a
> date and a source, because that is what makes the conversion defensible.

## Student contribution consent (short form)

> We are building a price-estimation tool for second-hand laptops on the LUCT
> Market. If you are willing, please share the laptop's make, model, specs,
> condition, battery health and the price you are asking or paid. This is
> voluntary, you can skip any field, and no personal details are recorded. The
> data is used only to estimate prices and will be reported in aggregate.

## Quality checklist before handing data to Phase 9

- [ ] File starts from the template and uses the exact column names.
- [ ] `record_id` is unique within the file.
- [ ] Every row has a `source` and a `collection_date`.
- [ ] `price_type` is filled for every row (`asking` or `completed`).
- [ ] No prices were manually converted — leave that to the pipeline.
- [ ] Prohibited sources were not used.

## Storage and privacy

- Store raw files under `data/raw/` (git-ignored) and cleaned output under
  `data/processed/`.
- The dataset contains **no personal data**: no names, no contact details, no
  seller identifiers. Keep it that way so the GDPR/data-protection position
  (Phase 12) stays simple.

## Reporting

When you report results, state:

- the **actual number of records collected**, and how many came from each tier;
- the collection date range;
- how many records were excluded and why (the cleaning report lists this);
- the limitation that asking prices are opinions, not realised sale values.
