# LUCT Market AI — Phase 8 (dataset) + Phase 9 (models) + Phase 10 (serving) + Phase 11 (metrics)

Code for building the used-laptop asking-price dataset, training/evaluating the
price estimator, and serving it as an HTTP API for the LUCT Market marketplace.

> **Data provenance.** The current dataset is a **public** dataset
> (`bitsnaps/laptops_specs`, Hugging Face, Apache-2.0, fetched 2026-11-11) run
> through the Phase 8 cleaner, plus a local collection kit for student-contributed
> rows. Counts are always reported as measured — nothing here invents records.

## Layout

```
luct-ai/
├── api/
│   ├── main.py                      # Phase 10 FastAPI service (/health, /metrics, /predict)
│   └── requirements.txt             # pinned serving dependencies
├── data/
│   ├── laptop_prices_template.csv   # header-only template for collection
│   ├── contributions/               # drop-off folder for the collection kit
│   └── fx_rates.csv                 # currency->LSL rates used by the cleaner
├── docs/
│   ├── data_dictionary.md           # every column, allowed values, flags
│   └── data_collection_plan.md      # how/were to collect, rules, consent
├── kit/
│   ├── contribute.html              # offline contribution form (no backend)
│   └── README.md                    # how to run a collection drive
├── models/
│   ├── laptop_price_model.joblib    # shipped model bundle (loaded by the API)
│   └── model_version.json           # version manifest (committed for Render)
├── notebooks/
│   ├── 01_data_cleaning.ipynb       # Phase 8 walkthrough
│   └── 02_model_training.ipynb      # Phase 9 walkthrough
├── src/luct_ai/
│   ├── schema.py                    # columns, vocabularies, validation rules
│   ├── cleaning.py                  # Phase 8 pipeline
│   ├── modeling.py                  # Phase 9 models + persistence
│   ├── serving.py                   # Phase 10 load + predict (reuses modeling)
│   ├── evaluate.py                  # metrics, error-by-band, learning curve, plots
│   └── pipeline.py                  # CLI orchestrating both phases
├── Dockerfile                       # image for the Phase 10 service
├── render.yaml                      # Render Blueprint (free tier)
└── tests/                           # unit tests (inline fixtures only)
```

## Quickstart

```powershell
cd luct-ai
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pip install -e .
.venv\Scripts\python -m pytest          # run the unit tests
```

## Running the pipeline

Once you have a collected CSV (start from `data/laptop_prices_template.csv`):

```powershell
.venv\Scripts\python -m luct_ai.pipeline clean --input data/raw/laptop_prices_raw.csv
.venv\Scripts\python -m luct_ai.pipeline train --input data/processed/laptop_prices_clean.csv
# or both at once
.venv\Scripts\python -m luct_ai.pipeline all --input data/raw/laptop_prices_raw.csv
```

If the input has no rows the pipeline stops with a clear message instead of
producing fake metrics.

### Outputs

| Path | Contents |
| --- | --- |
| `data/processed/laptop_prices_clean.csv` | cleaned dataset + derived columns |
| `reports/cleaning_report.md` / `.json` | missing values, duplicates, flags, FX, split |
| `reports/model_report.md` / `.json` | comparison tables, metrics, model version |
| `reports/figures/*.png` | actual-vs-predicted, error-by-band, learning curve, importance |
| `models/laptop_price_model.joblib` | best model (used by the Phase 10 API) |
| `models/model_version.json` | model version, evaluation date, dataset fingerprint |

## Phase 10 — serving the model

A small FastAPI service wraps the trained model. Feature engineering is reused
from `luct_ai.modeling`, so a served prediction goes through exactly the same
transformation as training; incoming text is normalised with the same
standardisers as the cleaner (send `"i5-8250U"` or `"lenovo"` and it still
resolves to the right category).

Run it locally:

```powershell
.venv\Scripts\python -m pip install -r api\requirements.txt
.venv\Scripts\python -m uvicorn api.main:app --port 8000
```

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Banner + link to the docs |
| `GET` | `/health` | `{status, model_loaded, error}` — used as the Render health check |
| `GET` | `/metrics` | The model version manifest + evaluation metrics, read from `models/model_version.json` |
| `POST` | `/predict` | Body = laptop specs, returns `{"predicted_price_lsl": <number>}` |
| `GET` | `/docs` | Auto-generated OpenAPI UI (also `/openapi.json`) |

`/metrics` is what the Phase 11 AI model dashboard in `security-admin/` charts.
It returns the best model name, target, training timestamp, library versions,
dataset rows and fingerprint, split sizes, and the per-model validation/test
MAE, RMSE and R². It answers `503` when no model or manifest is loaded, so the
dashboard can show a clear message instead of empty charts.

Example request:

```powershell
$body = @{ brand='Lenovo'; processor='Intel Core i5'; ram_gb=8; storage_gb=256;
           storage_type='SSD'; condition='Good'; age_years=3; battery_health=85 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType application/json -Body $body
```

The response is the estimate only (a number in LSL). Invalid payloads return
`422`; if the model failed to load, `/predict` returns `503` and `/health`
reports the error.

### Deploying (Render, free tier)

1. Push this repository to GitHub (the model bundle under `models/` is committed
   on purpose — Render deploys straight from git).
2. In Render: **New → Blueprint**, point it at the repo. `render.yaml` already
   defines a Docker web service named `luct-price-estimator` with
   `healthCheckPath: /health`.
3. After the first deploy, set `ALLOWED_ORIGINS` to the marketplace origin
   (defaults to `*` for testing). The model loads at container start.

`Dockerfile` pins the exact `numpy`/`pandas`/`scikit-learn`/`joblib` versions the
bundle was pickled with, so the joblib load is compatible. Docker could not be
build-tested on the authoring machine (Docker not installed) — verify on first
Render deploy.

## What was built (maps to the brief)

**Phase 8 — dataset**
- CSV schema + column template + data dictionary + collection plan.
- Duplicate removal, missing-value reporting, standardisation of brand/processor/
  RAM/storage/condition, impossible-spec and outlier flagging.
- Currency conversion to LSL that records the rate, date and source per record;
  unconvertible rows are flagged, never guessed.
- **Grouped** train/validation/test split so the same laptop specification cannot
  leak across the split.

**Phase 9 — models**
- Median-price baseline, linear regression and random forest, compared fairly.
- Tuning inside the training data only; the test set is touched once, at the end.
- MAE, RMSE, R², error by price band, learning curve, feature importance.
- Saved model bundle with a JSON version manifest (library versions, split sizes,
  metrics, dataset fingerprint, evaluation date).

**Phase 10 — serving**
- FastAPI service (`api/main.py`) that loads the saved bundle at startup and
  exposes `/health` + `/predict`, reusing the training-time feature pipeline.
- Input validation via Pydantic (impossible RAM/storage/ages rejected with `422`).
- Docker image + `render.yaml` Blueprint for a free-tier Render deploy.

**Phase 11 — model metrics endpoint**
- `GET /metrics` publishes the model manifest (best model, target, split sizes,
  library versions, dataset fingerprint) and the per-model validation/test
  MAE, RMSE and R² straight from `models/model_version.json`.
- Covered by `tests/test_api.py`; the figures are read, never re-typed, so the
  security-admin AI dashboard can never drift from the trained model.

## Honest reporting rules

- Report the **actual number** of records collected and the source breakdown.
- Asking prices are opinions, not realised sales; the two are kept separate.
- Do not claim an "accuracy %" — use MAE/RMSE (maloti) and R², and state the
  limitation when the sample is small.
