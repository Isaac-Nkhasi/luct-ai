"""Phase 10 — FastAPI service that serves the laptop price estimate.

Endpoints
---------
* ``GET  /``        — service banner.
* ``GET  /health``  — liveness + whether the model loaded (used by Render).
* ``POST /predict`` — one estimate, in maloti (LSL).

The model is loaded once at startup. If it is missing the service still starts
(so the failure is visible in the logs and ``/health``) and ``/predict`` returns
503 rather than crashing the whole deploy.

Run locally::

    uvicorn api.main:app --reload
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:  # installed package (normal case)
    from luct_ai import serving
except ModuleNotFoundError:  # pragma: no cover - bare checkout without `pip install -e .`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from luct_ai import serving


class PredictionRequest(BaseModel):
    """One laptop's specification."""

    brand: str = Field(..., examples=["Lenovo"], description="Manufacturer.")
    processor: str = Field(..., examples=["Intel Core i5"], description="CPU family or raw string.")
    ram_gb: float = Field(..., ge=1, le=256, examples=[8])
    storage_gb: float = Field(..., ge=8, le=20000, examples=[256])
    storage_type: str = Field(..., examples=["SSD"])
    condition: str = Field(..., examples=["Good"])
    age_years: float | None = Field(None, ge=0, le=25)
    battery_health: float | None = Field(None, ge=0, le=100)
    model: str | None = Field(None, description="Model name; unused by the default model.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "brand": "Lenovo",
                    "processor": "Intel Core i5",
                    "ram_gb": 8,
                    "storage_gb": 256,
                    "storage_type": "SSD",
                    "condition": "Good",
                    "age_years": 5,
                    "battery_health": 80,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    predicted_price_lsl: float = Field(..., description="Estimated asking price in maloti.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.bundle = serving.load_bundle()
        app.state.load_error = None
    except Exception as exc:  # surfaced via /health instead of failing the deploy
        app.state.bundle = None
        app.state.load_error = str(exc)
    yield


app = FastAPI(
    title="LUCT Market — laptop price estimator",
    version="0.1.0",
    description="Estimates a second-hand laptop asking price in maloti (LSL).",
    lifespan=lifespan,
)

_origins = os.environ.get("ALLOWED_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins.split(",")] or ["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "LUCT Market price estimator", "docs": "/docs"}


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model_loaded": app.state.bundle is not None,
        "error": app.state.load_error,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    if app.state.bundle is None:
        raise HTTPException(status_code=503, detail="Model is not loaded; check /health.")
    price = serving.predict_price(app.state.bundle, payload.model_dump())
    return PredictionResponse(predicted_price_lsl=round(price, 2))
