"""FastAPI bridge: expose the trained price model to the Next.js frontend.

Loads the fitted PCA + XGBoost model + the sentence encoder once at import (warm
start), then serves single-listing predictions, model metrics, and form metadata.
Reuses predict.predict_prices and models/metrics.json — no logic duplicated.

The encoder is held resident so a per-request single listing doesn't reload the
~278M model; threading it through build_name_embeddings leaves the batch
train/predict CLI paths untouched.
"""

import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from sentence_transformers import SentenceTransformer

from predict import XGB_MODEL_PATH, predict_prices
from src._02_feature_engineer import PCA_PATH
from src.common.embeddings import MODEL_NAME

PROJECT_ROOT = Path(__file__).resolve().parent
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"
RAW_TABLE_PATH = PROJECT_ROOT / "dataset" / "health_and_wellness_no_outliers.csv"
SAMPLE_INPUT_PATH = PROJECT_ROOT / "dataset" / "sample_input.csv"
_CACHE_DIR = Path.home() / ".cache" / "sentence_transformers"

# Load once at import so every request is warm. If the models are missing (e.g.
# the volume isn't populated / pipeline hasn't trained), the import fails loudly —
# which is the correct signal for `uvicorn api:app`.
_PCA = joblib.load(PCA_PATH)
_MODEL = joblib.load(XGB_MODEL_PATH)
_ENCODER = SentenceTransformer(MODEL_NAME, cache_folder=str(_CACHE_DIR))

app = FastAPI(title="Health & Wellness Price API", version="0.1.0")


class Listing(BaseModel):
    """Raw-schema product listing. Field names match the cleaned CSV columns."""

    model_config = ConfigDict(populate_by_name=True)

    Id: str | None = None
    Section: str | None = None
    Name: str | None = ""
    Price: float | None = 0.0
    total_sold: str | float | None = Field(None, alias="Total Sold")
    total_reviews: float | None = Field(0.0, alias="Total Reviews")
    shop_location: str | None = Field(None, alias="Shop Location")


def _listing_df(listing: Listing) -> pd.DataFrame:
    """Wrap one listing in the raw-schema DataFrame predict_prices expects."""
    return pd.DataFrame(
        [
            {
                "Id": listing.Id if listing.Id is not None else 0,
                "Section": listing.Section,
                "Name": listing.Name,
                "Price": listing.Price if listing.Price is not None else 0.0,
                "Total Sold": listing.total_sold
                if listing.total_sold is not None
                else 0,
                "Total Reviews": (
                    listing.total_reviews if listing.total_reviews is not None else 0
                ),
                "Shop Location": listing.shop_location,
            }
        ]
    )


@app.get("/")
def root() -> dict:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> JSONResponse:
    """Return models/metrics.json verbatim (baseline/gbdt/xgboost + CV)."""
    return JSONResponse(json.loads(METRICS_PATH.read_text()))


@lru_cache(maxsize=1)
def _meta() -> dict:
    """Unique Section + Shop Location values from the raw cleaned table.

    Read from the source CSV so the form dropdowns stay correct as the dataset
    changes, instead of hardcoding a list that drifts. Cached once.
    """
    df = pd.read_csv(RAW_TABLE_PATH)
    return {
        "sections": sorted(df["Section"].dropna().unique().tolist()),
        "shop_locations": sorted(df["Shop Location"].dropna().unique().tolist()),
    }


@app.get("/meta")
def meta() -> dict:
    return _meta()


@app.post("/predict")
def predict(listing: Listing) -> dict:
    """Predict a THB price for one product listing."""
    predictions = predict_prices(_listing_df(listing), encoder=_ENCODER)
    price = float(predictions.iloc[0])
    return {"predicted_price_thb": round(price, 2)}


if __name__ == "__main__":
    # Self-check: exercise the real predict path on dataset/sample_input.csv with
    # the resident encoder; no HTTP client needed.
    sample = pd.read_csv(SAMPLE_INPUT_PATH)
    preds = predict_prices(sample, encoder=_ENCODER)
    assert len(preds) == len(sample), (len(preds), len(sample))
    assert (preds > 0).all(), preds
    print(
        f"api.py self-check OK: {len(preds)} predictions, "
        f"mean {preds.mean():,.2f} THB, median {preds.median():,.2f} THB"
    )
