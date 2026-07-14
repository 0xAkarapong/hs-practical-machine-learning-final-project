"""Predict product prices from raw health & wellness listings.

This is the basic outcome entry point: load the trained XGBoost model and the
fitted name-embedding PCA, then predict THB prices for new products.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.feature_engineer import build_feature_table

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"

XGB_MODEL_PATH = MODELS_DIR / "xgboost.joblib"
SELECTED_FEATURES_PATH = MODELS_DIR / "selected_features.json"


def predict_prices(raw_df: pd.DataFrame) -> pd.Series:
    """Return predicted THB prices for a raw product listing table.

    The input DataFrame must contain the same columns as the cleaned raw table:
    `Id`, `Section`, `Name`, `Price`, `Total Sold`, `Total Reviews`, `Shop Location`.
    The `Price` column is ignored for prediction; it is only kept because the
    feature-engineering function expects the raw schema.

    ponytail: this is the minimal outcome entry point. It reuses the exact same
    feature-engineering logic as training so predictions are consistent. The
    selected-feature list and the XGBoost model are loaded from `models/`.
    """
    if "Price" not in raw_df.columns:
        raw_df = raw_df.copy()
        raw_df["Price"] = 0.0

    feature_table = build_feature_table(raw_df)
    X = feature_table.drop(columns=["log_price_thb"])

    selected_features = json.loads(SELECTED_FEATURES_PATH.read_text())
    missing = [f for f in selected_features if f not in X.columns]
    if missing:
        raise ValueError(f"Selected features missing from input: {missing}")
    X = X[selected_features]

    model = joblib.load(XGB_MODEL_PATH)
    log_predictions = model.predict(X)
    thb_predictions = np.expm1(log_predictions)
    return pd.Series(thb_predictions, index=raw_df.index, name="predicted_price_thb")


def main():
    """CLI entry point: predict prices for the cleaned raw table."""
    raw_path = PROJECT_ROOT / "dataset" / "health_and_wellness_no_outliers.csv"
    raw_df = pd.read_csv(raw_path)
    predictions = predict_prices(raw_df)

    output = pd.concat(
        [raw_df[["Id", "Name", "Section", "Shop Location"]], predictions],
        axis=1,
    )
    output_path = PROJECT_ROOT / "dataset" / "predictions.csv"
    output.to_csv(output_path, index=False)

    print(f"Predicted: {len(predictions):,} products")
    print(f"Saved:     {output_path.name} → dataset/")
    print(f"Mean:      {predictions.mean():,.2f} THB")
    print(f"Median:    {predictions.median():,.2f} THB")


if __name__ == "__main__":
    main()
