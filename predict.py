"""Predict product prices from raw health & wellness listings.

This is the basic outcome entry point: load the trained XGBoost model and the
fitted name-embedding PCA, then predict THB prices for new products.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src._02_feature_engineer import PCA_PATH, build_predict_table

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"

XGB_MODEL_PATH = MODELS_DIR / "xgboost.joblib"
SELECTED_FEATURES_PATH = MODELS_DIR / "selected_features.json"
DEFAULT_INPUT = PROJECT_ROOT / "dataset" / "health_and_wellness_no_outliers.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "dataset" / "predictions.csv"


def predict_prices(raw_df: pd.DataFrame) -> pd.Series:
    """Return predicted THB prices for a raw product listing table.

    The input DataFrame must contain the same columns as the cleaned raw table:
    `Id`, `Section`, `Name`, `Price`, `Total Sold`, `Total Reviews`, `Shop Location`.
    The `Price` column is ignored for prediction; it is only kept because the
    feature-engineering function expects the raw schema.

    ponytail: minimal outcome entry point. Reuses the fitted PCA from
    models/name_pca.joblib (transform, not refit) so train/predict share one basis,
    skips the unknown target column, and hits the embedding cache on repeat runs.
    The feature matrix is reindexed to the exact selected-feature set the model was
    trained on (absent one-hot columns fill with 0 — correct for one-hot absence).
    """
    if "Price" not in raw_df.columns:
        raw_df = raw_df.copy()
        raw_df["Price"] = 0.0

    pca = joblib.load(PCA_PATH)
    feature_table = build_predict_table(raw_df, pca=pca)

    selected_features = json.loads(SELECTED_FEATURES_PATH.read_text())
    X = feature_table.reindex(columns=selected_features, fill_value=0)

    model = joblib.load(XGB_MODEL_PATH)
    log_predictions = model.predict(X)
    thb_predictions = np.expm1(log_predictions)
    return pd.Series(thb_predictions, index=raw_df.index, name="predicted_price_thb")


def main():
    """CLI entry point: predict prices for the cleaned raw table or a custom CSV.

    Input CSV schema: Id, Section, Name, Price, Total Sold, Total Reviews,
    Shop Location. `Price` is optional (ignored for prediction; filled with 0
    if absent). Defaults predict the cleaned raw table to dataset/predictions.csv.
    """
    parser = argparse.ArgumentParser(
        description="Predict THB prices for health & wellness product listings.",
    )
    parser.add_argument(
        "input",
        nargs="?",
        default=str(DEFAULT_INPUT),
        help="input CSV with raw listing schema (default: cleaned raw table)",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=str(DEFAULT_OUTPUT),
        help="output predictions CSV (default: dataset/predictions.csv)",
    )
    args = parser.parse_args()

    raw_df = pd.read_csv(args.input)
    predictions = predict_prices(raw_df)

    output = pd.concat(
        [raw_df[["Id", "Name", "Section", "Shop Location"]], predictions],
        axis=1,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    print(f"Input:     {args.input}")
    print(f"Predicted: {len(predictions):,} products")
    print(f"Saved:     {output_path}")
    print(f"Mean:      {predictions.mean():,.2f} THB")
    print(f"Median:    {predictions.median():,.2f} THB")


if __name__ == "__main__":
    main()
