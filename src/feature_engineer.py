"""Reproducible feature engineering from the cleaned health & wellness table."""

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.embeddings import build_name_embeddings
from src.regions import PROVINCE_TO_REGION
from src.split_data import RANDOM_STATE, TARGET_COLUMN

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "dataset" / "health_and_wellness_no_outliers.csv"
OUTPUT_PATH = PROJECT_ROOT / "dataset" / "health_and_wellness_feature_engineered.csv"
OUTPUT_PATH_NO_EMB = (
    PROJECT_ROOT
    / "dataset"
    / "health_and_wellness_feature_engineered_no_embeddings.csv"
)
MODELS_DIR = PROJECT_ROOT / "models"
PCA_PATH = MODELS_DIR / "name_pca.joblib"

# Default: use region-level shop location instead of province-level.
USE_REGION = True


def clean_category_value(value: str) -> str:
    """Create readable, stable dummy-column suffixes from category values."""
    value = "Unknown" if pd.isna(value) else str(value).strip()
    value = re.sub(r"\W+", "_", value, flags=re.UNICODE).strip("_")
    return value or "Unknown"


def map_to_region(cleaned_province: str) -> str:
    """Return the region for a cleaned province name; unknown provinces fall back."""
    return PROVINCE_TO_REGION.get(cleaned_province, "unknown")


def build_feature_table(
    df: pd.DataFrame, use_region: bool = USE_REGION
) -> pd.DataFrame:
    """Build the modeling table from the cleaned raw table.

    ponytail: mirrors the logic in notebooks/03 feature_engineer.ipynb so the
    pipeline can be reproduced inside the Docker image (notebooks/ is excluded
    from the image). The province-level one-hot is replaced by region-level when
    use_region=True; this reduces sparse 72-column province dummies to ~9
    region columns while keeping the location signal.
    """
    df = df.copy()
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["Total Reviews"] = pd.to_numeric(df["Total Reviews"], errors="coerce").fillna(0)
    df["Section"] = df["Section"].fillna("Unknown")
    df["Shop Location"] = df["Shop Location"].fillna("Unknown")
    df["Name"] = df["Name"].fillna("")

    features = pd.DataFrame(index=df.index)
    features[TARGET_COLUMN] = np.log1p(df["Price"])
    features["log_total_reviews"] = np.log1p(df["Total Reviews"])
    features["has_reviews"] = (df["Total Reviews"] > 0).astype(int)
    features["name_length"] = df["Name"].str.len()
    features["name_word_count"] = df["Name"].str.split().str.len().fillna(0).astype(int)

    print("Encoding product names to sentence embeddings...")
    name_embeddings, _, name_pca = build_name_embeddings(df["Name"])
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(name_pca, PCA_PATH)
    print(f"Saved:  {PCA_PATH.name} → {PCA_PATH.parent.name}/")

    categorical_df = pd.DataFrame(
        {
            "section": df["Section"].map(clean_category_value),
        },
        index=df.index,
    )

    if use_region:
        categorical_df["shop_region"] = df["Shop Location"].map(
            lambda x: map_to_region(clean_category_value(x))
        )
    else:
        categorical_df["shop_location"] = df["Shop Location"].map(clean_category_value)

    dummy_features = pd.get_dummies(
        categorical_df,
        columns=categorical_df.columns.tolist(),
        prefix=categorical_df.columns.tolist(),
        dtype=int,
    )

    feature_df = pd.concat([features, name_embeddings, dummy_features], axis=1)
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return feature_df


def drop_embedding_columns(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Return the same feature table without the sentence-embedding columns."""
    emb_cols = [c for c in feature_df.columns if c.startswith("name_semantic_pca_")]
    return feature_df.drop(columns=emb_cols)


def run_feature_engineer() -> pd.DataFrame:
    """Load the cleaned table, engineer features, and save both modeling tables."""
    raw_df = pd.read_csv(INPUT_PATH)
    feature_df = build_feature_table(raw_df)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(OUTPUT_PATH, index=False)

    feature_df_no_emb = drop_embedding_columns(feature_df)
    feature_df_no_emb.to_csv(OUTPUT_PATH_NO_EMB, index=False)

    print(f"Engineered: {feature_df.shape[0]:,} rows × {feature_df.shape[1]:,} cols")
    print(f"Saved:      {OUTPUT_PATH.name} → dataset/")
    print(
        f"Saved:      {OUTPUT_PATH_NO_EMB.name} → dataset/ "
        f"({feature_df_no_emb.shape[1]:,} cols)"
    )
    print(f"Target:     {TARGET_COLUMN}, seed={RANDOM_STATE}")
    return feature_df


if __name__ == "__main__":
    run_feature_engineer()
