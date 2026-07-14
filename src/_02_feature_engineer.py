"""Reproducible feature engineering from the cleaned health & wellness table."""

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.common.embeddings import EMBEDDING_DIM, build_name_embeddings
from src.common.regions import PROVINCE_TO_REGION
from src._03_split_data import RANDOM_STATE, TARGET_COLUMN

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


# --- Text-derived features from Name + cleaned Total Sold ---------------------
# ponytail: cheap regex signals the sentence embeddings don't fully capture —
# dosage magnitude, unit/weight presence, authenticity/premium keywords, bundle
# ("เซต/แถม/PRE-SALE") flags, and Thai/English ratio. RFECV prunes whatever is
# noise. Also parses the messy `Total Sold` Thai string ("4,819 ชิ้น") that the
# pipeline never used — corr(log_total_sold, log_price) ≈ -0.22, a real signal.
_DIGIT_TOKEN = re.compile(r"\d[\d,]*")
_THAI_CHARS = re.compile(r"[฀-๿]")
_WEIGHT_UNITS = re.compile(r"mg|มก\.?|กรัม|ml|มล\.?", re.IGNORECASE)
_COUNT_UNITS = re.compile(
    r"แคปซูล|เม็ด|กล่อง|ซอง|ขวด|แผง|ชิ้น|tablets|capsules|pcs", re.IGNORECASE
)
_AUTHENTIC = re.compile(r"ของแท้|แท้|original|genuine", re.IGNORECASE)
_BUNDLE = re.compile(r"เซต|set|แถม|pre[-\s]?sale|value", re.IGNORECASE)


def _parse_total_sold(value) -> float:
    """Turn '4,819 ชิ้น' / '7 ชิ้น' / NaN into a float count (0 if missing)."""
    if pd.isna(value):
        return 0.0
    match = _DIGIT_TOKEN.search(str(value))
    return float(match.group().replace(",", "")) if match else 0.0


def _name_text_features(names: pd.Series) -> pd.DataFrame:
    """Regex-based features from the product Name (price-relevant, embedding-blind)."""
    s = names.fillna("").astype(str)

    def _max_qty(text: str) -> float:
        nums = [_to_float(t) for t in _DIGIT_TOKEN.findall(text)]
        return max(nums) if nums else 0.0

    qty = s.apply(_max_qty)
    non_space = s.str.replace(r"\s", "", regex=True)
    return pd.DataFrame(
        {
            "name_qty_log": np.log1p(qty),
            "name_has_weight_unit": s.str.contains(_WEIGHT_UNITS, regex=True).astype(
                int
            ),
            "name_has_count_unit": s.str.contains(_COUNT_UNITS, regex=True).astype(int),
            "name_is_authentic": s.str.contains(_AUTHENTIC, regex=True).astype(int),
            "name_has_bundle": s.str.contains(_BUNDLE, regex=True).astype(int),
            "name_thai_ratio": non_space.apply(
                lambda t: (len(_THAI_CHARS.findall(t)) / len(t)) if t else 0.0
            ),
        },
        index=names.index,
    )


def _to_float(token: str) -> float:
    try:
        return float(token.replace(",", ""))
    except ValueError:
        return 0.0


def build_feature_table(
    df: pd.DataFrame,
    use_region: bool = USE_REGION,
    include_interactions: bool = False,
    embedding_dim: int = EMBEDDING_DIM,
) -> pd.DataFrame:
    """Build the modeling table from the cleaned raw table.

    ponytail: mirrors the logic in notebooks/03 feature_engineer.ipynb so the
    pipeline can be reproduced inside the Docker image (notebooks/ is excluded
    from the image). The province-level one-hot is replaced by region-level when
    use_region=True; this reduces sparse 72-column province dummies to ~9
    region columns while keeping the location signal.

    include_interactions (default False) adds section × name_word_count columns
    from Summary.md's next-steps — off by default so the deployed feature set is
    unchanged; turn on for an ablation. embedding_dim makes the PCA component count
    configurable (default 16); a 32-dim ablation is a one-line change.
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
    features["log_total_sold"] = np.log1p(df["Total Sold"].apply(_parse_total_sold))

    print("Encoding product names to sentence embeddings...")
    name_embeddings, _, name_pca = build_name_embeddings(
        df["Name"], embedding_dim=embedding_dim
    )
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(name_pca, PCA_PATH)
    print(f"Saved:  {PCA_PATH.name} → {PCA_PATH.parent.name}/")

    name_text = _name_text_features(df["Name"])

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

    feature_df = pd.concat(
        [features, name_embeddings, name_text, dummy_features], axis=1
    )
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    if include_interactions:
        feature_df = add_interaction_features(feature_df)
    return feature_df


def add_interaction_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Add section × name_word_count interaction columns to an engineered table.

    ponytail: the canonical 'section × title length' interaction from Summary.md's
    next-steps. Builds one column per section dummy (section_<x> * name_word_count)
    so RFECV can prune if the interactions don't help. Operates on the already-
    engineered table so no sentence re-encoding is needed.
    """
    if "name_word_count" not in feature_df.columns:
        return feature_df
    section_cols = [c for c in feature_df.columns if c.startswith("section_")]
    if not section_cols:
        return feature_df
    interactions = {
        f"x_{col}_wordcount": feature_df[col] * feature_df["name_word_count"]
        for col in section_cols
    }
    interaction_df = pd.DataFrame(interactions, index=feature_df.index)
    return pd.concat([feature_df, interaction_df], axis=1)


def build_predict_table(
    df: pd.DataFrame,
    pca: object,
    use_region: bool = USE_REGION,
    embedding_dim: int = EMBEDDING_DIM,
) -> pd.DataFrame:
    """Build the feature table for prediction.

    Reuses a fitted PCA and skips the target column.

    ponytail: predict-time counterpart to build_feature_table. It omits
    log_price_thb (the target, unknown at predict time) and transforms name
    embeddings with the already-fitted PCA from models/name_pca.joblib instead of
    refitting — this fixes the latent train/predict PCA-basis mismatch and removes
    the refit cost. `Price` is coerced only to satisfy the raw schema; it is unused.
    """
    df = df.copy()
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df["Total Reviews"] = pd.to_numeric(df["Total Reviews"], errors="coerce").fillna(0)
    df["Section"] = df["Section"].fillna("Unknown")
    df["Shop Location"] = df["Shop Location"].fillna("Unknown")
    df["Name"] = df["Name"].fillna("")

    features = pd.DataFrame(index=df.index)
    features["log_total_reviews"] = np.log1p(df["Total Reviews"])
    features["has_reviews"] = (df["Total Reviews"] > 0).astype(int)
    features["name_length"] = df["Name"].str.len()
    features["name_word_count"] = df["Name"].str.split().str.len().fillna(0).astype(int)
    features["log_total_sold"] = np.log1p(df["Total Sold"].apply(_parse_total_sold))

    name_embeddings, _, _ = build_name_embeddings(
        df["Name"], embedding_dim=embedding_dim, pca=pca
    )
    name_text = _name_text_features(df["Name"])

    categorical_df = pd.DataFrame(
        {"section": df["Section"].map(clean_category_value)},
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

    feature_df = pd.concat(
        [features, name_embeddings, name_text, dummy_features], axis=1
    )
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
