"""Train/test split for the feature-engineered health & wellness table."""

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "dataset" / "health_and_wellness_feature_engineered.csv"
TRAIN_PATH = PROJECT_ROOT / "dataset" / "train.csv"
TEST_PATH = PROJECT_ROOT / "dataset" / "test.csv"

TARGET_COLUMN = "log_price_thb"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def split_data(
    df: pd.DataFrame, test_size: float = TEST_SIZE, seed: int = RANDOM_STATE
):
    """Split rows into train/test, stratified on the target binned into deciles.

    ponytail: numpy-seeded split, not sklearn — sklearn isn't a dep yet; swap to
    sklearn.model_selection.train_test_split once the modeling step adds it.
    Stratification on binned log-price keeps the price distribution balanced across
    splits; raw regression targets can't be stratified directly.
    """
    rng = np.random.default_rng(seed)
    price_bins = pd.qcut(df[TARGET_COLUMN], q=10, duplicates="drop")
    train_indices, test_indices = [], []
    for _, stratum_indices in df.groupby(price_bins, observed=True).indices.items():
        stratum_indices = np.asarray(stratum_indices)
        rng.shuffle(stratum_indices)
        train_cutoff = int(len(stratum_indices) * (1 - test_size))
        train_indices.append(stratum_indices[:train_cutoff])
        test_indices.append(stratum_indices[train_cutoff:])
    train_indices = np.concatenate(train_indices)
    test_indices = np.concatenate(test_indices)
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)
    return df.iloc[train_indices], df.iloc[test_indices]


def run_split():
    feature_table = pd.read_csv(DATA_PATH)
    train_table, test_table = split_data(feature_table)
    train_table.to_csv(TRAIN_PATH, index=False)
    test_table.to_csv(TEST_PATH, index=False)
    loaded_rows, loaded_cols = feature_table.shape
    print(f"Loaded: {loaded_rows:,} rows × {loaded_cols:,} cols from {DATA_PATH.name}")
    print(f"Train:  {train_table.shape[0]:,} rows → {TRAIN_PATH.name}")
    print(f"Test:   {test_table.shape[0]:,} rows → {TEST_PATH.name}")
    train_target_mean = train_table[TARGET_COLUMN].mean()
    test_target_mean = test_table[TARGET_COLUMN].mean()
    print(f"Target '{TARGET_COLUMN}' — train mean {train_target_mean:.4f}, "
          f"test mean {test_target_mean:.4f}")


if __name__ == "__main__":
    run_split()