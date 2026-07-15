"""Mean-predictor baseline for the health & wellness price regression."""

import pandas as pd
from sklearn.dummy import DummyRegressor


def train_baseline(X_train: pd.DataFrame, y_train: pd.Series) -> DummyRegressor:
    """Fit a mean-predictor baseline (DummyRegressor, strategy='mean').

    Gives an R²≈0 reference; the baseline's metrics are the floor the
    GradientBoostingRegressor must beat to justify its complexity. Predicts the
    training mean regardless of X (DummyRegressor ignores features), so the R²
    lands near 0, not exactly 0 — it predicts mean(y_train), not mean(y_test),
    which is the leak-free version of an R²=0 reference.
    """
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)
    return baseline
