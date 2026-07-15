"""Mean-predictor baseline for the health & wellness price regression."""

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor


def train_baseline(y_train: pd.Series) -> DummyRegressor:
    """Fit a mean-predictor baseline (DummyRegressor, strategy='mean').

    Gives a calibrated R²≈0 reference; the baseline's metrics are the floor the
    GradientBoostingRegressor must beat to justify its complexity. DummyRegressor
    ignores X, so we feed a placeholder column of zeros.
    """
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(np.zeros((len(y_train), 1)), y_train)
    return baseline
