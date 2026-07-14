import numpy as np
import pandas as pd

from src.baseline import train_baseline
from src.train import evaluate_model, train_model


def test_train_and_evaluate_produces_finite_metrics():
    rng = np.random.default_rng(0)
    X_train = pd.DataFrame(rng.normal(size=(200, 4)), columns=["a", "b", "c", "d"])
    y_train = pd.Series(
        X_train["a"] * 2.0 + rng.normal(scale=0.1, size=200), name="log_price_thb"
    )
    X_test = pd.DataFrame(rng.normal(size=(50, 4)), columns=["a", "b", "c", "d"])
    y_test = pd.Series(
        X_test["a"] * 2.0 + rng.normal(scale=0.1, size=50), name="log_price_thb"
    )

    model = train_model(X_train, y_train)
    preds = model.predict(X_test)

    assert np.all(np.isfinite(preds))
    metrics = evaluate_model(model, X_test, y_test)

    expected_keys = {"rmse_log", "mae_log", "r2_log", "rmse_thb", "mae_thb"}
    assert set(metrics) == expected_keys
    assert all(np.isfinite(v) for v in metrics.values())
    # ponytail: don't assert r2_log >= 0 in isolation — a trivial synthetic
    # target can fit poorly without signaling a bug; finiteness + the expected
    # keys are enough to catch a broken pipeline.

    baseline = train_baseline(y_train)
    baseline_metrics = evaluate_model(baseline, X_test, y_test)
    assert all(np.isfinite(v) for v in baseline_metrics.values())
    # The model uses informative features; it must beat the mean-predictor.
    assert metrics["r2_log"] > baseline_metrics["r2_log"]
    assert metrics["rmse_log"] < baseline_metrics["rmse_log"]
