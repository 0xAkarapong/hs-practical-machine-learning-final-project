import numpy as np
import pandas as pd

from src._05_baseline import train_baseline
from src._04_feature_selection import select_features
from src.common.metrics import cross_val_metrics, evaluate_model
from src._06_train import plot_result, train_gbdt


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

    model = train_gbdt(X_train, y_train)
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


def test_cross_val_metrics_returns_finite_mean_std():
    # ponytail: self-check that cross_val_metrics gives finite mean±std on a
    # synthetic linear target — the smallest thing that fails if the CV loop breaks.
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(150, 3)), columns=["a", "b", "c"])
    y = pd.Series(X["a"] * 2.0 + rng.normal(scale=0.1, size=150), name="log_price_thb")

    cv = cross_val_metrics(train_gbdt(X, y), X, y, k=5)

    assert cv["k"] == 5
    for key in ("r2_log_mean", "r2_log_std", "rmse_log_mean", "rmse_log_std"):
        assert np.isfinite(cv[key])


def test_select_features_keeps_a_subset():
    rng = np.random.default_rng(1)
    X_train = pd.DataFrame(
        rng.normal(size=(120, 8)), columns=[f"f{i}" for i in range(8)]
    )
    y_train = pd.Series(
        X_train["f0"] * 3.0 - X_train["f1"] * 1.5 + rng.normal(scale=0.2, size=120),
        name="log_price_thb",
    )

    selected = select_features(X_train, y_train, min_features=3, cv=3)

    assert isinstance(selected, list)
    assert len(selected) >= 3
    assert len(selected) <= X_train.shape[1]
    assert all(col in X_train.columns for col in selected)


def test_plot_result_creates_a_file(tmp_path):
    rng = np.random.default_rng(2)
    y_actual = pd.Series(rng.normal(size=30), name="log_price_thb")
    y_pred = y_actual.to_numpy() + rng.normal(scale=0.1, size=30)
    metrics = {
        "rmse_log": 0.1,
        "mae_log": 0.08,
        "r2_log": 0.95,
        "rmse_thb": 10.0,
        "mae_thb": 8.0,
    }
    save_path = tmp_path / "plot.png"

    result_path = plot_result(y_actual, y_pred, metrics, save_path, selected_count=2)

    assert result_path.exists()
    assert result_path.stat().st_size > 0
