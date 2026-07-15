"""Shared evaluation metrics, cross-validation, timing, and metrics persistence.

Hoists the `evaluate` helper (once copy-pasted across train.py and the compare
scripts) into one place, and adds the K-fold CV estimate. Metrics are persisted
to models/metrics.json so runs are comparable across changes — previously
numbers lived only in stdout.
"""

import json
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from src._03_split_data import RANDOM_STATE

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # src/common/ -> repo root
METRICS_PATH = PROJECT_ROOT / "models" / "metrics.json"


def evaluate_model(
    model, X_test, y_test, include_features: bool = False
) -> dict[str, float]:
    """Return RMSE/MAE/R² in log space and in THB (expm1 of the log target).

    Reports both spaces — the model fits log_price_thb, but THB is the unit the
    business reads. expm1 inverts the log1p transform applied during feature
    engineering. include_features lets the compare scripts keep their feature-count
    annotation in the same dict.
    """
    preds_log = model.predict(X_test)
    rmse_log = float(np.sqrt(mean_squared_error(y_test, preds_log)))
    mae_log = float(mean_absolute_error(y_test, preds_log))
    r2_log = float(r2_score(y_test, preds_log))

    y_test_thb = np.expm1(y_test.to_numpy())
    preds_thb = np.expm1(preds_log)
    rmse_thb = float(np.sqrt(np.mean((y_test_thb - preds_thb) ** 2)))
    mae_thb = float(np.mean(np.abs(y_test_thb - preds_thb)))

    metrics: dict[str, float] = {
        "rmse_log": rmse_log,
        "mae_log": mae_log,
        "r2_log": r2_log,
        "rmse_thb": rmse_thb,
        "mae_thb": mae_thb,
    }
    if include_features:
        metrics["features"] = X_test.shape[1]
    return metrics


def cross_val_metrics(
    model, X, y, k: int = 5, seed: int = RANDOM_STATE
) -> dict[str, float]:
    """K-fold CV mean±std for R²/RMSE/MAE in log space.

    The single held-out test set (526 rows) gives a high-variance R²; CV
    quantifies the uncertainty. The estimator is cloned per fold so no warm-start
    leakage.
    """
    cv = KFold(n_splits=k, shuffle=True, random_state=seed)
    X_arr = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
    y_arr = y.to_numpy() if hasattr(y, "to_numpy") else np.asarray(y)
    r2, rmse, mae = [], [], []
    for train_idx, test_idx in cv.split(X_arr):
        fold_model = clone(model)
        fold_model.fit(X_arr[train_idx], y_arr[train_idx])
        preds = fold_model.predict(X_arr[test_idx])
        r2.append(r2_score(y_arr[test_idx], preds))
        rmse.append(np.sqrt(mean_squared_error(y_arr[test_idx], preds)))
        mae.append(mean_absolute_error(y_arr[test_idx], preds))
    return {
        "k": k,
        "r2_log_mean": float(np.mean(r2)),
        "r2_log_std": float(np.std(r2)),
        "rmse_log_mean": float(np.mean(rmse)),
        "rmse_log_std": float(np.std(rmse)),
        "mae_log_mean": float(np.mean(mae)),
        "mae_log_std": float(np.std(mae)),
    }


def save_metrics(metrics: dict, path: Path = METRICS_PATH) -> Path:
    """Persist a metrics dict to JSON so runs are comparable across changes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return path


@contextmanager
def time_stage(label: str):
    """Print the wall-clock duration of a stage.

    time.perf_counter around expensive stages (embedding encode, feature
    selection, model fit) is enough to see where the pipeline spends time. Use as
    `with time_stage("fit XGBoost"): ...`.
    """
    start = time.perf_counter()
    yield
    print(f"  [{label}] {time.perf_counter() - start:.2f}s")
