"""Hyperparameter tuning capability for the health & wellness price regression.

ponytail: implements the deferred Summary.md next-steps — tune XGBoost/GBDT
hyperparameters and add HistGradientBoostingRegressor and RandomForestRegressor
as candidates, all scored by K-fold CV. This is capability, not deployment: it
reports CV mean±std and held-out metrics for each candidate and saves the tuned
winners to models/tuned_*.joblib, but it does NOT overwrite the deployed
models/xgboost.joblib.
Promote a winner manually if its held-out R² clearly beats the current XGBoost
(R²≈0.2676) by more than CLEAR_MARGIN.

n_jobs=1 throughout — torch/sentence-transformer threads + joblib parallelism
segfault on macOS. The search is slower but safe; the upgrade path is a
subprocess-isolated search if tuning wall-clock becomes a pain.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.model_selection import KFold, RandomizedSearchCV
from xgboost import XGBRegressor

from src._03_split_data import RANDOM_STATE, TARGET_COLUMN, TEST_PATH, TRAIN_PATH
from src._06_train import train_xgboost
from src.common.metrics import (
    cross_val_metrics,
    evaluate_model,
    save_metrics,
    time_stage,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
SELECTED_FEATURES_PATH = MODELS_DIR / "selected_features.json"

CV_FOLDS = 5
N_ITER = 20
CLEAR_MARGIN = 0.01  # held-out R² gain to call a candidate a clear winner

# ponytail: modest param distributions — RandomizedSearchCV samples N_ITER combos
# from these lists (no scipy needed). Keep ranges sane for a 2k-row, 55-feature table.
XGB_DIST = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 4, 5, 6, 7],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
    "reg_lambda": [1.0, 2.0, 5.0],
    "reg_alpha": [0.0, 0.1, 1.0],
}

GBDT_DIST = {
    "n_estimators": [100, 200, 300],
    "max_depth": [2, 3, 4, 5],
    "learning_rate": [0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 1.0],
}

HIST_DIST = {
    "max_iter": [100, 200, 300, 500],
    "max_depth": [None, 3, 5, 7],
    "learning_rate": [0.05, 0.1, 0.2],
    "l2_regularization": [0.0, 1.0, 5.0],
}

# ponytail: RandomForest is a bagging baseline vs the boosting candidates. The
# 2k-row table rules out deep forests; max_features as a fraction keeps trees cheap.
RF_DIST = {
    "n_estimators": [200, 400, 600],
    "max_depth": [None, 8, 16],
    "max_features": [0.5, 0.75, 1.0],
    "min_samples_leaf": [1, 5, 10],
}


def _load_selected_splits():
    train_table = pd.read_csv(TRAIN_PATH)
    test_table = pd.read_csv(TEST_PATH)
    selected = json.loads(SELECTED_FEATURES_PATH.read_text())
    X_train = train_table[selected]
    y_train = train_table[TARGET_COLUMN]
    X_test = test_table[selected]
    y_test = test_table[TARGET_COLUMN]
    return X_train, X_test, y_train, y_test


def _search(name, estimator, dist, X_train, y_train):
    """RandomizedSearchCV with K-fold CV, n_jobs=1.

    Returns (best_estimator, best_params).
    """
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator,
        dist,
        n_iter=N_ITER,
        cv=cv,
        scoring="neg_mean_squared_error",
        random_state=RANDOM_STATE,
        # ponytail: n_jobs=1 for the search too — torch/joblib thread clash
        # segfaults on macOS. Upgrade path: subprocess-isolated search once
        # embeddings are cached.
        n_jobs=1,
        refit=True,
    )
    with time_stage(f"search {name}"):
        search.fit(X_train, y_train)
    print(f"  {name} best params: {search.best_params_}")
    return search.best_estimator_, search.best_params_


def _eval_candidate(name, model, X_train, y_train, X_test, y_test, params=None):
    """CV mean±std + held-out test metrics for one fitted model. Prints a one-liner."""
    with time_stage(f"CV {name}"):
        cv = cross_val_metrics(model, X_train, y_train, k=CV_FOLDS)
    held = evaluate_model(model, X_test, y_test)
    print(
        f"{name}: CV R² {cv['r2_log_mean']:.4f}±{cv['r2_log_std']:.4f}"
        f"  held R² {held['r2_log']:.4f}  RMSE(log) {held['rmse_log']:.4f}"
        f"  RMSE(THB) {held['rmse_thb']:,.2f}"
    )
    return {"held": held, "cv": cv, "params": params or {}}


def _recommend(results: dict):
    """Print whether any candidate clearly beats the deployed XGBoost."""
    current_r2 = results["current_xgb"]["held"]["r2_log"]
    print(f"\nDeployed XGBoost held-out R²: {current_r2:.4f}")
    winners = []
    for name in ("tuned_xgb", "tuned_gbdt", "hist_gb", "rf"):
        delta = results[name]["held"]["r2_log"] - current_r2
        if delta > CLEAR_MARGIN:
            tag = "CLEAR WINNER"
        elif delta > 0:
            tag = "marginal"
        else:
            tag = "no gain"
        print(f"  {name}: ΔR²={delta:+.4f}  [{tag}]")
        if delta > CLEAR_MARGIN:
            winners.append(name)
    if winners:
        print(
            "Promote a winner manually: copy models/tuned_*.joblib to the deployed"
            " model path and update selected_features.json if the feature set changed."
        )
    else:
        print(
            "No candidate clearly beats the deployed model — keeping"
            " models/xgboost.joblib."
        )


def run_tune() -> dict:
    """Tune XGB/GBDT + HistGBM, report CV±std and held-out, save tuned artifacts.

    Run with: uv run python -m src._08_tune
    """
    X_train, X_test, y_train, y_test = _load_selected_splits()
    results = {}

    print("Evaluating deployed XGBoost baseline to beat...")
    current = train_xgboost(X_train, y_train)
    results["current_xgb"] = _eval_candidate(
        "current_xgb", current, X_train, y_train, X_test, y_test
    )

    print("\nTuning XGBoost...")
    xgb_best, xgb_params = _search(
        "tuned_xgb",
        XGBRegressor(n_jobs=-1, random_state=RANDOM_STATE),
        XGB_DIST,
        X_train,
        y_train,
    )
    results["tuned_xgb"] = _eval_candidate(
        "tuned_xgb", xgb_best, X_train, y_train, X_test, y_test, xgb_params
    )
    joblib.dump(xgb_best, MODELS_DIR / "tuned_xgboost.joblib")

    print("\nTuning sklearn GradientBoostingRegressor...")
    gbdt_best, gbdt_params = _search(
        "tuned_gbdt",
        GradientBoostingRegressor(random_state=RANDOM_STATE),
        GBDT_DIST,
        X_train,
        y_train,
    )
    results["tuned_gbdt"] = _eval_candidate(
        "tuned_gbdt", gbdt_best, X_train, y_train, X_test, y_test, gbdt_params
    )
    joblib.dump(gbdt_best, MODELS_DIR / "tuned_gbdt.joblib")

    print("\nTuning HistGradientBoostingRegressor...")
    hist_best, hist_params = _search(
        "hist_gb",
        HistGradientBoostingRegressor(random_state=RANDOM_STATE),
        HIST_DIST,
        X_train,
        y_train,
    )
    results["hist_gb"] = _eval_candidate(
        "hist_gb", hist_best, X_train, y_train, X_test, y_test, hist_params
    )
    joblib.dump(hist_best, MODELS_DIR / "tuned_hist_gb.joblib")

    print("\nTuning RandomForestRegressor...")
    rf_best, rf_params = _search(
        "rf",
        RandomForestRegressor(n_jobs=1, random_state=RANDOM_STATE),
        RF_DIST,
        X_train,
        y_train,
    )
    results["rf"] = _eval_candidate(
        "rf", rf_best, X_train, y_train, X_test, y_test, rf_params
    )
    joblib.dump(rf_best, MODELS_DIR / "tuned_rf.joblib")

    _recommend(results)
    results_path = save_metrics(results, MODELS_DIR / "tuning_results.json")
    print(f"Saved:  {results_path.name} → {results_path.parent.name}/")
    return results


if __name__ == "__main__":
    run_tune()
