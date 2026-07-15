"""Train gradient-boosted models on the health & wellness price splits."""

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # headless backend — safe inside the Docker container
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

from src._03_split_data import RANDOM_STATE, TARGET_COLUMN, TEST_PATH, TRAIN_PATH
from src._04_feature_selection import select_features
from src._05_baseline import train_baseline
from src._07_interpret import run_interpret
from src.common.metrics import (
    cross_val_metrics,
    evaluate_model,
    save_metrics,
    time_stage,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
GBDT_MODEL_PATH = MODELS_DIR / "gradient_boosting.joblib"
XGB_MODEL_PATH = MODELS_DIR / "xgboost.joblib"
SELECTED_FEATURES_PATH = MODELS_DIR / "selected_features.json"
FIGURES_DIR = PROJECT_ROOT / "notebooks" / "figures"
GBDT_FIGURE_PATH = FIGURES_DIR / "predicted_vs_actual_gbdt.png"
XGB_FIGURE_PATH = FIGURES_DIR / "predicted_vs_actual_xgboost.png"

# Validated dataviz palette (light surface) — see dataviz skill palette.md.
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SERIES_BLUE = "#2a78d6"


def train_gbdt(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int = RANDOM_STATE,
) -> GradientBoostingRegressor:
    """Fit a GradientBoostingRegressor on the training features/target.

    ponytail: default hyperparams (n_estimators=100, max_depth=3,
    learning_rate=0.1) — add GridSearchCV only if held-out metrics are poor.
    The feature table is already fully numeric with no missing values, so no
    preprocessing pipeline is wired in.
    """
    model = GradientBoostingRegressor(random_state=seed)
    model.fit(X_train, y_train)
    return model


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    seed: int = RANDOM_STATE,
) -> XGBRegressor:
    """Fit an XGBRegressor on the training features/target.

    ponytail: these ARE the tuned hyperparameters (from src/_08_tune.py
    RandomizedSearchCV, 5-fold CV) baked in as the deployed config, so a single
    main.py run reproduces the promoted model — no separate re-promote step.
    After re-tuning, update these defaults to the new best (single source of
    truth). _08_tune.py reuses this as the `current_xgb` baseline-to-beat.
    """
    model = XGBRegressor(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_lambda=5.0,
        reg_alpha=0.0,
        random_state=seed,
        # ponytail: n_jobs=-1 uses all cores for tree building (in-process threads,
        # not joblib fork). Safe because the torch encoder is freed before training
        # and joblib stays n_jobs=1 (no fork) — the macOS segfault was fork+torch.
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def plot_result(
    y_actual: pd.Series,
    y_pred: np.ndarray,
    metrics: dict[str, float],
    save_path: Path,
    model_name: str = "GradientBoostingRegressor",
    selected_count: int | None = None,
) -> Path:
    """Save a predicted-vs-actual scatter with a y=x reference diagonal.

    ponytail: one chart, one axis each — predicted vs actual in the model's
    native log space, where the ideal line is a clean 45°. The diagonal is a
    recessive annotation, not a second series; text wears ink tokens, not the
    series blue. Colors come from the validated dataviz palette (light surface).
    """
    fig, ax = plt.subplots(figsize=(6, 6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    ax.scatter(
        y_actual.to_numpy(),
        y_pred,
        s=14,
        color=SERIES_BLUE,
        alpha=0.55,
        edgecolors="none",
        label="Predicted vs actual",
    )
    lo = float(min(y_actual.min(), np.min(y_pred)))
    hi = float(max(y_actual.max(), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1.5, ls="--", label="Ideal: ŷ = y")

    ax.set_aspect("equal")
    ax.set_xlabel("Actual log_price_thb", color=INK_SECONDARY)
    ax.set_ylabel("Predicted log_price_thb", color=INK_SECONDARY)
    feature_note = f" ({selected_count} features)" if selected_count else ""
    title = (
        f"{model_name}{feature_note} — "
        f"R²={metrics['r2_log']:.3f}, RMSE={metrics['rmse_log']:.3f}"
    )
    ax.set_title(title, color=INK_PRIMARY)

    for spine in ax.spines.values():
        spine.set_color(GRIDLINE)
    ax.tick_params(colors=INK_SECONDARY)
    ax.grid(True, color=GRIDLINE, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(
        facecolor=SURFACE,
        edgecolor=GRIDLINE,
        labelcolor=INK_SECONDARY,
        frameon=True,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return save_path


def _fit_and_report(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    selected_columns: list[str],
    model_name: str,
    model_path: Path,
    figure_path: Path,
    baseline_metrics: dict[str, float],
) -> dict:
    """Fit one model, report CV + held-out metrics, save artifact and plot."""
    print(f"Fitting {model_name} on {X_train.shape[1]} features...")
    with time_stage(f"fit {model_name}"):
        if model_name == "XGBRegressor":
            model = train_xgboost(X_train, y_train)
        else:
            model = train_gbdt(X_train, y_train)

    # ponytail: cross_val_metrics clones the estimator per fold, so passing the
    # fitted model is safe. Quantifies the variance the single held-out set hides.
    with time_stage(f"CV {model_name}"):
        cv = cross_val_metrics(model, X_train, y_train)
    print(
        f"  CV R²(log): {cv['r2_log_mean']:.4f} ± {cv['r2_log_std']:.4f}"
        f"  RMSE(log): {cv['rmse_log_mean']:.4f} ± {cv['rmse_log_std']:.4f}"
    )

    metrics = evaluate_model(model, X_test, y_test)
    metrics["cv"] = cv
    joblib.dump(model, model_path)

    print(f"Saved:  {model_path.name} → {model_path.parent.name}/")
    print(
        f"Test RMSE (log): {metrics['rmse_log']:.4f}"
        f"  MAE (log): {metrics['mae_log']:.4f}"
        f"  R² (log): {metrics['r2_log']:.4f}"
    )
    print(
        f"Test RMSE (THB): {metrics['rmse_thb']:,.2f}"
        f"  MAE (THB): {metrics['mae_thb']:,.2f}"
    )
    r2_gain = metrics["r2_log"] - baseline_metrics["r2_log"]
    print(f"R² gain over baseline: {r2_gain:+.4f}")

    preds = model.predict(X_test)
    plot_result(
        y_test,
        preds,
        metrics,
        figure_path,
        model_name=model_name,
        selected_count=len(selected_columns),
    )
    print(f"Saved:  {figure_path.name} → notebooks/figures/")
    return metrics


def run_train() -> dict[str, dict[str, float]]:
    """Load splits, select features with a wrapper, fit baseline + models, plot.

    Prints a log mirroring run_split so the Docker container streams training
    output to stdout. Reports baseline metrics first so each model's gain over
    the naive mean-predictor is visible.
    """
    train_table = pd.read_csv(TRAIN_PATH)
    test_table = pd.read_csv(TEST_PATH)

    X_train = train_table.drop(columns=TARGET_COLUMN)
    y_train = train_table[TARGET_COLUMN]
    X_test = test_table.drop(columns=TARGET_COLUMN)
    y_test = test_table[TARGET_COLUMN]

    n_train, n_test = train_table.shape[0], test_table.shape[0]
    print(f"Loaded: {n_train:,} train rows, {n_test:,} test rows")

    print("Fitting baseline (mean-predictor)...")
    baseline = train_baseline(y_train)
    baseline_metrics = evaluate_model(baseline, X_test, y_test)
    print(
        f"Baseline RMSE (log): {baseline_metrics['rmse_log']:.4f}"
        f"  MAE (log): {baseline_metrics['mae_log']:.4f}"
        f"  R² (log): {baseline_metrics['r2_log']:.4f}"
    )

    print(f"Selecting features via RFECV wrapper from {X_train.shape[1]} features...")
    with time_stage("RFECV feature selection"):
        selected_columns = select_features(X_train, y_train)
    X_train = X_train[selected_columns]
    X_test = X_test[selected_columns]
    print(f"Kept:   {len(selected_columns)} features after wrapper selection")

    MODELS_DIR.mkdir(exist_ok=True)
    SELECTED_FEATURES_PATH.write_text(
        json.dumps(selected_columns, indent=2), encoding="utf-8"
    )
    print(
        f"Saved:  {SELECTED_FEATURES_PATH.name} → {SELECTED_FEATURES_PATH.parent.name}/"
    )

    gbdt_metrics = _fit_and_report(
        X_train,
        X_test,
        y_train,
        y_test,
        selected_columns,
        model_name="GradientBoostingRegressor",
        model_path=GBDT_MODEL_PATH,
        figure_path=GBDT_FIGURE_PATH,
        baseline_metrics=baseline_metrics,
    )

    xgb_metrics = _fit_and_report(
        X_train,
        X_test,
        y_train,
        y_test,
        selected_columns,
        model_name="XGBRegressor",
        model_path=XGB_MODEL_PATH,
        figure_path=XGB_FIGURE_PATH,
        baseline_metrics=baseline_metrics,
    )

    print("Interpreting GBDT with SHAP...")
    run_interpret()

    all_metrics = {
        "baseline": baseline_metrics,
        "gbdt": gbdt_metrics,
        "xgboost": xgb_metrics,
        "selected_features": selected_columns,
    }
    metrics_path = save_metrics(all_metrics)
    print(f"Saved:  {metrics_path.name} → {metrics_path.parent.name}/")
    return {"gbdt": gbdt_metrics, "xgboost": xgb_metrics}


if __name__ == "__main__":
    run_train()
