"""Train a GradientBoostingRegressor on the health & wellness price splits."""

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # headless backend — safe inside the Docker container
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.baseline import train_baseline
from src.split_data import RANDOM_STATE, TARGET_COLUMN, TEST_PATH, TRAIN_PATH

Regressor = GradientBoostingRegressor | DummyRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "gradient_boosting.joblib"
FIGURES_DIR = PROJECT_ROOT / "notebooks" / "figures"
FIGURE_PATH = FIGURES_DIR / "predicted_vs_actual.png"

# Validated dataviz palette (light surface) — see dataviz skill palette.md.
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SERIES_BLUE = "#2a78d6"


def train_model(
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


def evaluate_model(
    model: Regressor, X_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    """Return RMSE/MAE/R² in log space and in THB (expm1 of the log target).

    ponytail: report both spaces — the model fits log_price_thb, but THB is
    the unit the business reads. expm1 inverts the log1p transform applied
    during feature engineering.
    """
    preds_log = model.predict(X_test)
    rmse_log = float(np.sqrt(mean_squared_error(y_test, preds_log)))
    mae_log = float(mean_absolute_error(y_test, preds_log))
    r2_log = float(r2_score(y_test, preds_log))

    y_test_thb = np.expm1(y_test.to_numpy())
    preds_thb = np.expm1(preds_log)
    rmse_thb = float(np.sqrt(np.mean((y_test_thb - preds_thb) ** 2)))
    mae_thb = float(np.mean(np.abs(y_test_thb - preds_thb)))

    return {
        "rmse_log": rmse_log,
        "mae_log": mae_log,
        "r2_log": r2_log,
        "rmse_thb": rmse_thb,
        "mae_thb": mae_thb,
    }


def plot_result(
    y_actual: pd.Series,
    y_pred: np.ndarray,
    metrics: dict[str, float],
    save_path: Path,
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
    title = (
        f"GradientBoostingRegressor — "
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


def run_train() -> dict[str, float]:
    """Load splits, fit baseline + model, evaluate on test, persist the artifact.

    Prints a log mirroring run_split so the Docker container streams training
    output to stdout. Reports baseline metrics first so the model's gain over
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

    print(f"Fitting GradientBoostingRegressor on {X_train.shape[1]} features...")
    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"Saved:  {MODEL_PATH.name} → {MODEL_PATH.parent.name}/")
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
    figure_path = plot_result(y_test, preds, metrics, FIGURE_PATH)
    print(f"Saved:  {figure_path.name} → notebooks/figures/")
    return metrics


if __name__ == "__main__":
    run_train()
