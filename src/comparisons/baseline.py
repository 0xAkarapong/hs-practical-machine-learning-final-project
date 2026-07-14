"""Compare trained models against the mean-predictor baseline and plot results."""

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyRegressor

from src._03_split_data import TARGET_COLUMN, TEST_PATH, TRAIN_PATH
from src.common.metrics import evaluate_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "notebooks" / "figures"

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRIDLINE = "#e1e0d9"
BLUE = "#2a78d6"
CORAL = "#e07850"
GRAY = "#9a9995"


def _load_models() -> tuple[DummyRegressor, object, object, list[str]]:
    train_table = pd.read_csv(TRAIN_PATH)
    X_train = train_table.drop(columns=[TARGET_COLUMN])
    y_train = train_table[TARGET_COLUMN]

    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)

    selected = json.loads((MODELS_DIR / "selected_features.json").read_text())
    gbdt = joblib.load(MODELS_DIR / "gradient_boosting.joblib")
    xgb = joblib.load(MODELS_DIR / "xgboost.joblib")
    return baseline, gbdt, xgb, selected


def _plot_baseline_comparison(
    baseline_metrics: dict[str, float],
    gbdt_metrics: dict[str, float],
    xgb_metrics: dict[str, float],
    save_path: Path,
) -> Path:
    """Save a grouped-bar plot of baseline vs models across metrics."""
    labels = ["Baseline\n(mean)", "GBDT", "XGBoost"]
    r2_values = [
        max(baseline_metrics["r2_log"], 0.0),
        gbdt_metrics["r2_log"],
        xgb_metrics["r2_log"],
    ]
    rmse_log_values = [
        baseline_metrics["rmse_log"],
        gbdt_metrics["rmse_log"],
        xgb_metrics["rmse_log"],
    ]
    rmse_thb_values = [
        baseline_metrics["rmse_thb"],
        gbdt_metrics["rmse_thb"],
        xgb_metrics["rmse_thb"],
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), facecolor=SURFACE)
    metric_groups = [
        ("R² (log)", r2_values, BLUE),
        ("RMSE (log)", rmse_log_values, CORAL),
        ("RMSE (THB)", rmse_thb_values, GRAY),
    ]

    for ax, (title, values, color) in zip(axes, metric_groups, strict=True):
        ax.set_facecolor(SURFACE)
        for spine in ax.spines.values():
            spine.set_color(GRIDLINE)
        ax.tick_params(colors=INK_SECONDARY)
        ax.grid(True, color=GRIDLINE, lw=0.8, axis="y")
        ax.set_axisbelow(True)

        bars = ax.bar(labels, values, color=color, width=0.6)
        ax.set_title(title, color=INK_PRIMARY)
        ax.set_ylim(0, max(values) * 1.15)

        for bar in bars:
            height = bar.get_height()
            text = f"{height:.3f}" if title == "R² (log)" else f"{height:,.1f}"
            ax.annotate(
                text,
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color=INK_SECONDARY,
                fontsize=9,
            )

    fig.suptitle(
        "Baseline vs trained models on log-price prediction",
        color=INK_PRIMARY,
        fontsize=13,
    )
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return save_path


def run_compare_baseline() -> dict[str, dict[str, float]]:
    """Evaluate baseline, GBDT, and XGBoost on the held-out test set."""
    baseline, gbdt, xgb, selected = _load_models()

    test_table = pd.read_csv(TEST_PATH)
    X_test = test_table.drop(columns=[TARGET_COLUMN])
    y_test = test_table[TARGET_COLUMN]
    X_test_selected = X_test[selected]

    baseline_metrics = evaluate_model(baseline, X_test, y_test)
    gbdt_metrics = evaluate_model(gbdt, X_test_selected, y_test)
    xgb_metrics = evaluate_model(xgb, X_test_selected, y_test)

    print("Baseline metrics:")
    for key, value in baseline_metrics.items():
        print(f"  {key}: {value:.4f}")
    print("GBDT metrics:")
    for key, value in gbdt_metrics.items():
        print(f"  {key}: {value:.4f}")
    print("XGBoost metrics:")
    for key, value in xgb_metrics.items():
        print(f"  {key}: {value:.4f}")

    figure_path = _plot_baseline_comparison(
        baseline_metrics,
        gbdt_metrics,
        xgb_metrics,
        FIGURES_DIR / "baseline_comparison.png",
    )
    print(f"Saved:  {figure_path.name} → notebooks/figures/")

    return {
        "baseline": baseline_metrics,
        "gbdt": gbdt_metrics,
        "xgb": xgb_metrics,
    }


if __name__ == "__main__":
    run_compare_baseline()
