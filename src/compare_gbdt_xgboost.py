"""Compare GBDT vs XGBoost on the region-level selected feature set."""

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.metrics import evaluate_model
from src.split_data import TARGET_COLUMN, TEST_PATH

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "notebooks" / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRIDLINE = "#e1e0d9"
BLUE = "#2a78d6"
AQUA = "#1baf7a"


def _load_metrics(model_path: Path):
    model = joblib.load(model_path)
    test_table = pd.read_csv(TEST_PATH)
    X_test = test_table.drop(columns=TARGET_COLUMN)
    y_test = test_table[TARGET_COLUMN]
    selected = json.loads((MODELS_DIR / "selected_features.json").read_text())
    X_test = X_test[selected]
    return evaluate_model(model, X_test, y_test)


def _plot(metrics: dict[str, dict[str, float]], save_path: Path) -> Path:
    labels = ["GradientBoosting", "XGBoost"]
    r2s = [metrics["gbdt"]["r2_log"], metrics["xgboost"]["r2_log"]]
    rmses = [metrics["gbdt"]["rmse_log"], metrics["xgboost"]["rmse_log"]]

    fig, (ax_r2, ax_rmse) = plt.subplots(1, 2, figsize=(9, 4.5), facecolor=SURFACE)
    for ax in (ax_r2, ax_rmse):
        ax.set_facecolor(SURFACE)
        for spine in ax.spines.values():
            spine.set_color(GRIDLINE)
        ax.tick_params(colors=INK_SECONDARY)
        ax.grid(True, color=GRIDLINE, lw=0.8, axis="y")
        ax.set_axisbelow(True)

    x = np.arange(len(labels))
    width = 0.5

    bars_r2 = ax_r2.bar(x, r2s, width, color=BLUE)
    ax_r2.set_ylabel("R² (log)", color=INK_SECONDARY)
    ax_r2.set_title("Explained variance", color=INK_PRIMARY)
    ax_r2.set_xticks(x)
    ax_r2.set_xticklabels(labels, color=INK_SECONDARY)
    ax_r2.set_ylim(0, max(r2s) * 1.25)

    bars_rmse = ax_rmse.bar(x, rmses, width, color=AQUA)
    ax_rmse.set_ylabel("RMSE (log)", color=INK_SECONDARY)
    ax_rmse.set_title("Prediction error", color=INK_PRIMARY)
    ax_rmse.set_xticks(x)
    ax_rmse.set_xticklabels(labels, color=INK_SECONDARY)
    ax_rmse.set_ylim(0, max(rmses) * 1.15)

    for ax, bars in [(ax_r2, bars_r2), (ax_rmse, bars_rmse)]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color=INK_SECONDARY,
                fontsize=10,
            )

    fig.suptitle(
        "GBDT vs XGBoost — region-level selected features",
        color=INK_PRIMARY,
        fontsize=13,
    )
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return save_path


def run_compare_gbdt_xgboost() -> dict[str, dict[str, float]]:
    """Load saved models and plot their test-set metrics side by side."""
    metrics = {
        "gbdt": _load_metrics(MODELS_DIR / "gradient_boosting.joblib"),
        "xgboost": _load_metrics(MODELS_DIR / "xgboost.joblib"),
    }
    gbdt_r2, gbdt_rmse = metrics["gbdt"]["r2_log"], metrics["gbdt"]["rmse_log"]
    xgb_r2, xgb_rmse = metrics["xgboost"]["r2_log"], metrics["xgboost"]["rmse_log"]
    print(f"GBDT:    R²={gbdt_r2:.4f}, RMSE={gbdt_rmse:.4f}")
    print(f"XGBoost: R²={xgb_r2:.4f}, RMSE={xgb_rmse:.4f}")

    figure_path = _plot(metrics, FIGURES_DIR / "gbdt_vs_xgboost.png")
    print(f"Saved:  {figure_path.name} → notebooks/figures/")
    return metrics


if __name__ == "__main__":
    run_compare_gbdt_xgboost()
