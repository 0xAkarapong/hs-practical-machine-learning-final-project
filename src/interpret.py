"""Model interpretation via SHAP for the health & wellness price model."""

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.split_data import TEST_PATH

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
FIGURES_DIR = PROJECT_ROOT / "notebooks" / "figures"

# Validated dataviz palette (light surface)
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"


def explain_model(
    model_path: Path = MODELS_DIR / "gradient_boosting.joblib",
    selected_path: Path = MODELS_DIR / "selected_features.json",
    test_path: Path = TEST_PATH,
    top_n: int = 15,
) -> pd.Series:
    """Compute and save SHAP-based interpretation for the GBDT model.

    ponytail: TreeSHAP on the test set gives both global feature importance
    (mean |SHAP value|) and local explanations (beeswarm). SHAP is installed
    specifically for tree-based model interpretation; the gradient-boosted
    trees are small enough that TreeSHAP runs quickly on the 526-row test set.
    """
    model = joblib.load(model_path)
    selected_columns = json.loads(selected_path.read_text())

    test_table = pd.read_csv(test_path)
    X_test = test_table[selected_columns]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    importance = pd.Series(
        np.abs(shap_values).mean(axis=0), index=X_test.columns
    ).sort_values(ascending=False)

    print("Top features by mean |SHAP value|:")
    for name, value in importance.head(top_n).items():
        print(f"  {name}: {value:.4f}")

    _plot_importance(importance.head(top_n), FIGURES_DIR / "shap_importance.png")
    _plot_summary(X_test, shap_values, FIGURES_DIR / "shap_summary.png")
    return importance


def _plot_importance(importance: pd.Series, save_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    importance.sort_values().plot.barh(ax=ax, color="#2a78d6")
    ax.set_xlabel("Mean |SHAP value|", color=INK_SECONDARY)
    ax.set_title(
        "Feature importance — GBDT on region-level features",
        color=INK_PRIMARY,
    )
    for spine in ax.spines.values():
        spine.set_color("#e1e0d9")
    ax.tick_params(colors=INK_SECONDARY)
    ax.grid(True, color="#e1e0d9", lw=0.8, axis="x")
    ax.set_axisbelow(True)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved:  {save_path.name} → notebooks/figures/")
    return save_path


def _plot_summary(X_test: pd.DataFrame, shap_values, save_path: Path) -> Path:
    fig = plt.figure(figsize=(8, 7), facecolor=SURFACE)
    shap.summary_plot(
        shap_values,
        X_test,
        show=False,
        plot_size=None,
    )
    fig.patch.set_facecolor(SURFACE)
    ax = fig.gca()
    ax.set_facecolor(SURFACE)
    for spine in ax.spines.values():
        spine.set_color("#e1e0d9")
    ax.tick_params(colors=INK_SECONDARY)
    ax.set_xlabel("SHAP value", color=INK_SECONDARY)
    ax.set_title("SHAP summary — GBDT on region-level features", color=INK_PRIMARY)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved:  {save_path.name} → notebooks/figures/")
    return save_path


def run_interpret() -> pd.Series:
    """Entry point for model interpretation."""
    return explain_model()


if __name__ == "__main__":
    run_interpret()
