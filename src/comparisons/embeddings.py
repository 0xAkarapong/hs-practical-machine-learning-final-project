"""Compare model performance with vs without name embeddings and plot results."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

from src._03_split_data import RANDOM_STATE, TARGET_COLUMN, split_data
from src._04_feature_selection import select_features
from src.common.metrics import evaluate_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURE_PATH = PROJECT_ROOT / "dataset" / "health_and_wellness_feature_engineered.csv"
FEATURE_PATH_NO_EMB = (
    PROJECT_ROOT
    / "dataset"
    / "health_and_wellness_feature_engineered_no_embeddings.csv"
)
SELECTED_PATH = PROJECT_ROOT / "models" / "selected_features.json"
FIGURES_DIR = PROJECT_ROOT / "notebooks" / "figures"

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRIDLINE = "#e1e0d9"
BLUE = "#2a78d6"
AQUA = "#1baf7a"
CORAL = "#e07850"


def _train_and_eval(feature_table: pd.DataFrame, label: str, use_emb: bool):
    train_table, test_table = split_data(feature_table)
    X_train = train_table.drop(columns=TARGET_COLUMN)
    y_train = train_table[TARGET_COLUMN]
    X_test = test_table.drop(columns=TARGET_COLUMN)
    y_test = test_table[TARGET_COLUMN]

    baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
    baseline_metrics = evaluate_model(baseline, X_test, y_test, include_features=True)

    if use_emb:
        selected_columns = json.loads(SELECTED_PATH.read_text())
        selected_columns = [c for c in selected_columns if c in X_train.columns]
    else:
        selected_columns = select_features(X_train, y_train, min_features=5)

    X_train = X_train[selected_columns]
    X_test = X_test[selected_columns]

    gbdt = GradientBoostingRegressor(random_state=RANDOM_STATE).fit(X_train, y_train)
    xgb = XGBRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
        n_jobs=1,
    ).fit(X_train, y_train)

    gbdt_metrics = evaluate_model(gbdt, X_test, y_test, include_features=True)
    xgb_metrics = evaluate_model(xgb, X_test, y_test, include_features=True)
    print(
        f"{label}: {gbdt_metrics['features']} features, "
        f"GBDT R²={gbdt_metrics['r2_log']:.4f}, "
        f"XGB R²={xgb_metrics['r2_log']:.4f}"
    )
    return {
        "baseline": baseline_metrics,
        "gbdt": gbdt_metrics,
        "xgb": xgb_metrics,
    }


def _plot_comparison(with_emb: dict, without_emb: dict, save_path: Path) -> Path:
    """Save a small-multiples comparison of R² and RMSE for both variants."""
    labels = ["Without embeddings", "With embeddings"]
    gbdt_r2 = [without_emb["gbdt"]["r2_log"], with_emb["gbdt"]["r2_log"]]
    xgb_r2 = [without_emb["xgb"]["r2_log"], with_emb["xgb"]["r2_log"]]
    gbdt_rmse = [without_emb["gbdt"]["rmse_log"], with_emb["gbdt"]["rmse_log"]]
    xgb_rmse = [without_emb["xgb"]["rmse_log"], with_emb["xgb"]["rmse_log"]]
    feature_counts = [
        without_emb["gbdt"]["features"],
        with_emb["gbdt"]["features"],
    ]

    fig, (ax_r2, ax_rmse) = plt.subplots(1, 2, figsize=(10, 4.5), facecolor=SURFACE)
    for ax in (ax_r2, ax_rmse):
        ax.set_facecolor(SURFACE)
        for spine in ax.spines.values():
            spine.set_color(GRIDLINE)
        ax.tick_params(colors=INK_SECONDARY)
        ax.grid(True, color=GRIDLINE, lw=0.8, axis="y")
        ax.set_axisbelow(True)

    x = np.arange(len(labels))
    width = 0.35

    bars_gbdt_r2 = ax_r2.bar(x - width / 2, gbdt_r2, width, label="GBDT", color=BLUE)
    bars_xgb_r2 = ax_r2.bar(x + width / 2, xgb_r2, width, label="XGBoost", color=CORAL)
    ax_r2.set_ylabel("R² (log)", color=INK_SECONDARY)
    ax_r2.set_title("Explained variance", color=INK_PRIMARY)
    ax_r2.set_xticks(x)
    ax_r2.set_xticklabels(labels, color=INK_SECONDARY)
    ax_r2.set_ylim(0, max(*gbdt_r2, *xgb_r2) * 1.25)
    ax_r2.legend(frameon=False, labelcolor=INK_SECONDARY)

    bars_gbdt_rmse = ax_rmse.bar(
        x - width / 2, gbdt_rmse, width, label="GBDT", color=BLUE
    )
    bars_xgb_rmse = ax_rmse.bar(
        x + width / 2, xgb_rmse, width, label="XGBoost", color=CORAL
    )
    ax_rmse.set_ylabel("RMSE (log)", color=INK_SECONDARY)
    ax_rmse.set_title("Prediction error", color=INK_PRIMARY)
    ax_rmse.set_xticks(x)
    ax_rmse.set_xticklabels(labels, color=INK_SECONDARY)
    ax_rmse.set_ylim(0, max(*gbdt_rmse, *xgb_rmse) * 1.15)
    ax_rmse.legend(frameon=False, labelcolor=INK_SECONDARY)

    panels = [
        (ax_r2, bars_gbdt_r2, bars_xgb_r2),
        (ax_rmse, bars_gbdt_rmse, bars_xgb_rmse),
    ]
    for ax, bars1, bars2 in panels:
        for bar in (*bars1, *bars2):
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color=INK_SECONDARY,
                fontsize=8,
            )

    for ax, count in zip((ax_r2, ax_rmse), feature_counts, strict=True):
        ax.annotate(
            f"({count} features)",
            xy=(0.5, 0.02),
            xycoords="axes fraction",
            ha="center",
            va="bottom",
            color=INK_SECONDARY,
            fontsize=9,
        )

    fig.suptitle(
        "Effect of name embeddings on GBDT and XGBoost",
        color=INK_PRIMARY,
        fontsize=13,
    )
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return save_path


def run_compare_embeddings() -> dict[str, dict[str, dict[str, float]]]:
    """Train and compare with/without name embeddings."""
    without_emb = _train_and_eval(
        pd.read_csv(FEATURE_PATH_NO_EMB),
        "Without embeddings",
        use_emb=False,
    )
    with_emb = _train_and_eval(
        pd.read_csv(FEATURE_PATH),
        "With embeddings",
        use_emb=True,
    )

    figure_path = _plot_comparison(
        with_emb,
        without_emb,
        FIGURES_DIR / "embedding_comparison.png",
    )
    print(f"Saved:  {figure_path.name} → notebooks/figures/")
    return {"without_embeddings": without_emb, "with_embeddings": with_emb}


if __name__ == "__main__":
    run_compare_embeddings()
