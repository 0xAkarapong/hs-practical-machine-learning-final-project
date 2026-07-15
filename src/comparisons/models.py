"""Compare province-level vs region-level model performance and plot results."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import RFECV
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

from src._02_feature_engineer import OUTPUT_PATH, build_feature_table
from src._03_split_data import RANDOM_STATE, TARGET_COLUMN
from src.common.metrics import evaluate_model

# src/comparisons/ is one level deeper than the src/ modules -> repo root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_PATH = PROJECT_ROOT / "dataset" / "health_and_wellness_no_outliers.csv"
FIGURES_DIR = PROJECT_ROOT / "notebooks" / "figures"

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BLUE = "#2a78d6"
AQUA = "#1baf7a"


def _select_features(X_train, y_train) -> list[str]:
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    selector = RFECV(
        estimator=RidgeCV(),
        min_features_to_select=10,
        cv=cv,
        scoring="neg_mean_squared_error",
        # ponytail: n_jobs=1 — consistent with the main pipeline's feature_selection.
        # torch/sentence-transformer threads + joblib parallelism segfault on macOS.
        n_jobs=1,
    )
    selector.fit(X_train, y_train)
    return X_train.columns[selector.get_support()].tolist()


def _split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce the stratified split from src._03_split_data on any feature table."""
    from src._03_split_data import split_data

    return split_data(df)


def _train_and_eval(feature_table: pd.DataFrame, label: str):
    train_table, test_table = _split(feature_table)
    X_train = train_table.drop(columns=TARGET_COLUMN)
    y_train = train_table[TARGET_COLUMN]
    X_test = test_table.drop(columns=TARGET_COLUMN)
    y_test = test_table[TARGET_COLUMN]

    selected = _select_features(X_train, y_train)
    X_train = X_train[selected]
    X_test = X_test[selected]

    model = GradientBoostingRegressor(random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test, include_features=True)
    print(
        f"{label}: {metrics['features']} features, "
        f"R²={metrics['r2_log']:.4f}, RMSE={metrics['rmse_log']:.4f}, "
        f"MAE={metrics['mae_log']:.4f}"
    )
    return model, metrics


def _plot_comparison(province_metrics, region_metrics, save_path: Path) -> Path:
    """Save a small-multiples comparison of R² and RMSE for the two encodings.

    ponytail: two separate subplots instead of a dual-axis chart — R² and RMSE
    live on different natural scales, so each gets its own y-axis.
    """
    labels = ["Province-level", "Region-level"]
    r2s = [province_metrics["r2_log"], region_metrics["r2_log"]]
    rmses = [province_metrics["rmse_log"], region_metrics["rmse_log"]]
    feature_counts = [
        province_metrics["features"],
        region_metrics["features"],
    ]

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

    panels = [
        (ax_r2, bars_r2, feature_counts),
        (ax_rmse, bars_rmse, feature_counts),
    ]
    for ax, bars, counts in panels:
        for bar, count in zip(bars, counts, strict=True):
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}\n({count} feats)",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                color=INK_SECONDARY,
                fontsize=9,
            )

    fig.suptitle(
        "Province-level vs Region-level GradientBoostingRegressor",
        color=INK_PRIMARY,
        fontsize=13,
    )
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return save_path


def run_compare() -> dict[str, dict[str, float]]:
    """Train and compare province-level vs region-level models.

    ponytail: this script rebuilds the province-encoded table on the fly by
    toggling use_region=False in build_feature_table. It exists only for the
    comparison plot; the main pipeline uses the region-encoded CSV.
    """
    raw_df = pd.read_csv(RAW_PATH)
    province_table = build_feature_table(raw_df, use_region=False)
    region_table = pd.read_csv(OUTPUT_PATH)

    _, province_metrics = _train_and_eval(province_table, "Province")
    _, region_metrics = _train_and_eval(region_table, "Region")

    figure_path = _plot_comparison(
        province_metrics,
        region_metrics,
        FIGURES_DIR / "model_comparison.png",
    )
    print(f"Saved:  {figure_path.name} → notebooks/figures/")
    return {"province": province_metrics, "region": region_metrics}


if __name__ == "__main__":
    run_compare()
