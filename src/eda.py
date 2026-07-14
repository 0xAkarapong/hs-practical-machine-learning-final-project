"""Exploratory data analysis on the raw health & wellness scrape.

Ports `notebooks/01 eda.ipynb` into a runnable script. Standalone analysis — not
a pipeline step. Writes figures to notebooks/figures/.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src._01_clean_data import RAW_PATH, parse_price

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "notebooks" / "figures"

SURFACE = "#fcfcfb"
SERIES_BLUE = "#2a78d6"
SERIES_ORANGE = "#f58518"
SERIES_GREEN = "#54a24b"

# Raw-input contract — guards a trust boundary, kept as asserts.
RAW_ROW_COUNT = 68499
COMMA_NAME_COUNT = 3441


def load_raw() -> pd.DataFrame:
    """Load the raw CSV and assert the known shape / quoting sanity."""
    df = pd.read_csv(RAW_PATH)
    assert df.shape[0] == RAW_ROW_COUNT, df.shape
    assert (df["Section"] == "Stress").sum() == 0, "comma-split artifact survived"
    assert (df["Section"] == "Stress, Sleep, and Anxiety").sum() == 640
    assert df["Name"].astype(str).str.contains(",").sum() == COMMA_NAME_COUNT
    print(f"rows {df.shape[0]} | columns {list(df.columns)} | Section cats {df['Section'].nunique()}")
    return df


def missingness(df: pd.DataFrame) -> pd.DataFrame:
    """Missing count/pct table + bar plot."""
    miss = df.isna().sum().sort_values(ascending=False)
    miss_pct = (miss / len(df) * 100).round(1)
    table = pd.DataFrame({"missing": miss, "pct": miss_pct})
    print(table)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(x=miss_pct.values, y=miss_pct.index, ax=ax, color=SERIES_BLUE)
    ax.set_xlabel("% missing")
    ax.set_title("Missingness by column")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "missingness.png", dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("Saved:  missingness.png → notebooks/figures/")
    return table


def price_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Parse price, plot raw + log1p distributions and price-by-section boxplots."""
    df = parse_price(df)
    print(df["Price"].describe().round(1))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(df["Price"], bins=80, ax=axes[0], color=SERIES_BLUE)
    axes[0].set_title("Price (raw, ฿)")
    sns.histplot(np.log1p(df["Price"]), bins=80, ax=axes[1], color=SERIES_ORANGE)
    axes[1].set_title("Price (log1p)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "price_dist.png", dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("Saved:  price_dist.png → notebooks/figures/")

    order = df.groupby("Section")["Price"].median().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.boxplot(data=df, y="Section", x="Price", order=order, ax=ax, fliersize=1, color=SERIES_BLUE)
    ax.set_xscale("log")
    ax.set_title("Price by Section (log x)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "price_by_section.png", dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("Saved:  price_by_section.png → notebooks/figures/")
    return df


def categorical_profile(df: pd.DataFrame) -> None:
    """Section counts + top-20 shop location bar plots."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    sec = df["Section"].value_counts()
    sns.barplot(x=sec.values, y=sec.index, ax=axes[0], color=SERIES_BLUE)
    axes[0].set_title("Rows per Section")
    axes[0].set_xscale("log")

    loc = df["Shop Location"].value_counts().head(20)
    sns.barplot(x=loc.values, y=loc.index, ax=axes[1], color=SERIES_ORANGE)
    axes[1].set_title("Top-20 Shop Locations")
    axes[1].set_xscale("log")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "section_counts.png", dpi=150, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "location_counts.png", dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("Saved:  section_counts.png, location_counts.png → notebooks/figures/")


def price_vs_features(df: pd.DataFrame) -> pd.DataFrame:
    """Median price per section table + price boxplots by top-12 shop location."""
    agg = df.groupby("Section")["Price"].agg(["count", "mean", "median"]).sort_values("median", ascending=False)
    print(agg.round(1))

    top_loc = df["Shop Location"].value_counts().head(12).index
    sub = df[df["Shop Location"].isin(top_loc)]
    order = sub.groupby("Shop Location")["Price"].median().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.boxplot(data=sub, y="Shop Location", x="Price", order=order, ax=ax, fliersize=1, color=SERIES_GREEN)
    ax.set_xscale("log")
    ax.set_title("Price by top-12 Shop Location (log x)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "price_by_location.png", dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("Saved:  price_by_location.png → notebooks/figures/")
    return agg


def run_eda() -> None:
    """Run the full EDA sweep over the raw CSV."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    df = load_raw()
    missingness(df)
    df = price_profile(df)
    categorical_profile(df)
    price_vs_features(df)


if __name__ == "__main__":
    run_eda()