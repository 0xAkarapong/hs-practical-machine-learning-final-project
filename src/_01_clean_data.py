"""Dedup + anomaly-based cleaning of the raw health & wellness scrape.

Ports `notebooks/02 anomaly_detection.ipynb` so the cleaned training input
(`dataset/health_and_wellness_no_outliers.csv`) is reproducible from the raw
CSV without running the notebook by hand. Two methods, both pure pandas/numpy:

  A. Global IQR on log1p(Price) — coarse, ignores category.
  B. Per-section robust z (median/MAD) — flags prices that don't fit their own
     section; robust to skew and to sections with very different medians.

Listings flagged by BOTH are dropped as high-confidence anomalies.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend, matches src._06_train
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "dataset" / "health_and_wellness_cleaned_20230810_213000.csv"
OUTPUT_PATH = PROJECT_ROOT / "dataset" / "health_and_wellness_no_outliers.csv"
FIGURES_DIR = PROJECT_ROOT / "notebooks" / "figures"

# Anomaly thresholds (notebook 02 defaults).
IQR_K = 1.5
Z_SCALE = 0.6745
ROBUST_Z_THRESHOLD = 3.5
DROP_FLAG = "anom_both"  # 'anom_robust' for a leaner training set

# Validated dataviz palette (light surface) — see dataviz skill palette.md.
SURFACE = "#fcfcfb"
SERIES_BLUE = "#2a78d6"
SERIES_ORANGE = "#f58518"
SERIES_RED = "#e45756"

ORIGINAL_COLUMNS = [
    "Id",
    "Section",
    "Name",
    "Price",
    "Total Sold",
    "Total Reviews",
    "Shop Location",
]


def parse_price(df: pd.DataFrame) -> pd.DataFrame:
    """Strip the ฿ prefix and thousand separators → numeric, add log_price."""
    df = df.copy()
    df["Price"] = pd.to_numeric(
        df["Price"]
        .astype(str)
        .str.replace("฿", "", regex=False)
        .str.replace(",", "", regex=False),
        errors="coerce",
    )
    assert df["Price"].isna().sum() == 0, "Price has NaN after parsing"
    df["log_price"] = np.log1p(df["Price"])
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicates; each listing is scored once for anomalies."""
    n_raw = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    assert df.duplicated().sum() == 0
    print(f"dedup: {n_raw:,} -> {len(df):,} ({n_raw - len(df):,} removed)")
    return df


def flag_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Add anom_iqr (global Tukey fences) and anom_robust (per-section |z|>3.5)."""
    df = df.copy()

    # A. Global IQR on log1p(Price).
    q1, q3 = df["log_price"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - IQR_K * iqr, q3 + IQR_K * iqr
    df["anom_iqr"] = (df["log_price"] < lo) | (df["log_price"] > hi)
    n_iqr, pct_iqr = df["anom_iqr"].sum(), df["anom_iqr"].mean() * 100
    print(f"A global IQR: {n_iqr} of {len(df)} ({pct_iqr:.1f}%)")
    print(f"  fences: ฿{np.expm1(lo):.0f} .. ฿{np.expm1(hi):.0f}")

    # B. Per-section robust z (median/MAD). MAD=0 (constant-price section) -> NaN
    # so those rows aren't flagged.
    g = df.groupby("Section")["log_price"]
    med = g.transform("median")
    mad = g.transform(lambda s: np.median(np.abs(s - np.median(s))))
    mad = mad.replace(0, np.nan)
    df["robust_z"] = Z_SCALE * (df["log_price"] - med) / mad
    df["anom_robust"] = df["robust_z"].abs() > ROBUST_Z_THRESHOLD
    df["anom_both"] = df["anom_iqr"] & df["anom_robust"]
    n_rz, pct_rz = df["anom_robust"].sum(), df["anom_robust"].mean() * 100
    print(f"B per-section |z|>{ROBUST_Z_THRESHOLD}: {n_rz} ({pct_rz:.1f}%)")
    print(f"  both A & B: {df['anom_both'].sum()}")
    return df


def drop_anomalies(flagged: pd.DataFrame) -> pd.DataFrame:
    """Drop high-confidence anomalies and helper columns -> original 7-col schema."""
    n_before = len(flagged)
    clean = flagged[~flagged[DROP_FLAG]].drop(
        columns=["log_price", "anom_iqr", "robust_z", "anom_robust", "anom_both"]
    )
    assert list(clean.columns) == ORIGINAL_COLUMNS, "schema drift"
    assert len(clean) == n_before - int(flagged[DROP_FLAG].sum())
    assert clean["Price"].isna().sum() == 0
    dropped, pct_drop = n_before - len(clean), flagged[DROP_FLAG].mean() * 100
    print(
        f"before {n_before} -> after {len(clean)} (dropped {dropped}, {pct_drop:.1f}%)"
    )
    return clean


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Parse price -> dedup -> flag -> drop high-confidence anomalies.

    Returns the cleaned table with the original 7-column schema (helpers dropped).
    Pure function: no I/O, the unit-testable seam.
    """
    return drop_anomalies(flag_anomalies(remove_duplicates(parse_price(df))))


def plot_anomalies(df: pd.DataFrame, clean: pd.DataFrame, figures_dir: Path) -> None:
    """Reproduce the two notebook 02 figures."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Price-by-section boxplots with high-confidence anomalies overlaid.
    order = df.groupby("Section")["Price"].median().sort_values(ascending=False).index
    fig, ax = plt.subplots(figsize=(11, 12))
    sns.boxplot(
        data=df,
        y="Section",
        x="Price",
        order=order,
        ax=ax,
        fliersize=0,
        color=SERIES_BLUE,
        width=0.6,
    )
    anom = df[df["anom_both"]]
    sns.stripplot(
        data=anom,
        y="Section",
        x="Price",
        order=order,
        ax=ax,
        color=SERIES_RED,
        size=2,
        alpha=0.6,
    )
    ax.set_xscale("log")
    ax.set_title(
        "Price by Section (log x) — red = high-confidence anomalies (IQR ∩ robust-z)"
    )
    fig.tight_layout()
    fig.savefig(
        figures_dir / "anomaly_price_by_section.png",
        dpi=150,
        facecolor=SURFACE,
        bbox_inches="tight",
    )
    plt.close(fig)
    print("Saved:  anomaly_price_by_section.png → notebooks/figures/")

    # Before/after price histograms (raw + log1p).
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(df["Price"], bins=80, ax=axes[0], color=SERIES_BLUE, label="before")
    sns.histplot(
        clean["Price"], bins=80, ax=axes[0], color=SERIES_RED, label="after", alpha=0.6
    )
    axes[0].set_title("Price (raw, ฿)")
    axes[0].legend()
    sns.histplot(
        np.log1p(df["Price"]), bins=80, ax=axes[1], color=SERIES_BLUE, label="before"
    )
    sns.histplot(
        np.log1p(clean["Price"]),
        bins=80,
        ax=axes[1],
        color=SERIES_RED,
        label="after",
        alpha=0.6,
    )
    axes[1].set_title("Price (log1p)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(
        figures_dir / "anomaly_before_after.png",
        dpi=150,
        facecolor=SURFACE,
        bbox_inches="tight",
    )
    plt.close(fig)
    print("Saved:  anomaly_before_after.png → notebooks/figures/")


def run_clean_data() -> None:
    """Load raw CSV -> clean -> write health_and_wellness_no_outliers.csv + figures."""
    df = pd.read_csv(RAW_PATH)
    flagged = flag_anomalies(remove_duplicates(parse_price(df)))
    clean = drop_anomalies(flagged)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(OUTPUT_PATH, index=False)
    print(f"wrote {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size // 1024} KB)")

    plot_anomalies(flagged, clean, FIGURES_DIR)


if __name__ == "__main__":
    run_clean_data()
