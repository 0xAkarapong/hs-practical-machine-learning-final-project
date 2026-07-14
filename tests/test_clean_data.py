import numpy as np
import pandas as pd

from src._01_clean_data import ORIGINAL_COLUMNS, clean_data


def _fixture() -> pd.DataFrame:
    """Two sections with varied prices + one injected high-price outlier."""
    rows = []
    # Section A: ฿50–฿300 spread (so MAD>0) + ฿50,000 outlier.
    for i, price in enumerate([50, 80, 100, 120, 150, 200, 250, 300]):
        rows.append([f"a{i}", "A", f"name A{i}", f"฿{price}", "1 ชิ้น", "1", "Bangkok"])
    rows.append(["a_out", "A", "expensive A", "฿50,000", "1 ชิ้น", "1", "Bangkok"])
    # Section B: ฿300 x8 (stable, different median).
    for i in range(8):
        rows.append([f"b{i}", "B", f"name B{i}", "฿300", "1 ชิ้น", "1", "Chiang Mai"])
    return pd.DataFrame(rows, columns=ORIGINAL_COLUMNS)


def test_clean_data_drops_outlier_and_preserves_schema():
    df = _fixture()
    clean = clean_data(df)

    assert list(clean.columns) == ORIGINAL_COLUMNS
    assert clean["Price"].isna().sum() == 0
    # The ฿50,000 outlier is gone; the 16 stable rows survive.
    assert len(clean) == 16
    assert (clean["Price"] == 50000).sum() == 0
    assert np.log1p(clean["Price"]).max() < np.log1p(50000)


def test_clean_data_keeps_all_rows_when_no_outliers():
    # Distinct Ids (so dedup keeps them), same stable price -> no anomalies.
    df = pd.DataFrame(
        [[f"a{i}", "A", f"name {i}", "฿100", "1 ชิ้น", "1", "Bangkok"] for i in range(10)],
        columns=ORIGINAL_COLUMNS,
    )
    clean = clean_data(df)
    assert list(clean.columns) == ORIGINAL_COLUMNS
    assert len(clean) == 10