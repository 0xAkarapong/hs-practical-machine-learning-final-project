import numpy as np
import pandas as pd

from src.split_data import split_data


def test_split_partitions_without_overlap():
    feature_table = pd.DataFrame(
        {"log_price_thb": np.linspace(1, 100, 1000), "x": range(1000)}
    )
    train_table, test_table = split_data(feature_table)
    assert len(train_table) + len(test_table) == len(feature_table)
    assert set(train_table.index) & set(test_table.index) == set()
    assert abs(len(test_table) / len(feature_table) - 0.2) < 0.05
