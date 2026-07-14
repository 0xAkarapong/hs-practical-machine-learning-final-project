"""Entry point for the health & wellness price-recommendation pipeline."""

from src._01_clean_data import run_clean_data
from src._02_feature_engineer import run_feature_engineer
from src._03_split_data import run_split
from src._06_train import run_train
from src.comparisons.baseline import run_compare_baseline


def main():
    run_clean_data()
    run_feature_engineer()
    run_split()
    run_train()
    run_compare_baseline()


if __name__ == "__main__":
    main()
