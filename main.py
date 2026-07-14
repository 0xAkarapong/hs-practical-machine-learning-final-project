"""Entry point for the health & wellness price-recommendation pipeline."""

from src.feature_engineer import run_feature_engineer
from src.split_data import run_split
from src.train import run_train


def main():
    run_feature_engineer()
    run_split()
    run_train()


if __name__ == "__main__":
    main()
