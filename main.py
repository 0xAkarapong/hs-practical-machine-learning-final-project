"""Entry point for the health & wellness price-recommendation pipeline."""

from src.split_data import run_split


def main():
    run_split()


if __name__ == "__main__":
    main()