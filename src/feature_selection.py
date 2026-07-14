"""Wrapper-based feature selection for the health & wellness price regression."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

from src.split_data import RANDOM_STATE, TARGET_COLUMN, TRAIN_PATH

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
SELECTOR_PATH = MODELS_DIR / "feature_selector.joblib"


def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    min_features: int = 10,
    cv: int = 5,
    seed: int = RANDOM_STATE,
) -> list[str]:
    """Return the column names selected by RFECV with a RidgeCV wrapper.

    ponytail: RidgeCV is the wrapper estimator because it is fast and stable on
    the one-hot feature table; the final model remains GradientBoostingRegressor.
    RFECV greedily prunes the least important features and cross-validates the
    subset size, so the kept set is the one that maximizes the wrapper's CV score.
    """
    cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=seed)
    selector = RFECV(
        estimator=RidgeCV(),
        min_features_to_select=min_features,
        cv=cv_splitter,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    selector.fit(X_train, y_train)
    mask = selector.get_support()
    selected = X_train.columns[mask].tolist()
    return selected


def run_select_features() -> list[str]:
    """Load the training split and persist the selected feature set."""
    train_table = pd.read_csv(TRAIN_PATH)
    X_train = train_table.drop(columns=TARGET_COLUMN)
    y_train = train_table[TARGET_COLUMN]

    selected = select_features(X_train, y_train)
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(selected, SELECTOR_PATH)

    print(f"Selected {len(selected)} of {X_train.shape[1]} features via RFECV wrapper")
    print(f"Saved:  {SELECTOR_PATH.name} → {SELECTOR_PATH.parent.name}/")
    return selected


if __name__ == "__main__":
    run_select_features()
