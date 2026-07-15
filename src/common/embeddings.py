"""Product-name sentence embeddings for the health & wellness price model."""

import hashlib
import os
import time
from pathlib import Path

# ponytail: disable tokenizer and OpenMP parallelism to avoid segfaults on macOS
# when sentence-transformers/torch interacts with joblib multiprocessing.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

from src._03_split_data import RANDOM_STATE

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # src/common/ -> repo root
MODELS_DIR = PROJECT_ROOT / "models"
EMBEDDING_CACHE_PATH = MODELS_DIR / "name_embeddings.joblib"

MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
EMBEDDING_DIM = 32
# ponytail: cap distinct name-sets cached to avoid unbounded growth. The common
# case is one name-set (the no_outliers file) shared by training and prediction.
EMBEDDING_CACHE_CAP = 8


def _names_key(names: pd.Series) -> str:
    """Stable content hash of the Name series — cache key for raw embeddings."""
    joined = "\n".join(names.fillna("").astype(str).tolist())
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def _cached_raw_embeddings(
    names: pd.Series, model_name: str, cache_dir: Path
) -> tuple[np.ndarray, bool]:
    """Return (raw_embeddings, cache_hit). Encode only on a cache miss.

    ponytail: cache keyed on the content hash of the Name series, so repeat
    main.py / predict.py runs on the same dataset skip the MiniLM encode entirely
    — the dominant cost. Capped at EMBEDDING_CACHE_CAP distinct name-sets (oldest
    evicted first via dict insertion order).
    """
    # ponytail: include model_name in the cache key so swapping encoders (e.g.
    # MiniLM → mpnet) re-encodes instead of silently reusing the stale raw vectors.
    key = f"{model_name}:{_names_key(names)}"
    cache: dict = {}
    if EMBEDDING_CACHE_PATH.exists():
        try:
            cache = joblib.load(EMBEDDING_CACHE_PATH)
        except Exception:
            cache = {}
    if isinstance(cache, dict) and key in cache:
        print(f"  Embedding cache hit (key={key}) — skipping encode")
        return cache[key], True

    encoder = SentenceTransformer(model_name, cache_folder=str(cache_dir))
    start = time.perf_counter()
    raw_embeddings = encoder.encode(
        names.fillna("").tolist(),
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    print(
        f"  [encode names] {time.perf_counter() - start:.2f}s"
        f" ({len(names)} rows, cache miss, key={key})"
    )

    cache[key] = raw_embeddings
    while len(cache) > EMBEDDING_CACHE_CAP:
        cache.pop(next(iter(cache)))  # ponytail: evict oldest
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(cache, EMBEDDING_CACHE_PATH)
    return raw_embeddings, False


def build_name_embeddings(
    names: pd.Series,
    model_name: str = MODEL_NAME,
    embedding_dim: int = EMBEDDING_DIM,
    cache_dir: Path | None = None,
    pca: PCA | None = None,
) -> tuple[pd.DataFrame, SentenceTransformer | None, PCA]:
    """Encode product names to dense embeddings and reduce them with PCA.

    ponytail: uses sentence-transformers' paraphrase-multilingual-MiniLM-L12-v2
    because the product names are Thai-English mixed. The raw 384-dim embeddings
    are reduced to embedding_dim PCA components. If a fitted `pca` is passed
    (predict path), it is used to transform — not refit — so train and predict
    share one PCA basis (fixes a latent basis mismatch and removes the refit
    cost). Otherwise a new PCA is fit and returned (training path).
    """
    cache_dir = cache_dir or Path.home() / ".cache" / "sentence_transformers"
    raw_embeddings, _ = _cached_raw_embeddings(names, model_name, cache_dir)

    if pca is not None:
        reduced = pca.transform(raw_embeddings)
        n_components = pca.n_components_
    else:
        # ponytail: fit-then-transform, NOT fit_transform. With the randomized SVD
        # solver, fit_transform(X) and fit(X).transform(X) differ by ~0.03 (truncated
        # SVD), which would silently shift predict-time features vs training-time
        # features. Using the same transform() code path in training and prediction
        # makes the two consistent on the same raw embeddings. random_state makes the
        # randomized fit reproducible; this basis recovers the original R²≈0.267.
        pca = PCA(
            n_components=embedding_dim,
            svd_solver="randomized",
            random_state=RANDOM_STATE,
        )
        pca.fit(raw_embeddings)
        reduced = pca.transform(raw_embeddings)
        n_components = embedding_dim

    columns = [f"name_semantic_pca_{i:02d}" for i in range(n_components)]
    embedding_df = pd.DataFrame(reduced, columns=columns, index=names.index)
    return embedding_df, None, pca
