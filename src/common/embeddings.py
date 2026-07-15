"""Product-name sentence embeddings for the health & wellness price model."""

import hashlib
import os
import sys
import time
from pathlib import Path

# ponytail: the OMP=1 + n_jobs=1 ceiling exists only because torch threads +
# joblib *fork* (RFECV/RandomizedSearchCV n_jobs>1) segfault on macOS. On Linux
# (Docker) there is no fork here (joblib stays n_jobs=1), so multithreading is
# safe and a big win: force OMP=1 only on macOS; let Linux use all cores for the
# e5-base encode (the dominant runtime cost). TOKENIZERS_PARALLELISM=false stays
# on all platforms (avoids fast-tokenizer deadlock/warnings).
if sys.platform == "darwin":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

from src._03_split_data import RANDOM_STATE

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # src/common/ -> repo root
MODELS_DIR = PROJECT_ROOT / "models"
EMBEDDING_CACHE_PATH = MODELS_DIR / "name_embeddings.joblib"

MODEL_NAME = "intfloat/multilingual-e5-base"
EMBEDDING_DIM = 64
# ponytail: e5 models REQUIRE a "query: "/"passage: " input prefix, even non-English.
# The e5 model card says: "Use 'query: ' prefix if you want to use embeddings as
# features, such as linear probing classification, clustering." We use the name
# embeddings as regression features (symmetric), so "query: " is the correct prefix
# — applied here in the single shared encode path so train + predict stay consistent.
E5_PREFIX = "query: "
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
    # ponytail: e5 needs the "query: " prefix on every input (see E5_PREFIX). For
    # empty names the prefix alone is harmless — encode keeps a fixed dim either way.
    prefixed = [f"{E5_PREFIX}{n}" for n in names.fillna("").tolist()]
    start = time.perf_counter()
    raw_embeddings = encoder.encode(
        prefixed,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    print(
        f"  [encode names] {time.perf_counter() - start:.2f}s"
        f" ({len(names)} rows, cache miss, key={key})"
    )

    # ponytail: free the torch model + its threads before any downstream
    # joblib/XGBoost stage so in-process threading is clean and memory is
    # reclaimed. Raw embeddings are plain numpy; nothing else needs the encoder.
    del encoder
    import gc

    gc.collect()

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

    ponytail: uses sentence-transformers' intfloat/multilingual-e5-base (12-layer
    XLM-R, 768-dim, MIT/non-gated) — lighter than bge-m3 (278M vs 568M, ~2x faster
    CPU encode) with strong Thai/English coverage. e5 REQUIRES a "query: " prefix
    on every input (applied in _cached_raw_embeddings); the raw 768-dim embeddings
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
