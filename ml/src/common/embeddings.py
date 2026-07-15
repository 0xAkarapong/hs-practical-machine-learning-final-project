"""Product-name sentence embeddings for the health & wellness price model."""

import hashlib
import os
import sys
import time
from pathlib import Path

# OMP=1 + n_jobs=1 exists only because torch threads + joblib *fork*
# (RFECV/RandomizedSearchCV n_jobs>1) segfault on macOS. On Linux (Docker) there
# is no fork here (joblib stays n_jobs=1), so multithreading is safe and a big
# win: force OMP=1 only on macOS; let Linux use all cores for the e5-base encode
# (the dominant runtime cost). TOKENIZERS_PARALLELISM=false stays on all platforms
# (avoids fast-tokenizer deadlock/warnings).
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
# Production uses 64 PCA components. Env-overridable so an encoder ablation can
# run at another dim (e.g. EMBEDDING_DIM=32) without touching this constant.
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "64"))
# e5 models REQUIRE a "query: "/"passage: " input prefix, even non-English. The
# e5 model card says: "Use 'query: ' prefix if you want to use embeddings as
# features, such as linear probing classification, clustering." We use the name
# embeddings as regression features (symmetric), so "query: " is the correct
# prefix — applied here in the single shared encode path so train + predict stay
# consistent.
E5_PREFIX = "query: "
# Cap distinct name-sets cached to avoid unbounded growth. The common case is
# one name-set (the no_outliers file) shared by training and prediction.
EMBEDDING_CACHE_CAP = 8


def _names_key(names: pd.Series) -> str:
    """Stable content hash of the Name series — cache key for raw embeddings."""
    joined = "\n".join(names.fillna("").astype(str).tolist())
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def _cached_raw_embeddings(
    names: pd.Series,
    model_name: str,
    cache_dir: Path,
    encoder: SentenceTransformer | None = None,
) -> tuple[np.ndarray, bool]:
    """Return (raw_embeddings, cache_hit). Encode only on a cache miss.

    Cache keyed on the content hash of the Name series, so repeat main.py /
    predict.py runs on the same dataset skip the e5 encode entirely — the dominant
    cost. Capped at EMBEDDING_CACHE_CAP distinct name-sets (oldest evicted first
    via dict insertion order).

    `encoder` lets a long-lived caller (the FastAPI server in api.py) pass in a
    resident SentenceTransformer so a per-request single listing does not reload
    the ~278M model each cache miss. Default None builds + tears down its own
    encoder, so the batch train/predict paths are unchanged.
    """
    # Include model_name in the cache key so swapping encoders (e.g. e5 → mpnet)
    # re-encodes instead of silently reusing the stale raw vectors.
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

    # Reuse an injected encoder; only construct + tear down a local one when no
    # encoder is supplied (the batch path).
    owns_encoder = encoder is None
    if owns_encoder:
        encoder = SentenceTransformer(model_name, cache_folder=str(cache_dir))
    # e5 needs the "query: " prefix on every input (see E5_PREFIX). For empty
    # names the prefix alone is harmless — encode keeps a fixed dim either way.
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

    # Free the torch model + its threads before any downstream joblib/XGBoost
    # stage so in-process threading is clean and memory is reclaimed. Raw
    # embeddings are plain numpy; nothing else needs the encoder. Only release an
    # encoder we own — an injected one is the caller's to keep.
    if owns_encoder:
        del encoder
        import gc

        gc.collect()

    cache[key] = raw_embeddings
    while len(cache) > EMBEDDING_CACHE_CAP:
        cache.pop(next(iter(cache)))  # evict oldest
    # Best-effort persist: the api container mounts models/ read-only, so a cache
    # miss there can't write back. Skipping is fine — the in-memory encode still
    # serves this request; only cross-restart reuse is lost on a read-only mount.
    try:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(cache, EMBEDDING_CACHE_PATH)
    except OSError:
        pass
    return raw_embeddings, False


def build_name_embeddings(
    names: pd.Series,
    model_name: str = MODEL_NAME,
    embedding_dim: int = EMBEDDING_DIM,
    cache_dir: Path | None = None,
    pca: PCA | None = None,
    encoder: SentenceTransformer | None = None,
) -> tuple[pd.DataFrame, SentenceTransformer | None, PCA]:
    """Encode product names to dense embeddings and reduce them with PCA.

    Uses sentence-transformers' intfloat/multilingual-e5-base (12-layer XLM-R,
    768-dim, MIT/non-gated) — lighter than bge-m3 (278M vs 568M, ~2x faster CPU
    encode) with strong Thai/English coverage. e5 REQUIRES a "query: " prefix on
    every input (applied in _cached_raw_embeddings); the raw 768-dim embeddings
    are reduced to embedding_dim PCA components. If a fitted `pca` is passed
    (predict path), it is used to transform — not refit — so train and predict
    share one PCA basis (fixes a latent basis mismatch and removes the refit
    cost). Otherwise a new PCA is fit and returned (training path).
    """
    cache_dir = cache_dir or Path.home() / ".cache" / "sentence_transformers"
    raw_embeddings, _ = _cached_raw_embeddings(
        names, model_name, cache_dir, encoder=encoder
    )

    if pca is not None:
        reduced = pca.transform(raw_embeddings)
        n_components = pca.n_components_
    else:
        # fit-then-transform, NOT fit_transform. With the randomized SVD solver,
        # fit_transform(X) and fit(X).transform(X) differ by ~0.03 (truncated SVD),
        # which would silently shift predict-time features vs training-time features.
        # Using the same transform() code path in training and prediction makes the
        # two consistent on the same raw embeddings. random_state makes the
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
