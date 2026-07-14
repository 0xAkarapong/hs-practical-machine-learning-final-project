"""Product-name sentence embeddings for the health & wellness price model."""

import os
from pathlib import Path

# ponytail: disable tokenizer and OpenMP parallelism to avoid segfaults on macOS
# when sentence-transformers/torch interacts with joblib multiprocessing.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 16


def build_name_embeddings(
    names: pd.Series,
    model_name: str = MODEL_NAME,
    embedding_dim: int = EMBEDDING_DIM,
    cache_dir: Path | None = None,
) -> tuple[pd.DataFrame, SentenceTransformer, PCA]:
    """Encode product names to dense embeddings and reduce them with PCA.

    ponytail: uses sentence-transformers' paraphrase-multilingual-MiniLM-L12-v2
    because the product names are Thai-English mixed. The raw 384-dim embeddings
    are reduced to 16 PCA components for the small modeling table. The fitted PCA
    is returned so it can be persisted and reused at prediction time.
    """
    cache_dir = cache_dir or Path.home() / ".cache" / "sentence_transformers"
    encoder = SentenceTransformer(model_name, cache_folder=str(cache_dir))
    raw_embeddings = encoder.encode(
        names.fillna("").tolist(),
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    pca = PCA(n_components=embedding_dim, random_state=42)
    reduced = pca.fit_transform(raw_embeddings)

    columns = [f"name_semantic_pca_{i:02d}" for i in range(embedding_dim)]
    embedding_df = pd.DataFrame(reduced, columns=columns, index=names.index)
    return embedding_df, encoder, pca
