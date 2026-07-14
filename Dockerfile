# ---- builder: install deps + bake model, no wheel cache written to layer ----
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH=/app/.venv/bin:$PATH \
    MPLBACKEND=Agg \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1

WORKDIR /app

# Install deps first so they cache across source changes.
# --no-cache: uv uses a temp dir, writes no ~3.4GB ~/.cache/uv to the layer.
# --no-dev: skip ruff/pytest (not needed at runtime).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev --no-cache

# Pre-bake the sentence-transformer model into the image so runtime runs
# offline / without re-downloading ~1GB each container start. Cache path
# matches src/common/embeddings.py (~/.cache/sentence_transformers). Needs
# network at build; drop this RUN to skip baking (then runtime downloads).
RUN python -c "from pathlib import Path; from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('paraphrase-multilingual-mpnet-base-v2', \
    cache_folder=str(Path.home() / '.cache' / 'sentence_transformers'))"

# ---- runtime: slim, only the venv + baked model + source ----
# python:3.13-slim == the uv builder's base CPython build, so the .venv copies cleanly.
FROM python:3.13-slim

# libgomp1: OpenMP runtime for torch + sklearn (only system lib needed; Agg is headless).
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

ENV PATH=/app/.venv/bin:$PATH \
    MPLBACKEND=Agg \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=1

WORKDIR /app

# venv + baked model only (HF hub cache stays in builder, discarded -> no duplication).
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /root/.cache/sentence_transformers /root/.cache/sentence_transformers

# Copy source + data
COPY main.py predict.py ./
COPY src ./src
COPY dataset ./dataset

CMD ["python", "main.py"]