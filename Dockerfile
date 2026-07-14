FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH=/app/.venv/bin:$PATH

WORKDIR /app

# Install deps first so they cache across source changes
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# Copy source + data
COPY main.py ./
COPY src ./src
COPY dataset ./dataset

CMD ["python", "main.py"]