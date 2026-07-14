# Health & Wellness Price-Recommendation Pipeline

Regression pipeline that recommends THB prices for Thai health & wellness
marketplace listings from `Section`, `Shop Location`, `Name` text, and review
signals. Full design notes, results, and the per-step rationale live in
[`Summary.md`](./Summary.md).

## Run with Docker

Prerequisite: Docker Desktop (or any container engine with Compose v2) running.

```bash
docker compose build          # first build downloads the ~1GB sentence-transformer model
docker compose up             # trains the pipeline, then predicts -> predictions.csv
```

`docker compose up` runs `python main.py && python predict.py` end-to-end:
clean → feature-engineer → split → select → train (GBDT + XGBoost) →
interpret (SHAP) → compare baseline → predict. Trained models and generated
CSVs persist in the `models` and `dataset` named volumes across runs.

### Common tasks

```bash
# Re-predict only, reusing the persisted trained models (no retrain)
docker compose run pipeline python predict.py

# Run a comparison or EDA script
docker compose run pipeline python -m src.comparisons.models
docker compose run pipeline python -m src.comparisons.gbdt_xgboost
docker compose run pipeline python -m src.comparisons.embeddings
docker compose run pipeline python -m src.eda

# Inspect persisted outputs
docker compose run --entrypoint sh pipeline -c "ls -la models dataset"

# Copy an output out of the volume
docker compose cp pipeline:/app/dataset/predictions.csv ./predictions.csv
```

### Reset

Named volumes are seeded from the image on first mount (so the raw input CSV is
present). If the raw dataset or code changes and you want a clean rebuild of
the artifacts:

```bash
docker compose down -v        # deletes the models + dataset volumes
docker compose build
docker compose up
```

## Run without Docker

```bash
uv sync                       # or: uv sync --group dev (adds ruff + pytest)
uv run python main.py         # train + evaluate + interpret + compare
uv run python predict.py      # generate dataset/predictions.csv
uv run pytest                 # tests (needs --group dev)
```