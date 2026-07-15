# Health & Wellness Price Recommender — Monorepo

Recommends THB prices for Thai health & wellness marketplace listings from
`Section`, `Shop Location`, `Name` text, and review signals. Two packages:

- **`ml/`** — the Python regression pipeline (clean → feature-engineer → select →
  train GBDT/XGBoost → interpret → compare) plus a FastAPI bridge (`api.py`) that
  serves single-listing predictions, model metrics, and form metadata. Full
  design notes, results, and per-step rationale: [`ml/Summary.md`](./ml/Summary.md).
- **`web/`** — a Next.js (App Router, TypeScript, Tailwind) frontend: a
  single-listing price predictor form + a model-metrics dashboard. It talks to the
  FastAPI bridge through Next route handlers (no browser-side CORS).

```
ml/   Python pipeline + FastAPI bridge (uv, Docker)
web/  Next.js frontend (npm)
```

## Quickstart — Docker (everything)

From the repo root (the `docker-compose.yml` orchestrates `pipeline` + `api` + `web`):

```bash
docker compose build          # builds hs-ml:latest (bakes the ~1GB e5 model) + the web image
docker compose up pipeline    # one-shot: train models into the `models` volume, then predict
docker compose up api web      # serve the API on :8000 and the frontend on :3000
```

Open `http://localhost:3000`, fill the form, get a predicted THB price; the metrics
panel shows baseline / GBDT / XGBoost R², RMSE, MAE.

Re-run `docker compose up pipeline` whenever `ml/` source changes to refresh the
persisted models the `api` service loads.

## Quickstart — local dev (no Docker)

Two terminals:

```bash
# 1) API bridge (loads the trained models in ml/models/)
cd ml && uv sync && uv run uvicorn api:app --reload --port 8000

# 2) frontend (proxies to http://localhost:8000 via web/.env.local)
cd web && npm install && npm run dev
```

The frontend defaults `API_URL=http://localhost:8000` (see `web/.env.local`).

## ML pipeline (local, no Docker)

```bash
cd ml
uv sync                       # or: uv sync --group dev (adds ruff + pytest)
uv run python main.py         # train + evaluate + interpret + compare
uv run python predict.py      # batch: generate ml/dataset/predictions.csv
uv run python api.py          # self-check: predict ml/dataset/sample_input.csv
uv run pytest                 # tests (needs --group dev)
```

## Makefile targets (repo root)

`make help` lists them. Highlights: `make train`, `make up` (api+web),
`make api-dev` / `make web-dev` (local dev), `make predict-sample`, `make metrics`,
`make clean` (wipes volumes → forces retrain).

## Notes

- The `models/` artifacts are gitignored and reproduced by the pipeline. If the
  source's encoder or PCA dim changes, retrain (`make train` or `docker compose up
  pipeline`) so the persisted models match what `api.py` loads.
- `ml/` keeps its own README with the original Docker/task detail; the canonical
  quickstart is this file.