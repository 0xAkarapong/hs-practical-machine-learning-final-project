# Makefile — monorepo control. ML lives in ./ml (uv + docker), frontend in ./web.
# `docker compose` targets use the root compose file (pipeline + api + web).

.PHONY: help build up train predict predict-file predict-sample shell logs ps metrics figures down clean api api-dev web web-dev

help:  ## show available targets
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*##' '{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

build:  ## build all images (hs-ml:latest + web)
	docker compose build

up:  ## build + run api + web (assumes models volume already trained; run `make train` first)
	docker compose up --build api web

train:  ## one-shot: train the ML pipeline + predict into the models/dataset volumes
	docker compose up --abort-on-container-exit --exit-code-from pipeline pipeline

predict:  ## re-predict the full cleaned raw table using persisted models
	docker compose run --rm pipeline python predict.py

predict-file:  ## predict a custom CSV in ./ml/dataset: make predict-file FILE=my.csv
	docker compose run --rm -v "$(CURDIR)/ml/dataset:/app/io" pipeline \
		python predict.py /app/io/$(FILE) /app/io/$(FILE:.csv=_predictions.csv)
	@echo "Wrote ml/dataset/$(FILE:.csv=_predictions.csv)"

predict-sample:  ## predict for ml/dataset/sample_input.csv (5 demo rows)
	docker compose run --rm -v "$(CURDIR)/ml/dataset:/app/io" pipeline \
		python predict.py /app/io/sample_input.csv /app/io/sample_predictions.csv
	@echo "Wrote ml/dataset/sample_predictions.csv"

shell:  ## open a shell inside the ML container (volumes mounted)
	docker compose run --rm pipeline sh

logs:  ## tail logs from the last run
	docker compose logs --tail=200

ps:  ## show container status
	docker compose ps

metrics:  ## print ml/models/metrics.json from the volume
	docker compose run --rm pipeline cat /app/models/metrics.json

figures:  ## copy generated PNGs from the figures volume to ml/notebooks/figures
	@mkdir -p ml/notebooks/figures
	docker compose run --rm --no-deps -v $(PWD)/ml/notebooks/figures:/out pipeline \
		sh -c 'cp -r /app/notebooks/figures/. /out/'
	@echo "Copied figures -> ml/notebooks/figures/"

down:  ## stop + remove containers (keeps volumes)
	docker compose down --remove-orphans

clean:  ## wipe all volumes (models, dataset, figures) — forces retrain + re-encode
	docker compose down -v --remove-orphans
	@echo "Removed all volumes — next run re-trains from scratch."

# --- local dev (no Docker) ------------------------------------------------
api-dev:  ## run the FastAPI bridge locally with reload (from ml/)
	cd ml && uv run uvicorn api:app --reload --host 0.0.0.0 --port 8000

web-dev:  ## run the Next.js dev server (from web/)
	cd web && npm run dev