# Makefile — control the ML pipeline Docker container.
# Service `pipeline` (image hs-ml:latest) runs: train (main.py) + predict (predict.py).
# Artifacts persist in named volumes: models, dataset, figures.

.PHONY: help build up train predict shell logs ps down clean metrics figures clean-models

help:  ## show available targets
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*##' '{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

build:  ## build the hs-ml:latest image (deps + model bake cached after first run)
	docker compose build

up:  ## run the full pipeline + predict, stream logs, exit on completion
	docker compose up --abort-on-container-exit --exit-code-from pipeline

train:  ## run only training (main.py) in a fresh container
	docker compose run --rm pipeline python main.py

predict:  ## run only predict.py (uses trained models in the models volume)
	docker compose run --rm pipeline python predict.py

shell:  ## open a shell inside the container (volumes mounted)
	docker compose run --rm pipeline sh

logs:  ## tail logs from the last run
	docker compose logs --tail=200

ps:  ## show container status
	docker compose ps

metrics:  ## print models/metrics.json from the volume
	docker compose run --rm pipeline cat /app/models/metrics.json

figures:  ## copy generated PNGs from the figures volume to ./notebooks/figures
	@mkdir -p notebooks/figures
	docker compose run --rm --no-deps -v $(PWD)/notebooks/figures:/out pipeline \
		sh -c 'cp -r /app/notebooks/figures/. /out/'
	@echo "Copied figures -> notebooks/figures/"

down:  ## stop + remove containers + network (keeps volumes)
	docker compose down --remove-orphans

clean-models:  ## wipe the models volume (forces retrain + re-encode on next run)
	docker compose down -v --remove-orphans
	@echo "Removed all volumes (models, dataset, figures) — next run re-trains from scratch."

clean: clean-models  ## alias of clean-models