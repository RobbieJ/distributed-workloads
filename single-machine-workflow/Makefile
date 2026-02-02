.PHONY: build up down clean shell \
       run-feast run-sft run-rag-hf run-rag-st run-feast-rag-ingest run-sft-feast-rag \
       run-deepspeed run-dreambooth run-hpo run-all-demos demo-report \
       test test-feast test-sft test-rag test-feast-rag-ingest test-sft-feast-rag \
       test-deepspeed test-dreambooth test-hpo

COMPOSE := docker compose -f docker-compose.yml -f docker-compose.gpu.yml
ARGS ?=

# --- Build & Infrastructure ---

build:
	$(COMPOSE) build

up:
	docker compose -f docker-compose.yml up -d

down:
	$(COMPOSE) down

clean:
	$(COMPOSE) down -v --remove-orphans

shell:
	$(COMPOSE) run --rm training bash

# --- Run Examples ---

run-feast:
	$(COMPOSE) run --rm training python scripts/01_feast_feature_store.py $(ARGS)

run-sft:
	$(COMPOSE) run --rm training python scripts/02_sft_llm.py $(ARGS)

run-rag-hf:
	$(COMPOSE) run --rm training python scripts/03_rag_huggingface.py $(ARGS)

run-rag-st:
	$(COMPOSE) run --rm training python scripts/03_rag_sentence_transformers.py $(ARGS)

run-feast-rag-ingest:
	$(COMPOSE) run --rm training python scripts/04_feast_rag_ingest.py $(ARGS)

run-sft-feast-rag:
	$(COMPOSE) run --rm training python scripts/05_sft_feast_rag.py $(ARGS)

run-deepspeed:
	$(COMPOSE) run --rm training python scripts/06_deepspeed_finetune.py $(ARGS)

run-dreambooth:
	$(COMPOSE) run --rm training python scripts/07_dreambooth.py $(ARGS)

run-hpo:
	$(COMPOSE) run --rm training python scripts/08_hpo_optuna.py $(ARGS)

create-dataset:
	$(COMPOSE) run --rm training python scripts/create_dataset.py $(ARGS)

run-all-demos:
	bash scripts/run_all_demos.sh $(ARGS)

demo-report:
	python scripts/report_all_demos.py

# --- Tests ---

test:
	bash tests/test_all.sh

test-feast:
	bash tests/test_01_feast.sh

test-sft:
	bash tests/test_02_sft.sh

test-rag:
	bash tests/test_03_rag.sh

test-feast-rag-ingest:
	bash tests/test_04_feast_rag_ingest.sh

test-sft-feast-rag:
	bash tests/test_05_sft_feast_rag.sh

test-deepspeed:
	bash tests/test_06_deepspeed.sh

test-dreambooth:
	bash tests/test_07_dreambooth.sh

test-hpo:
	bash tests/test_08_hpo.sh
