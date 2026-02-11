# BioScholar Makefile
# Run `make help` to see available targets.

.PHONY: help train ingest eval eval-retrieval mlflow docker clean finetune-data finetune-llm eval-finemodel

PYTHON ?= python
MODEL_PATH ?= $(shell ls -d outputs/models/run_*/final_model 2>/dev/null | sort | tail -1)
COLLECTION ?= bio_guidelines
QDRANT_PATH ?= data/qdrant_db
LLM_MODEL ?= llama3.2

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

train:  ## Train the NER model
	$(PYTHON) src/train.py --config config.yaml

# ---------------------------------------------------------------------------
# Data Pipeline
# ---------------------------------------------------------------------------

download-pdfs:  ## Download sample open-access medical PDFs
	$(PYTHON) scripts/download_sample_pdfs.py

ingest:  ## Ingest PDFs into Qdrant vector store
	$(PYTHON) scripts/ingest_documents.py \
		--collection-name $(COLLECTION) \
		--model-path "$(MODEL_PATH)" \
		--qdrant-path $(QDRANT_PATH)

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

eval:  ## Run full evaluation (RAGAS + medical metrics + MLflow)
	$(PYTHON) eval/run_eval.py \
		--model-path "$(MODEL_PATH)" \
		--collection-name $(COLLECTION) \
		--qdrant-path $(QDRANT_PATH) \
		--llm-model $(LLM_MODEL)

eval-retrieval:  ## Run retrieval-only evaluation (no LLM needed)
	$(PYTHON) eval/run_eval.py \
		--skip-llm \
		--model-path "$(MODEL_PATH)" \
		--collection-name $(COLLECTION) \
		--qdrant-path $(QDRANT_PATH)

eval-ragas:  ## Run RAGAS evaluation only
	$(PYTHON) eval/ragas_eval.py \
		--model-path "$(MODEL_PATH)" \
		--collection-name $(COLLECTION) \
		--qdrant-path $(QDRANT_PATH)

# ---------------------------------------------------------------------------
# Phase 7: Fine-Tuning
# ---------------------------------------------------------------------------

finetune-data:  ## Generate training data from gold set + retrieval
	$(PYTHON) scripts/generate_finetune_data.py \
		--qdrant-path $(QDRANT_PATH) \
		--collection-name $(COLLECTION) \
		--model-path "$(MODEL_PATH)"

finetune-llm:  ## Fine-tune LLM with QLoRA (use --no-quantize on Apple Silicon)
	$(PYTHON) scripts/finetune_llm_qlora.py \
		--data data/finetune/medqa_train.jsonl \
		--model meta-llama/Llama-3.2-3B-Instruct \
		--no-quantize

eval-finemodel:  ## Evaluate fine-tuned RAG (set FINETUNE_ADAPTER path)
	$(PYTHON) eval/run_eval.py \
		--model-path "$(MODEL_PATH)" \
		--qdrant-path $(QDRANT_PATH) \
		--llm-adapter-path "$(FINETUNE_ADAPTER)"

# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------

mlflow:  ## Launch MLflow UI at http://localhost:5000
	mlflow ui --port 5000

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

docker:  ## Build and start Docker services (FastAPI + Qdrant)
	docker-compose up --build

docker-bg:  ## Build and start Docker services in background
	docker-compose up --build -d

docker-down:  ## Stop Docker services
	docker-compose down

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

serve:  ## Start FastAPI development server
	uvicorn app.main:app --reload --port 8000

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

test:  ## Run all tests
	pytest tests/ -v

test-cov:  ## Run tests with coverage
	pytest tests/ -v --cov=src --cov-report=term-missing

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

clean:  ## Remove generated files (caches, logs)
	rm -rf __pycache__ src/__pycache__ app/__pycache__ eval/__pycache__
	rm -rf .pytest_cache
	rm -rf mlruns
	find . -name "*.pyc" -delete
