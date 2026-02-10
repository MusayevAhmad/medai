# BioScholar

An AI-powered clinical evidence research assistant that extracts medical entities from biomedical text and retrieves relevant information from clinical guidelines with citations.

The system combines a fine-tuned BioBERT NER model (for extracting diseases, chemicals, and symptoms) with a RAG pipeline that ingests PDFs, stores entity-enriched chunks in a vector database, and answers medical questions grounded in source documents.

## Current Status

**Phases 1–4 are complete.** The system includes: NER model, ingestion pipeline, FastAPI backend with entity-filtered retrieval, LLM answer generation (Ollama), Docker setup, RAGAS evaluation framework with MLflow tracking, and multimodal ingestion (tables + figures). Phase 5 (LangGraph Agent) is next. See [ROADMAP.md](ROADMAP.md) for the full plan.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare the BC5CDR dataset
python data/prepare_data.py

# 3. Train the NER model (~2 min on Apple Silicon, ~7 min on CPU)
python src/train.py --config config.yaml

# 4. Run inference
python src/predict.py --text "Patient presents with fever, headache, and was prescribed aspirin"

# Or use programmatically
python -c "
from src.inference import MedicalNER
ner = MedicalNER('outputs/models/<run_dir>/final_model')
entities = ner.predict_entities('Patient has fever and diabetes')
for e in entities: print(f'{e.label}: {e.text}')
"
```

### Ingestion Pipeline

```bash
# Start Qdrant (requires Docker)
docker run -p 6333:6333 qdrant/qdrant

# Ingest PDFs into the vector store
python scripts/ingest_documents.py --pdf-dir data/raw_pdfs/ --collection-name bio_guidelines
```

### FastAPI Backend

```bash
# Run the API server locally (requires Ollama for LLM generation)
uvicorn app.main:app --reload --port 8000

# API docs at http://localhost:8000/docs
# Endpoints: GET /health, POST /entities, POST /search, POST /query
```

### Docker Deployment

```bash
# Start the full stack (FastAPI + Qdrant)
docker-compose up

# Or build and run in background
docker-compose up -d --build

# API available at http://localhost:8000
# Qdrant dashboard at http://localhost:6333/dashboard
```

## Project Structure

```
medai/
├── config.yaml                # Training hyperparameters and model settings
├── config_xray.yaml           # X-ray classification training config
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Multi-stage build for the FastAPI app
├── docker-compose.yml         # FastAPI + Qdrant services
├── ROADMAP.md                 # Development roadmap (8 phases)
├── CONTEXT.md                 # Architecture decisions and coding conventions
│
├── app/                       # FastAPI backend (Phase 2)
│   ├── main.py                # API routes (/health, /entities, /search, /query)
│   ├── schemas.py             # Pydantic request/response models
│   └── dependencies.py        # Singleton dependency management
│
├── data/
│   ├── prepare_data.py        # Download and preprocess BC5CDR dataset
│   └── synthetic_symptoms.csv # Custom symptom examples
│
├── src/
│   ├── model.py               # BioBERT + LoRA model creation and loading
│   ├── dataset.py             # NER Dataset class with subword alignment
│   ├── train.py               # Training loop (HuggingFace Trainer)
│   ├── evaluate.py            # Evaluation metrics (seqeval, strict/partial)
│   ├── inference.py           # MedicalNER class for production inference
│   ├── predict.py             # CLI inference wrapper
│   ├── ingest.py              # PDF → chunks → NER → metadata pipeline
│   ├── vector_store.py        # Qdrant vector store (add, search, filtered search)
│   ├── retrieve.py            # Hybrid retrieval (NER entity filtering + semantic)
│   ├── llm.py                 # OpenAI-compatible LLM client for answer generation
│   ├── multimodal_ingest.py   # Phase 4: text + table + figure extraction from PDFs
│   ├── image_model.py         # X-ray CNN classifier (transfer learning)
│   ├── image_dataset.py       # X-ray image dataset and data loaders
│   ├── image_train.py         # X-ray model training script
│   └── image_predict.py       # X-ray inference
│
├── scripts/
│   └── ingest_documents.py    # Batch PDF ingestion CLI
│
├── notebooks/
│   ├── 01_explore_pretrained.ipynb
│   ├── 02_baseline_evaluation.ipynb
│   ├── 03_finetuned_comparison.ipynb
│   ├── 04_comprehensive_evaluation.ipynb
│   ├── 05_xray_exploration.ipynb
│   └── 06_xray_training.ipynb
│
├── streamlit_app/             # Streamlit UI (NER + X-ray demos)
│   ├── app.py
│   ├── pages/
│   └── utils/
│
├── tests/
│   ├── test_dataset.py
│   ├── test_inference.py
│   ├── test_ingest.py
│   ├── test_predict.py
│   └── test_vector_store.py
│
└── outputs/
    ├── models/                # Saved LoRA adapters per training run
    └── logs/                  # Query logs (queries.jsonl)
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| NER Model | BioBERT (`dmis-lab/biobert-base-cased-v1.2`) + LoRA |
| Fine-tuning | PEFT/LoRA (r=16, alpha=32, Q/K/V projections) |
| Training | HuggingFace Trainer with early stopping |
| Dataset | BC5CDR (1,500 PubMed abstracts, Chemical + Disease entities) |
| Document Parsing | PyMuPDF |
| Vector Database | Qdrant |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Backend | FastAPI |
| LLM Integration | OpenAI-compatible API (Ollama, OpenAI, etc.) |
| Deployment | Docker + Docker Compose (Azure Container Apps planned) |

## NER Training Results

Trained on BC5CDR with 7 epochs, LoRA on query/key/value projections, eval every 50 steps:

| Metric | Result |
|--------|--------|
| F1 Score | **0.8325** |
| Precision | 0.8210 |
| Recall | 0.8444 |
| Accuracy | 0.9600 |

Training time: ~2 minutes on M2 Pro (MPS), ~7 minutes on CPU.

## Example Output

```
Input: "Patient presents with fever, headache, and was prescribed aspirin"

Entities:
  Disease: fever          (confidence: 0.94)
  Disease: headache       (confidence: 0.91)
  Chemical: aspirin       (confidence: 0.96)
```

## Roadmap Overview

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Entity-Aware Ingestion Pipeline | Complete |
| 2 | FastAPI Backend + Entity-Filtered Retrieval | Complete |
| 3 | Evaluation Framework (RAGAS + MLflow) | Complete |
| 4 | Multimodal Upgrade (tables, figures) | Complete |
| 5 | LangGraph Agent (multi-step reasoning) | Next |
| 6 | Azure Deployment | Planned |
| 7 | Fine-Tuning Experiment (RAG vs RAG+FT) | Planned |
| 8 | Research Contribution | Planned |

See [ROADMAP.md](ROADMAP.md) for detailed task breakdowns.

## Hardware Requirements

- **Minimum**: 8GB RAM, any CPU
- **Recommended**: Apple Silicon Mac (MPS) or CUDA GPU
- **Qdrant**: Docker required for the vector database

## License

MIT License - for educational and research purposes.
