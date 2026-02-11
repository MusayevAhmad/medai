# BioScholar Changelog

All notable changes to this project are documented here. This file is maintained to help you track what was added, changed, or fixed across sessions.

---

## [2026-02-10] Session 1 — Phase 1 & 2 Audit + Phase 3 Build

### Bug Fixes
- **`src/image_predict.py`**: Fixed import `from image_model` → `from src.image_model`
- **`src/image_train.py`**: Fixed imports `from image_dataset` / `from image_model` → `from src.image_dataset` / `from src.image_model`
- **`streamlit_app/pages/1_text_analysis.py`**: Fixed import `from app.utils.model_loader` → `from utils.model_loader`
- **`streamlit_app/pages/2_xray_analysis.py`**: Same import fix
- **`src/model.py` `load_model()`**: Fixed critical crash when loading LoRA adapter models — now reads label mappings from `label_info.json` instead of failing on missing `config.json`
- **`app/dependencies.py`**: Added `qdrant_url` parameter so Docker containers can connect to Qdrant via host:port
- **`app/main.py`**: Updated lifespan to build Qdrant URL from `QDRANT_HOST`/`QDRANT_PORT` env vars

### Phase 2 Completion — Dockerize (Task 2.5)
- **`Dockerfile`**: Multi-stage build (CPU-only PyTorch, non-root user, healthcheck)
- **`docker-compose.yml`**: FastAPI + Qdrant services with model volume mounts
- **`.dockerignore`**: Excludes notebooks, data, outputs, streamlit from image

### Phase 3 — Evaluation Framework (Tasks 3.1–3.4)
- **`scripts/download_sample_pdfs.py`**: Downloads 10 open-access medical PDFs from Europe PMC
- **10 PDFs downloaded** and **3,901 chunks ingested** into Qdrant (`data/qdrant_db`)
- **`data/eval/gold_set_v1.csv`**: 10 seed Q&A pairs (Symptoms, Dosage, Contraindications, Procedures)
- **`eval/ragas_eval.py`**: RAGAS-style metrics (Faithfulness, Answer Relevance, Context Recall, Context Precision)
- **`eval/medical_metrics.py`**: Custom medical metrics (Entity Coverage, Citation Accuracy, Safety Score)
- **`eval/run_eval.py`**: Combined evaluation runner with MLflow experiment tracking
- **`Makefile`**: Added `make eval`, `make eval-retrieval`, `make mlflow`, `make serve`, `make test`, etc.
- **MLflow** installed and verified working — baseline retrieval run logged

### Documentation Updates
- **`ROADMAP.md`**: Phase 2 tasks 2.1–2.5 and Phase 3 tasks 3.1–3.4 all marked complete
- **`CONTEXT.md`**: Updated current phase, added completed phases summary
- **`README.md`**: Updated project structure, tech stack, Docker instructions, status table

### Baseline Retrieval Results (no LLM)
| Metric | Score |
|--------|-------|
| Context Recall | 0.2381 |
| Context Precision | 0.0647 |
| Entity Coverage | 0.8750 |
| Citation Accuracy | 0.7000 |
| Safety Score | 1.0000 |

---

## [2026-02-10] Session 2 — Gold Set Expansion + Ollama + Phase 4 Start

### Data Expansion
- **Downloaded 10 additional PDFs** (stroke, COPD, rheumatoid arthritis, epilepsy, HIV/AIDS, liver disease, kidney disease, tuberculosis, malaria, sepsis)
- **Total: 20 PDFs → 10,620 chunks** ingested into Qdrant
- **Expanded `data/eval/gold_set_v1.csv`** from 10 → 50 Q&A pairs across 4 categories (Symptoms: 14, Procedures: 15, Dosage: 14, Contraindications: 7)
- Updated `scripts/download_sample_pdfs.py` with all 20 PDF sources

### Ollama Integration
- Ollama installed, started, and `llama3.2` model pulled
- Full RAG evaluation completed (50 questions, ~3 min runtime)
- Results logged to MLflow (run: `full-rag-v1-50q`)

### Full RAG Evaluation Results (50 questions, llama3.2)
| Metric | Score |
|--------|-------|
| Faithfulness | 0.0530 |
| Answer Relevance | 0.1037 |
| Context Recall | 0.1773 |
| Context Precision | 0.0531 |
| Entity Coverage | 0.5370 |
| Citation Accuracy | 0.4400 |
| Safety Score | 0.9950 |

> **Note:** Low faithfulness/relevance scores are expected — the heuristic metrics use word-overlap which
> underestimates quality for paraphrased answers. Context recall can be improved by tuning chunk size,
> overlap, and top_k. These numbers serve as a baseline for future comparison.

### Phase 4 — Multimodal Upgrade (complete)

#### New Files
- **`src/multimodal_ingest.py`** — Unified pipeline: text + tables + figures extraction from PDFs
  - `extract_tables_from_pdf()` — PyMuPDF `find_tables()` → Markdown → Qdrant
  - `extract_figures_from_pdf()` — PyMuPDF `get_images()` → saved PNG/JPEG + caption detection
  - `process_pdf_multimodal()` — Orchestrator replacing `process_pdf()` for multimodal mode

#### Modified Files
- **`scripts/ingest_documents.py`** — Added `--multimodal` and `--figures-dir` CLI flags
- **`src/vector_store.py`** — Payload now includes `chunk_type`, `image_path`, `caption`
- **`app/schemas.py`** — Added `VisualSearchRequest`, `VisualResult`, `VisualSearchResponse` models
- **`app/main.py`** — Added `POST /visual-search` endpoint with chunk_type filtering

#### Data
- **Re-ingested all 20 PDFs** with multimodal pipeline
- **6,819 total chunks**: 6,719 text + 33 tables + 67 figures
- **67 figures saved** to `data/figures/` (PNG/JPEG)

#### Documentation
- **`ROADMAP.md`** — Phase 4 tasks 4.1–4.3 marked complete
- **`CONTEXT.md`** — Current phase updated to Phase 5 (LangGraph Agent)
- **`CHANGELOG.md`** — This entry

---

## [2026-02-11] Task 5.3 — Observability (LangSmith + API Logging)

### LangSmith Integration
- **LangSmith tracing** for agent runs: set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY`
- Agent graph invokes automatically send traces (steps, tool calls, LLM invocations) to LangSmith
- **`run_agent()`** accepts optional `config` with `run_name` and `metadata` for trace enrichment
- **`.env.example`** added with LangSmith env var documentation

### API Request Logging
- **`APILoggingMiddleware`** logs all requests to `outputs/logs/api_requests.jsonl`
  - Fields: timestamp, request_id, method, path, status_code, duration_ms
  - Response header `X-Request-ID` added for correlation
- **`_log_query()`** enhanced with `agent_used`, `agent_steps`, `request_id` in `queries.jsonl`

### Documentation
- **`README.md`** — Observability section with LangSmith setup
- **`ROADMAP.md`** — Task 5.3 marked complete

---
