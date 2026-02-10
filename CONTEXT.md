# BioScholar: Multimodal Medical RAG System

## Project Overview
You are working on BioScholar, an AI-powered clinical evidence research assistant. The system uses a fine-tuned BioBERT NER model to extract medical entities (diseases, symptoms, chemicals) and combines it with RAG (Retrieval Augmented Generation) to answer medical questions with citations to Clinical Practice Guidelines (CPGs) and PubMed papers.

## Tech Stack
- **Base NER Model**: BioBERT (`dmis-lab/biobert-base-cased-v1.2`) with LoRA fine-tuning
- **Backend**: FastAPI (Python 3.10+)
- **Vector Database**: Qdrant (local mode initially, cloud later)
- **Document Parsing**: PyMuPDF for text, LlamaParse for tables (Phase 2)
- **Evaluation**: RAGAS framework
- **Deployment**: Docker → Azure Container Apps (Phase 4)

## Architecture Principles
1. **Modularity**: Keep NER inference, ingestion, retrieval, and API layers separate
2. **Entity-Aware Search**: Use BioBERT to extract entities from queries AND documents to enable metadata-filtered retrieval
3. **Citation Tracking**: Every chunk stored in Qdrant must include `source_file`, `page_number`, and `extracted_entities` metadata
4. **Fail-Safe**: If retrieval returns no results, return "I don't have information about that" instead of hallucinating

## Code Style
- Use type hints for all function signatures
- Prefer Pydantic models for API schemas and configs
- Use `pathlib.Path` instead of string paths
- Keep functions under 50 lines (extract helpers if needed)
- Add docstrings to all public functions (Google style)

## File Organization Rules
- `src/model.py`: Contains ONLY the BioBERT+LoRA model class. Do not add ingestion logic here.
- `src/inference.py`: The NER prediction engine. Must be importable by other modules (not just CLI).
- `src/ingest.py`: PDF → chunks → NER → Qdrant pipeline. This is the "data prep" layer.
- `src/retrieve.py`: Handles hybrid search (entity filtering + semantic similarity).
- `src/llm.py`: OpenAI-compatible LLM client for answer generation. No business logic.
- `src/vector_store.py`: Qdrant CRUD (add_chunks, search, search_with_filters).
- `app/main.py`: FastAPI routes. Keep business logic OUT of this file (call functions from src/).
- `app/schemas.py`: Pydantic request/response models.
- `app/dependencies.py`: Singleton management for NER, vector store, retriever, LLM.
- `eval/`: Evaluation scripts must be runnable via `make eval` command.

## Current Phase: Phase 5 (LangGraph Agent)
**Goal**: Multi-step reasoning for complex questions.

### Completed Phases
- **Phase 1** (Entity-Aware Ingestion): NER inference, PDF ingestion, Qdrant vector store, retrieval, ingestion script — all done.
- **Phase 2** (FastAPI Backend): FastAPI app (`app/main.py`), hybrid retrieval (`src/retrieve.py`), LLM integration (`src/llm.py`), guardrails, query logging, Dockerfile + docker-compose — all done.
- **Phase 3** (Evaluation Framework): RAGAS-style metrics, custom medical metrics, MLflow tracking, 50-question gold set — all done. Ollama running with llama3.2.
- **Phase 4** (Multimodal Upgrade): Table extraction (33 tables), figure extraction (67 figures from 20 PDFs), visual retrieval API endpoint — all done.

### Immediate Tasks
1. Design agent workflow with LangGraph StateGraph
2. Add tools: search_guidelines, lookup_drug_interaction, summarize_section
3. Add observability (LangSmith or Arize Phoenix)

## Dependency Management
- Add new dependencies to `requirements.txt` with version pinning (e.g., `qdrant-client==1.7.0`)
- Group dependencies with comments: `# NER`, `# RAG`, `# API`, `# Eval`

## Testing Strategy
- Unit tests: `tests/test_inference.py` (mock the model), `tests/test_ingest.py` (use a dummy 1-page PDF)
- Integration test: `tests/test_e2e.py` (ingest → retrieve → answer pipeline)
- Run tests via `pytest tests/ -v`

## What NOT to Do
- Do NOT use LangChain in Phase 1 (we add it in Phase 3 for agents)
- Do NOT implement multimodal features yet (tables/charts are Phase 2)
- Do NOT connect to OpenAI API until Phase 2 (use the NER model only for now)
- Do NOT write a UI/frontend yet (FastAPI backend only)

## When You're Stuck
1. Check if the task belongs in the current phase (don't jump ahead)
2. If a file grows beyond 200 lines, ask how to split it
3. If you need to install a new library, ask if there's a lighter alternative first
4. Always provide a minimal working example when suggesting code changes

## Cursor-Specific Instructions
- When editing files, show ONLY the changed sections (use `// ... existing code ...` for omitted parts)
- If I paste an error message, show the fix AND explain why the error happened
- When I say "next step", implement the next unchecked task from the roadmap
- Use Agent Mode for multi-file changes (e.g., refactoring `predict.py` into `inference.py`)
