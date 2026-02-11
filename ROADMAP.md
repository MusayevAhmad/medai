# BioScholar Development Roadmap

**Status Tracking**: 
- [ ] Not Started
- [~] In Progress  
- [x] Completed

---

## Phase 1: Entity-Aware Ingestion Pipeline (Week 1-2)
**Goal**: Make the existing NER model production-ready and build the data pipeline.

### Tasks
- [x] **Task 1.1**: Refactor `src/predict.py` into class-based `src/inference.py`
  - Create `MedicalNER` class with `load_model()` and `predict_entities(text)` methods
  - Add caching for the model (load once, reuse for all predictions)
  - Write unit test: `tests/test_inference.py`
  
- [x] **Task 1.2**: Build PDF ingestion pipeline
  - Create `src/ingest.py` with `extract_text_from_pdf(path)` using PyMuPDF
  - Implement semantic chunking: Split on section headers first, then by token count
  - Run NER on each chunk and store entities in metadata
  - Write integration test with a sample 5-page medical guideline PDF

- [x] **Task 1.3**: Set up Qdrant vector database
  - Install Qdrant in Docker: `docker run -p 6333:6333 qdrant/qdrant`
  - Create `src/vector_store.py` with `QdrantStore` class
  - Implement `add_chunks(chunks: List[Chunk])` method
  - Test: Insert 10 chunks and verify via Qdrant dashboard (`http://localhost:6333/dashboard`)

- [x] **Task 1.4**: Build baseline retrieval
  - Implement `search(query: str, top_k: int)` in `vector_store.py`
  - Use a simple embedding model: `sentence-transformers/all-MiniLM-L6-v2`
  - Test: Query "fever treatment" and verify it returns relevant chunks

- [x] **Task 1.5**: Create ingestion script
  - Write `scripts/ingest_documents.py` that processes all PDFs in `data/raw_pdfs/`
  - Add progress bar (use `tqdm`)
  - Add CLI arg: `--collection-name` to specify Qdrant collection

**Deliverable**: Run `python scripts/ingest_documents.py --collection-name bio_guidelines` successfully with 10 sample PDFs.

---

## Phase 2: FastAPI Backend + Entity-Filtered Retrieval (Week 3-4)
**Goal**: Build the API and implement "smart search" using NER metadata.

### Tasks
- [x] **Task 2.1**: Create FastAPI app structure
  - Set up `app/main.py` with health check endpoint
  - Create `app/schemas.py` with Pydantic models: `QueryRequest`, `QueryResponse`, `Citation`
  - Add `/entities`, `/search`, and `/query` endpoints with full implementation

- [x] **Task 2.2**: Implement hybrid retrieval
  - Created `src/retrieve.py` with `HybridRetriever` class
  - Process user query through NER to extract entities, build filter keys
  - Filter Qdrant results with entity overlap, automatic fallback to semantic search

- [x] **Task 2.3**: Add LLM answer generation
  - Created `src/llm.py` with `LLMClient` (OpenAI-compatible API — works with Ollama, OpenAI, etc.)
  - Prompt template enforces grounded answers with `[Source N]` citations
  - Parse citations from LLM response and map back to original PDF pages

- [x] **Task 2.4**: Add guardrails
  - Prompt injection detection with regex patterns in `app/main.py`
  - Retrieval quality check: Returns "I don't have enough relevant information" below score threshold
  - All queries and responses logged to `outputs/logs/queries.jsonl`

- [x] **Task 2.5**: Dockerize the application
  - Created `Dockerfile` (multi-stage build: deps → model weights → app)
  - Created `docker-compose.yml` (FastAPI + Qdrant services)
  - Run: `docker-compose up` to start the full stack

**Deliverable**: Working API at `http://localhost:8000/query` that answers medical questions with citations.

---

## Phase 3: Evaluation Framework (Week 5-6)
**Goal**: Measure answer quality using RAGAS and custom medical metrics.

### Tasks
- [x] **Task 3.1**: Create evaluation dataset
  - Built `data/eval/gold_set_v1.csv` with 10 seed Q&A pairs (4 categories)
  - Columns: `question`, `expected_answer`, `required_entities`, `category`, `source_documents`, `difficulty`
  - Cover categories: Dosage, Symptoms, Contraindications, Procedures
  - *Note*: Expand to 50 pairs as more domain knowledge is added

- [x] **Task 3.2**: Implement RAGAS evaluation
  - Created `eval/ragas_eval.py` with standalone RAGAS-style metrics
  - Metrics: Faithfulness, Answer Relevance, Context Recall, Context Precision
  - Generates JSON report: `outputs/eval_report_<timestamp>.json`

- [x] **Task 3.3**: Add custom medical metrics
  - **Entity Coverage**: % of required entities mentioned in answer
  - **Citation Accuracy**: Did the cited source file actually get retrieved?
  - **Safety Score**: Rule-based checker for dangerous advice patterns
  - Created `eval/medical_metrics.py`

- [x] **Task 3.4**: Set up experiment tracking
  - Integrated MLflow (local, self-hosted)
  - Created `eval/run_eval.py` that combines RAGAS + medical metrics + MLflow logging
  - View results: `mlflow ui --port 5000` (or `make mlflow`)

**Deliverable**: Run `make eval` to generate a metrics report comparing v1.0 vs v1.1 of the system.

---

## Phase 4: Multimodal Upgrade (Week 7-9)
**Goal**: Add table and chart understanding.

### Tasks
- [x] **Task 4.1**: Table extraction with PyMuPDF
  - Used PyMuPDF's built-in `find_tables()` for local table detection (no cloud API needed)
  - Convert extracted tables to Markdown format
  - Store tables as separate chunks with `chunk_type: "table"` metadata + caption detection

- [x] **Task 4.2**: Figure detection and extraction
  - PyMuPDF `get_images()` extracts figures, filtered by size (min 100x100px, 5KB)
  - Saved to `data/figures/` as PNG/JPEG (67 figures from 20 PDFs)
  - Automatic caption detection via "Figure N:" regex pattern matching
  - Store captions as searchable text with `chunk_type: "figure"` metadata + image path

- [x] **Task 4.3**: Visual retrieval endpoint
  - New `POST /visual-search` endpoint with chunk_type filtering
  - Returns text, tables, and figures with type metadata, captions, and image paths
  - Updated vector store payload to include `chunk_type`, `image_path`, `caption`

**Deliverable**: `POST /visual-search?chunk_types=["table"]` returns table content in Markdown format.

---

## Phase 5: LangGraph Agent (Week 10-12)
**Goal**: Multi-step reasoning for complex questions.

### Tasks
- [x] **Task 5.1**: Design agent workflow
  - ReAct-pattern StateGraph: agent → tools → agent (loop until final answer)
  - LLM-driven query decomposition via system prompt
  - Complexity auto-detection (regex patterns + NER entity counting)
  - Custom tool executor captures structured citations in state
  - `/query` endpoint upgraded with `use_agent` flag (auto/true/false)
  - Robust arg parsing handles small-model tool-call quirks

- [x] **Task 5.2**: Add tools
  - Implemented `src/agent/tools.py` with all three tool interfaces
  - `search_guidelines(query)`: Wraps existing hybrid retrieval + citations
  - `lookup_drug_interaction(drug1, drug2)`: RxNav external API call with graceful fallback
  - `summarize_section(doc_id, section)`: Document-filtered section synthesis + citations

- [ ] **Task 5.3**: Add observability
  - Integrate LangSmith or Arize Phoenix
  - Log the full agent trace (all steps, tool calls, intermediate results)

**Deliverable**: Agent can answer "What are the side effects of Drug A and how do they compare to Drug B?"

---

## Phase 6: Azure Deployment (Week 13-14)
**Goal**: Production deployment.

### Tasks
- [ ] **Task 6.1**: Migrate to Azure OpenAI
- [ ] **Task 6.2**: Deploy Qdrant to Azure Container Instances
- [ ] **Task 6.3**: Deploy FastAPI to Azure Container Apps
- [ ] **Task 6.4**: Set up Application Insights (logging + monitoring)
- [ ] **Task 6.5**: Obtain AI-102 certification

**Deliverable**: Live API endpoint with Swagger docs at `https://bioscholar.azurewebsites.net/docs`.

---

## Phase 7: Fine-Tuning Experiment (Week 15-17)
**Goal**: Compare RAG vs RAG+Fine-tuned model.

### Tasks
- [ ] **Task 7.1**: Generate training data from retrieved chunks
- [ ] **Task 7.2**: Fine-tune Llama-3-8B with QLoRA on medical Q&A
- [ ] **Task 7.3**: Evaluate: Base RAG vs Fine-tuned RAG
- [ ] **Task 7.4**: Publish model card on Hugging Face

**Deliverable**: Blog post: "Does Fine-Tuning Improve Medical RAG?"

---

## Phase 8: Research Contribution (Week 18-24)
**Goal**: Publish a technical report or contribute to open source.

### Ideas
1. **DSPy Optimization**: Use DSPy to auto-optimize the retrieval prompt
2. **Table RAG**: Write a paper comparing different table parsing methods
3. **Open Source**: Contribute a medical evaluation dataset to RAGAS

**Deliverable**: Technical report + GitHub PR to a major repo.
