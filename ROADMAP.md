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

- [ ] **Task 1.4**: Build baseline retrieval
  - Implement `search(query: str, top_k: int)` in `vector_store.py`
  - Use a simple embedding model: `sentence-transformers/all-MiniLM-L6-v2`
  - Test: Query "fever treatment" and verify it returns relevant chunks

- [ ] **Task 1.5**: Create ingestion script
  - Write `scripts/ingest_documents.py` that processes all PDFs in `data/raw_pdfs/`
  - Add progress bar (use `tqdm`)
  - Add CLI arg: `--collection-name` to specify Qdrant collection

**Deliverable**: Run `python scripts/ingest_documents.py --collection-name bio_guidelines` successfully with 10 sample PDFs.

---

## Phase 2: FastAPI Backend + Entity-Filtered Retrieval (Week 3-4)
**Goal**: Build the API and implement "smart search" using NER metadata.

### Tasks
- [ ] **Task 2.1**: Create FastAPI app structure
  - Set up `app/main.py` with health check endpoint
  - Create `app/schemas.py` with Pydantic models: `QueryRequest`, `QueryResponse`, `Citation`
  - Add `/query` endpoint skeleton (returns mock data)

- [ ] **Task 2.2**: Implement hybrid retrieval
  - Extend `vector_store.py` with `search_with_filters(query, entity_filters)`
  - Process user query through NER to extract entities
  - Filter Qdrant results: Only return chunks where `metadata.entities` overlaps with query entities
  - Test: "What treats fever in children?" should prioritize chunks tagged with ["Fever", "Pediatric"]

- [ ] **Task 2.3**: Add LLM answer generation
  - Integrate OpenAI API (or Anthropic Claude) for answer synthesis
  - Prompt template: "Answer the question using ONLY the context below. Cite sources as [Source 1]."
  - Parse citations from LLM response and map back to original PDF pages

- [ ] **Task 2.4**: Add guardrails
  - Implement prompt injection detection (reject queries containing "ignore previous instructions")
  - Add retrieval quality check: If top result has score < 0.5, return "No relevant information found"
  - Log all queries and responses to `outputs/logs/queries.jsonl`

- [ ] **Task 2.5**: Dockerize the application
  - Create `Dockerfile` (multi-stage build: deps → model weights → app)
  - Create `docker-compose.yml` (FastAPI + Qdrant services)
  - Test: `docker-compose up` should start the full stack

**Deliverable**: Working API at `http://localhost:8000/query` that answers medical questions with citations.

---

## Phase 3: Evaluation Framework (Week 5-6)
**Goal**: Measure answer quality using RAGAS and custom medical metrics.

### Tasks
- [ ] **Task 3.1**: Create evaluation dataset
  - Build `data/eval/gold_set_v1.csv` with 50 Q&A pairs
  - Columns: `question`, `expected_answer`, `required_entities`, `source_documents`
  - Cover categories: Dosage, Symptoms, Contraindications, Procedures

- [ ] **Task 3.2**: Implement RAGAS evaluation
  - Create `eval/ragas_eval.py` script
  - Metrics: Faithfulness, Answer Relevance, Context Recall
  - Generate report: `outputs/eval_report_v1.json`

- [ ] **Task 3.3**: Add custom medical metrics
  - **Entity Coverage**: % of required entities mentioned in answer
  - **Citation Accuracy**: Did the cited page actually contain the information?
  - **Safety Score**: Does the answer include dangerous advice? (Use a rule-based checker)

- [ ] **Task 3.4**: Set up experiment tracking
  - Integrate Weights & Biases or MLflow
  - Log: Retrieval parameters (top_k, score threshold), LLM model, metrics

**Deliverable**: Run `make eval` to generate a metrics report comparing v1.0 vs v1.1 of the system.

---

## Phase 4: Multimodal Upgrade (Week 7-9)
**Goal**: Add table and chart understanding.

### Tasks
- [ ] **Task 4.1**: Integrate LlamaParse for table extraction
  - Replace PyMuPDF with LlamaParse for documents with tables
  - Convert extracted tables to Markdown format
  - Store tables as separate chunks with `chunk_type: "table"` metadata

- [ ] **Task 4.2**: Add figure detection
  - Use a YOLO model to detect "Figure X" regions in PDFs
  - Crop images and save to `data/figures/`
  - Generate captions using a vision model (LLaVA or GPT-4V)
  - Store captions as searchable text with `chunk_type: "figure"` metadata

- [ ] **Task 4.3**: Implement visual retrieval
  - When user asks "show me the survival curve", retrieve the figure image
  - Return both the caption AND the image URL in the API response

**Deliverable**: System can answer "What does Table 2 show?" by retrieving the table content.

---

## Phase 5: LangGraph Agent (Week 10-12)
**Goal**: Multi-step reasoning for complex questions.

### Tasks
- [ ] **Task 5.1**: Design agent workflow
  - Decompose: "Compare Drug A and Drug B" → [Query 1: "Drug A", Query 2: "Drug B"]
  - Implement using LangGraph's StateGraph

- [ ] **Task 5.2**: Add tools
  - `search_guidelines(query)`: Existing retrieval
  - `lookup_drug_interaction(drug1, drug2)`: External API call
  - `summarize_section(doc_id, section)`: Extract specific section

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

