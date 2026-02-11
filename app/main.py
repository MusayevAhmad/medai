"""
BioScholar FastAPI Application

Medical RAG API that combines NER entity extraction with vector retrieval
and LLM answer generation.

Endpoints:
    GET  /health     - Health check
    POST /entities   - Extract medical entities from text
    POST /search     - Entity-filtered semantic search
    POST /query      - Full RAG pipeline (NER + retrieval + LLM answer)

Run:
    uvicorn app.main:app --reload --port 8000
"""

import json
import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware

from app.dependencies import (
    get_agent_graph,
    get_llm,
    get_ner,
    get_retriever,
    get_store,
    init_dependencies,
)
from app.schemas import (
    Citation,
    EntitiesRequest,
    EntitiesResponse,
    EntityOut,
    QueryRequest,
    QueryResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    VisualResult,
    VisualSearchRequest,
    VisualSearchResponse,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("bioscholar")

# ---------------------------------------------------------------------------
# Log paths
# ---------------------------------------------------------------------------
_LOG_DIR = Path("outputs/logs")
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_QUERY_LOG = _LOG_DIR / "queries.jsonl"
_API_REQUEST_LOG = _LOG_DIR / "api_requests.jsonl"

# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------

_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?prior", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+", re.IGNORECASE),
    re.compile(r"system\s*:\s*", re.IGNORECASE),
    re.compile(r"<\|?\s*system\s*\|?>", re.IGNORECASE),
]


def _check_prompt_injection(text: str) -> bool:
    """Return True if text contains likely prompt injection patterns."""
    return any(p.search(text) for p in _INJECTION_PATTERNS)


def _log_query(
    endpoint: str,
    request_data: dict,
    response_data: dict,
    request_id: str | None = None,
    agent_used: bool | None = None,
    agent_steps: int | None = None,
) -> None:
    """Append a query log entry to outputs/logs/queries.jsonl."""
    summary_keys = ("count", "retrieval_count", "model", "answer", "agent_used", "agent_steps")
    response_summary = {k: v for k, v in response_data.items() if k in summary_keys}
    if agent_used is not None:
        response_summary["agent_used"] = agent_used
    if agent_steps is not None:
        response_summary["agent_steps"] = agent_steps

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "request": request_data,
        "response_summary": response_summary,
    }
    if request_id:
        entry["request_id"] = request_id

    try:
        with open(_QUERY_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.warning("Failed to log query: %s", e)


def _log_api_request(
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
    request_id: str,
) -> None:
    """Append an API request log entry to outputs/logs/api_requests.jsonl."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "method": method,
        "path": path,
        "status_code": status_code,
        "duration_ms": round(duration_ms, 2),
    }

    try:
        with open(_API_REQUEST_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.warning("Failed to log API request: %s", e)


class APILoggingMiddleware(BaseHTTPMiddleware):
    """Log all API requests to outputs/logs/api_requests.jsonl.

    Logs method, path, status_code, duration_ms, and request_id.
    Request/response bodies are captured by endpoint-level _log_query
    for /entities, /search, /query, /visual-search.
    """

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        start = time.perf_counter()

        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000

        _log_api_request(
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
            request_id=request_id,
        )

        response.headers["X-Request-ID"] = request_id
        return response


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and connect to Qdrant on startup."""
    logger.info("Starting BioScholar API...")

    # LangSmith tracing (agent traces) — enabled when LANGCHAIN_TRACING_V2=true
    langsmith_enabled = os.environ.get("LANGCHAIN_TRACING_V2", "").lower() in ("true", "1")
    if langsmith_enabled:
        logger.info(
            "LangSmith tracing enabled (project=%s)",
            os.environ.get("LANGCHAIN_PROJECT", "bioscholar"),
        )
    else:
        logger.info(
            "LangSmith tracing disabled. Set LANGCHAIN_TRACING_V2=true and "
            "LANGCHAIN_API_KEY to enable agent trace logging."
        )

    # Build Qdrant URL from QDRANT_HOST/PORT if set (Docker), else use local path
    qdrant_host = os.environ.get("QDRANT_HOST")
    qdrant_port = os.environ.get("QDRANT_PORT", "6333")
    qdrant_url = f"http://{qdrant_host}:{qdrant_port}" if qdrant_host else None

    init_dependencies(
        model_path=os.environ.get("NER_MODEL_PATH") or None,
        collection_name=os.environ.get("QDRANT_COLLECTION", "bio_guidelines"),
        qdrant_path=os.environ.get("QDRANT_PATH", "data/qdrant_db"),
        qdrant_url=qdrant_url,
        llm_base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"),
        llm_model=os.environ.get("LLM_MODEL", "llama3.2"),
    )

    logger.info("BioScholar API ready!")
    yield
    logger.info("Shutting down BioScholar API.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="BioScholar",
    description="Medical RAG API — entity-aware retrieval with citations",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(APILoggingMiddleware)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Health check endpoint."""
    store = get_store()
    llm = get_llm()
    return {
        "status": "ok",
        "ner_loaded": get_ner() is not None,
        "collection_count": store.count(),
        "llm_available": llm.is_available(),
    }


@app.post("/entities", response_model=EntitiesResponse)
async def extract_entities(request: Request, req: EntitiesRequest):
    """Extract medical entities from text using the fine-tuned NER model."""
    if _check_prompt_injection(req.text):
        raise HTTPException(status_code=400, detail="Input rejected by safety filter.")

    ner = get_ner()
    entities = ner.predict_entities(req.text, threshold=req.threshold)

    result = EntitiesResponse(
        text=req.text,
        entities=[
            EntityOut(
                text=e.text,
                label=e.label,
                confidence=round(e.confidence, 4),
                span=list(e.span),
            )
            for e in entities
        ],
        count=len(entities),
    )

    _log_query(
        "/entities",
        req.model_dump(),
        result.model_dump(),
        request_id=getattr(request.state, "request_id", None),
    )
    return result


@app.post("/search", response_model=SearchResponse)
async def search(request: Request, req: SearchRequest):
    """Search ingested documents with optional entity filtering."""
    if _check_prompt_injection(req.query):
        raise HTTPException(status_code=400, detail="Input rejected by safety filter.")

    retriever = get_retriever()
    search_result = retriever.search(
        query=req.query,
        top_k=req.top_k,
        entity_filter=req.entity_filter,
        score_threshold=req.score_threshold,
    )

    result = SearchResponse(
        query=req.query,
        query_entities=[
            EntityOut(
                text=e.text,
                label=e.label,
                confidence=round(e.confidence, 4),
                span=list(e.span),
            )
            for e in search_result["query_entities"]
        ],
        results=[
            SearchResult(
                chunk_id=r["chunk_id"] or "",
                text=r["text"] or "",
                score=round(r["score"], 4),
                source_file=r["source_file"] or "",
                page_number=r["page_number"] or 0,
                extracted_entities=r.get("extracted_entities", []),
            )
            for r in search_result["results"]
        ],
        count=len(search_result["results"]),
    )

    _log_query(
        "/search",
        req.model_dump(),
        result.model_dump(),
        request_id=getattr(request.state, "request_id", None),
    )
    return result


@app.post("/query", response_model=QueryResponse)
async def query(request: Request, req: QueryRequest):
    """Full RAG pipeline: NER + retrieval + LLM answer generation.

    Automatically routes complex questions (comparisons, multi-entity)
    through the LangGraph agent for multi-step reasoning.  Simple
    questions use the fast single-pass pipeline.

    Set ``use_agent`` to override auto-detection:
    - ``null`` (default): auto-detect complexity
    - ``true``: always use the agent
    - ``false``: always use the fast pipeline
    """
    if _check_prompt_injection(req.question):
        raise HTTPException(status_code=400, detail="Input rejected by safety filter.")

    retriever = get_retriever()
    llm = get_llm()
    ner = get_ner()

    # ------------------------------------------------------------------
    # Decide whether to use the agent
    # ------------------------------------------------------------------
    agent_graph = get_agent_graph()

    if req.use_agent is True:
        use_agent = agent_graph is not None
    elif req.use_agent is False:
        use_agent = False
    else:
        # Auto-detect: extract entities and check complexity
        from src.agent import is_complex_query

        query_entities = ner.predict_entities(req.question, threshold=0.3)
        use_agent = agent_graph is not None and is_complex_query(
            req.question, query_entities
        )

    # ------------------------------------------------------------------
    # Agent path — multi-step reasoning
    # ------------------------------------------------------------------
    if use_agent and agent_graph is not None:
        from src.agent import run_agent

        logger.info("Routing to agent pipeline: %r", req.question)
        agent_result = run_agent(
            agent_graph,
            req.question,
            config={
                "metadata": {
                    "endpoint": "/query",
                    "use_agent": True,
                },
            },
        )

        answer = agent_result["answer"]
        agent_steps = agent_result["steps"]

        # Build citations from agent-accumulated structured data
        citations = [
            Citation(
                source_file=c.get("source_file", ""),
                page_number=c.get("page_number", 0),
                chunk_id=c.get("chunk_id", ""),
                score=c.get("score", 0.0),
                text_preview=c.get("text_preview", ""),
                extracted_entities=c.get("extracted_entities", []),
            )
            for c in agent_result["citations"]
        ]

        # Extract entities from the original question for the response
        query_entities_out = []
        try:
            entities = ner.predict_entities(req.question, threshold=0.3)
            query_entities_out = [
                EntityOut(
                    text=e.text,
                    label=e.label,
                    confidence=round(e.confidence, 4),
                    span=list(e.span),
                )
                for e in entities
            ]
        except Exception:
            pass

        result = QueryResponse(
            question=req.question,
            answer=answer,
            citations=citations,
            query_entities=query_entities_out,
            model=llm.model,
            retrieval_count=len(citations),
            agent_used=True,
            agent_steps=agent_steps,
        )

        _log_query(
            "/query",
            req.model_dump(),
            result.model_dump(),
            request_id=getattr(request.state, "request_id", None),
            agent_used=True,
            agent_steps=agent_steps,
        )
        return result

    # ------------------------------------------------------------------
    # Fast path — single-pass RAG pipeline
    # ------------------------------------------------------------------
    logger.info("Using fast pipeline: %r", req.question)

    # Step 1: Retrieve relevant chunks
    search_result = retriever.search(
        query=req.question,
        top_k=req.top_k,
        entity_filter=req.entity_filter,
        score_threshold=req.score_threshold,
    )

    chunks = search_result["results"]

    # Guardrail: check retrieval quality
    if not chunks or (chunks and chunks[0]["score"] < req.score_threshold):
        answer = "I don't have enough relevant information to answer this question."
        citations = []
    else:
        # Step 2: Generate answer with LLM
        try:
            answer = llm.generate_answer(
                question=req.question,
                context_chunks=chunks,
            )
        except ConnectionError as e:
            raise HTTPException(
                status_code=503,
                detail=str(e),
            )
        except RuntimeError as e:
            raise HTTPException(
                status_code=502,
                detail=f"LLM generation failed: {e}",
            )

        # Step 3: Build citations
        citations = [
            Citation(
                source_file=c["source_file"] or "",
                page_number=c["page_number"] or 0,
                chunk_id=c["chunk_id"] or "",
                score=round(c["score"], 4),
                text_preview=(c["text"] or "")[:200],
                extracted_entities=c.get("extracted_entities", []),
            )
            for c in chunks
        ]

    result = QueryResponse(
        question=req.question,
        answer=answer,
        citations=citations,
        query_entities=[
            EntityOut(
                text=e.text,
                label=e.label,
                confidence=round(e.confidence, 4),
                span=list(e.span),
            )
            for e in search_result["query_entities"]
        ],
        model=llm.model,
        retrieval_count=len(chunks),
        agent_used=False,
        agent_steps=None,
    )

    _log_query(
        "/query",
        req.model_dump(),
        result.model_dump(),
        request_id=getattr(request.state, "request_id", None),
        agent_used=False,
        agent_steps=None,
    )
    return result


@app.post("/visual-search", response_model=VisualSearchResponse)
async def visual_search(request: Request, req: VisualSearchRequest):
    """Search for text, tables, and figures with optional type filtering.

    Use this endpoint to find tables ("show me Table 2") or figures
    ("survival curve") alongside regular text results.
    """
    if _check_prompt_injection(req.query):
        raise HTTPException(status_code=400, detail="Input rejected by safety filter.")

    retriever = get_retriever()
    search_result = retriever.search(
        query=req.query,
        top_k=req.top_k,
        entity_filter=True,
    )

    results: list = []
    for r in search_result["results"]:
        chunk_type = r.get("chunk_type", "text")

        # Filter by requested chunk types
        if req.chunk_types and chunk_type not in req.chunk_types:
            continue

        results.append(VisualResult(
            chunk_id=r.get("chunk_id", ""),
            text=r.get("text", ""),
            score=round(r.get("score", 0.0), 4),
            source_file=r.get("source_file", ""),
            page_number=r.get("page_number", 0),
            chunk_type=chunk_type,
            image_path=r.get("image_path"),
            caption=r.get("caption"),
            extracted_entities=r.get("extracted_entities", []),
        ))

    tables_found = sum(1 for r in results if r.chunk_type == "table")
    figures_found = sum(1 for r in results if r.chunk_type == "figure")

    result = VisualSearchResponse(
        query=req.query,
        query_entities=[
            EntityOut(
                text=e.text,
                label=e.label,
                confidence=round(e.confidence, 4),
                span=list(e.span),
            )
            for e in search_result["query_entities"]
        ],
        results=results,
        count=len(results),
        tables_found=tables_found,
        figures_found=figures_found,
    )

    _log_query(
        "/visual-search",
        req.model_dump(),
        result.model_dump(),
        request_id=getattr(request.state, "request_id", None),
    )
    return result
