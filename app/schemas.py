"""
Pydantic schemas for the BioScholar API.

Defines request/response models for the /query, /entities, and /search endpoints.
"""

from typing import List, Optional, Any, Dict

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared sub-models
# ---------------------------------------------------------------------------

class EntityOut(BaseModel):
    """A single extracted medical entity."""

    text: str = Field(..., description="Entity text as it appears in the input")
    label: str = Field(..., description="Entity type (Disease, Chemical, Symptom)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    span: List[int] = Field(..., min_length=2, max_length=2, description="[start, end] character offsets")


class Citation(BaseModel):
    """Source citation for a retrieved chunk."""

    source_file: str = Field(..., description="Original PDF filename")
    page_number: int = Field(..., description="1-indexed page number")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    score: float = Field(..., description="Retrieval similarity score")
    text_preview: str = Field(..., description="First 200 chars of the chunk")
    extracted_entities: List[str] = Field(default_factory=list, description="Entities found in this chunk")


# ---------------------------------------------------------------------------
# /entities endpoint
# ---------------------------------------------------------------------------

class EntitiesRequest(BaseModel):
    """Request body for the /entities endpoint."""

    text: str = Field(..., min_length=1, description="Text to extract entities from")
    threshold: float = Field(0.3, ge=0.0, le=1.0, description="Minimum confidence threshold")


class EntitiesResponse(BaseModel):
    """Response from the /entities endpoint."""

    text: str
    entities: List[EntityOut]
    count: int


# ---------------------------------------------------------------------------
# /search endpoint
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    """Request body for the /search endpoint."""

    query: str = Field(..., min_length=1, description="Search query")
    top_k: int = Field(5, ge=1, le=20, description="Number of results to return")
    entity_filter: bool = Field(True, description="Whether to use NER-based entity filtering")
    score_threshold: float = Field(0.0, ge=0.0, le=1.0, description="Minimum similarity score")


class SearchResult(BaseModel):
    """A single search result."""

    chunk_id: str
    text: str
    score: float
    source_file: str
    page_number: int
    extracted_entities: List[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    """Response from the /search endpoint."""

    query: str
    query_entities: List[EntityOut]
    results: List[SearchResult]
    count: int


# ---------------------------------------------------------------------------
# /query endpoint (full RAG pipeline)
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """Request body for the /query endpoint."""

    question: str = Field(..., min_length=1, description="Medical question to answer")
    top_k: int = Field(5, ge=1, le=20, description="Number of context chunks to retrieve")
    entity_filter: bool = Field(True, description="Use NER-based entity filtering")
    score_threshold: float = Field(0.3, ge=0.0, le=1.0, description="Minimum retrieval score")
    use_agent: Optional[bool] = Field(
        None,
        description=(
            "Route through the LangGraph agent for multi-step reasoning. "
            "None = auto-detect (uses agent for comparison/complex queries), "
            "True = always use agent, False = never use agent."
        ),
    )


class QueryResponse(BaseModel):
    """Response from the /query endpoint."""

    question: str
    answer: str
    citations: List[Citation]
    query_entities: List[EntityOut]
    model: str = Field(..., description="LLM model used for answer generation")
    retrieval_count: int = Field(..., description="Number of chunks used as context")
    agent_used: bool = Field(False, description="Whether the LangGraph agent was used")
    agent_steps: Optional[int] = Field(None, description="Number of agent reasoning steps (if agent was used)")
    agent_trace: Optional[List[dict]] = Field(None, description="Full agent execution trace (messages)")


# ---------------------------------------------------------------------------
# /visual-search endpoint
# ---------------------------------------------------------------------------

class VisualSearchRequest(BaseModel):
    """Request body for /visual-search."""
    query: str
    top_k: int = 5
    chunk_types: Optional[List[str]] = None


class VisualResult(BaseModel):
    """Visual search result."""
    chunk_id: str
    text: str
    score: float
    source_file: str
    page_number: int
    chunk_type: str
    image_path: Optional[str] = None
    image_url: Optional[str] = None
    caption: Optional[str] = None
    extracted_entities: List[str] = []


class VisualSearchResponse(BaseModel):
    """Response for /visual-search."""
    query: str
    query_entities: List[EntityOut]
    results: List[VisualResult]
    count: int
    tables_found: int
    figures_found: int


# ---------------------------------------------------------------------------
# /ingest endpoint
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    """Request body for /ingest."""
    collection_name: Optional[str] = None
    qdrant_path: str = "data/qdrant_db"
    model_path: Optional[str] = None
    pdf_dir: str = "data/raw_pdfs"
    figures_dir: str = "data/figures"
    multimodal: bool = True
    max_tokens: int = 500
    overlap: int = 50
    threshold: float = 0.0
    batch_size: int = 64


class IngestResponse(BaseModel):
    """Response for /ingest."""
    pdf_count: int
    pdfs: List[str]
    chunks_inserted: int
    chunk_type_counts: Dict[str, int]
    collection_name: str
    figures_dir: str
    duration_ms: float


# ---------------------------------------------------------------------------
# /analyze-image endpoint
# ---------------------------------------------------------------------------

class AnalyzeImageRequest(BaseModel):
    """Request body for /analyze-image."""
    image_base64: str = Field(..., description="Base64 encoded image")
    prompt: str = Field(..., description="Question or instruction")
    model: Optional[str] = Field(None, description="Model override (e.g. llama3.2-vision)")


class AnalyzeImageResponse(BaseModel):
    """Response for /analyze-image."""
    analysis: str
    model: str
