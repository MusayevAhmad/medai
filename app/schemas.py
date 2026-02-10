"""
Pydantic schemas for the BioScholar API.

Defines request/response models for the /query, /entities, and /search endpoints.
"""

from typing import List, Optional

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


# ---------------------------------------------------------------------------
# /visual-search endpoint (Phase 4 — multimodal)
# ---------------------------------------------------------------------------

class VisualSearchRequest(BaseModel):
    """Request body for the /visual-search endpoint."""

    query: str = Field(..., min_length=1, description="Search query (e.g. 'show me Table 2')")
    top_k: int = Field(5, ge=1, le=20, description="Number of results")
    chunk_types: Optional[List[str]] = Field(
        None,
        description="Filter by chunk type: 'text', 'table', 'figure'. None returns all types.",
    )


class VisualResult(BaseModel):
    """A single visual search result."""

    chunk_id: str
    text: str
    score: float
    source_file: str
    page_number: int
    chunk_type: str = Field("text", description="Type: text, table, or figure")
    image_path: Optional[str] = Field(None, description="Path to figure image (if chunk_type=figure)")
    caption: Optional[str] = Field(None, description="Table/figure caption")
    extracted_entities: List[str] = Field(default_factory=list)


class VisualSearchResponse(BaseModel):
    """Response from the /visual-search endpoint."""

    query: str
    query_entities: List[EntityOut]
    results: List[VisualResult]
    count: int
    tables_found: int = Field(0, description="Number of table results")
    figures_found: int = Field(0, description="Number of figure results")
