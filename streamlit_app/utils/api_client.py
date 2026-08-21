from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx


class BioScholarAPIError(RuntimeError):
    pass


@dataclass(frozen=True)
class BioScholarHealth:
    status: str
    ner_loaded: bool
    collection_count: int
    llm_available: bool


@dataclass(frozen=True)
class EntityOut:
    text: str
    label: str
    confidence: float
    span: List[int]


@dataclass(frozen=True)
class Citation:
    source_file: str
    page_number: int
    chunk_id: str
    score: float
    text_preview: str
    extracted_entities: List[str]


@dataclass(frozen=True)
class QueryResponse:
    question: str
    answer: str
    citations: List[Citation]
    query_entities: List[EntityOut]
    model: str
    retrieval_count: int
    agent_used: bool
    agent_steps: Optional[int]
    agent_trace: Optional[List[Dict[str, Any]]]


@dataclass(frozen=True)
class VisualResult:
    chunk_id: str
    text: str
    score: float
    source_file: str
    page_number: int
    chunk_type: str
    image_path: Optional[str]
    image_url: Optional[str]
    caption: Optional[str]
    extracted_entities: List[str]


@dataclass(frozen=True)
class VisualSearchResponse:
    query: str
    query_entities: List[EntityOut]
    results: List[VisualResult]
    count: int
    tables_found: int
    figures_found: int


def _base_url() -> str:
    return os.environ.get("BIOSCHOLAR_API_URL", "http://localhost:8000").rstrip("/")


def _raise_for_status(resp: httpx.Response) -> None:
    if resp.status_code < 400:
        return
    detail = ""
    try:
        payload = resp.json()
        detail = payload.get("detail") if isinstance(payload, dict) else ""
    except Exception:
        detail = resp.text
    raise BioScholarAPIError(f"API error {resp.status_code}: {detail}".strip())


def _as_entity_list(payload: Any) -> List[EntityOut]:
    if not isinstance(payload, list):
        return []
    out: List[EntityOut] = []
    for e in payload:
        if not isinstance(e, dict):
            continue
        out.append(
            EntityOut(
                text=str(e.get("text", "")),
                label=str(e.get("label", "")),
                confidence=float(e.get("confidence", 0.0)),
                span=list(e.get("span", [0, 0])),
            )
        )
    return out


def _as_citations(payload: Any) -> List[Citation]:
    if not isinstance(payload, list):
        return []
    out: List[Citation] = []
    for c in payload:
        if not isinstance(c, dict):
            continue
        out.append(
            Citation(
                source_file=str(c.get("source_file", "")),
                page_number=int(c.get("page_number", 0) or 0),
                chunk_id=str(c.get("chunk_id", "")),
                score=float(c.get("score", 0.0) or 0.0),
                text_preview=str(c.get("text_preview", "")),
                extracted_entities=list(c.get("extracted_entities", []) or []),
            )
        )
    return out


def get_client(timeout_s: float = 60.0) -> httpx.Client:
    return httpx.Client(base_url=_base_url(), timeout=timeout_s)


def health(client: httpx.Client) -> BioScholarHealth:
    resp = client.get("/health")
    _raise_for_status(resp)
    data = resp.json()
    return BioScholarHealth(
        status=str(data.get("status", "")),
        ner_loaded=bool(data.get("ner_loaded", False)),
        collection_count=int(data.get("collection_count", 0) or 0),
        llm_available=bool(data.get("llm_available", False)),
    )


def entities(client: httpx.Client, text: str, threshold: float = 0.3) -> List[EntityOut]:
    resp = client.post("/entities", json={"text": text, "threshold": threshold})
    _raise_for_status(resp)
    data = resp.json()
    return _as_entity_list(data.get("entities"))


def query(
    client: httpx.Client,
    question: str,
    top_k: int = 5,
    score_threshold: float = 0.3,
    entity_filter: bool = True,
    use_agent: Optional[bool] = None,
) -> QueryResponse:
    payload: Dict[str, Any] = {
        "question": question,
        "top_k": top_k,
        "entity_filter": entity_filter,
        "score_threshold": score_threshold,
        "use_agent": use_agent,
    }
    resp = client.post("/query", json=payload)
    _raise_for_status(resp)
    data = resp.json()
    return QueryResponse(
        question=str(data.get("question", question)),
        answer=str(data.get("answer", "")),
        citations=_as_citations(data.get("citations")),
        query_entities=_as_entity_list(data.get("query_entities")),
        model=str(data.get("model", "")),
        retrieval_count=int(data.get("retrieval_count", 0) or 0),
        agent_used=bool(data.get("agent_used", False)),
        agent_steps=data.get("agent_steps", None),
        agent_trace=data.get("agent_trace", None),
    )


def visual_search(
    client: httpx.Client,
    query_text: str,
    top_k: int = 5,
    chunk_types: Optional[List[str]] = None,
) -> VisualSearchResponse:
    payload: Dict[str, Any] = {"query": query_text, "top_k": top_k, "chunk_types": chunk_types}
    resp = client.post("/visual-search", json=payload)
    _raise_for_status(resp)
    data = resp.json()

    base = _base_url()
    results: List[VisualResult] = []
    for r in (data.get("results") or []):
        if not isinstance(r, dict):
            continue
        image_url = r.get("image_url") or None
        if isinstance(image_url, str) and image_url.startswith("/"):
            image_url = f"{base}{image_url}"
        results.append(
            VisualResult(
                chunk_id=str(r.get("chunk_id", "")),
                text=str(r.get("text", "")),
                score=float(r.get("score", 0.0) or 0.0),
                source_file=str(r.get("source_file", "")),
                page_number=int(r.get("page_number", 0) or 0),
                chunk_type=str(r.get("chunk_type", "text")),
                image_path=r.get("image_path") or None,
                image_url=image_url,
                caption=r.get("caption") or None,
                extracted_entities=list(r.get("extracted_entities", []) or []),
            )
        )

    return VisualSearchResponse(
        query=str(data.get("query", query_text)),
        query_entities=_as_entity_list(data.get("query_entities")),
        results=results,
        count=int(data.get("count", 0) or 0),
        tables_found=int(data.get("tables_found", 0) or 0),
        figures_found=int(data.get("figures_found", 0) or 0),
    )


def analyze_image(
    client: httpx.Client,
    image_base64: str,
    prompt: str,
    model: Optional[str] = None,
) -> str:
    payload = {"image_base64": image_base64, "prompt": prompt, "model": model}
    resp = client.post("/analyze-image", json=payload)
    _raise_for_status(resp)
    data = resp.json()
    return str(data.get("analysis", ""))


def ingest(
    client: httpx.Client,
    collection_name: str = "bio_guidelines",
    qdrant_path: str = "data/qdrant_db",
    model_path: Optional[str] = None,
    pdf_dir: str = "data/raw_pdfs",
    figures_dir: str = "data/figures",
    multimodal: bool = True,
    max_tokens: int = 500,
    overlap: int = 50,
    threshold: float = 0.0,
    batch_size: int = 64,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "collection_name": collection_name,
        "qdrant_path": qdrant_path,
        "model_path": model_path,
        "pdf_dir": pdf_dir,
        "figures_dir": figures_dir,
        "multimodal": multimodal,
        "max_tokens": max_tokens,
        "overlap": overlap,
        "threshold": threshold,
        "batch_size": batch_size,
    }
    resp = client.post("/ingest", json=payload)
    _raise_for_status(resp)
    data = resp.json()
    if not isinstance(data, dict):
        raise BioScholarAPIError("Unexpected /ingest response format.")
    return data

