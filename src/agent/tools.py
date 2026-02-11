"""
Tool definitions for the medical research agent.

Each tool wraps an existing BioScholar capability (retrieval, summarisation,
etc.) and exposes it to the LangGraph ReAct agent via LangChain's ``@tool``
decorator.

Tools are created via factory functions so that dependencies (retriever,
vector store) can be injected at graph-build time.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List

from langchain_core.tools import tool

if TYPE_CHECKING:
    from src.retrieve import HybridRetriever

logger = logging.getLogger(__name__)


def _format_search_results(results: List[Dict]) -> str:
    """Format retriever results as numbered text for the LLM.

    Each result is labelled ``[Source N]`` with file, page, and relevance
    so the agent can cite them accurately in its final answer.
    """
    if not results:
        return "No relevant guidelines found for this query."

    parts: List[str] = []
    for i, r in enumerate(results, 1):
        source = r.get("source_file", "unknown")
        page = r.get("page_number", "?")
        text = (r.get("text", "") or "")[:600]
        score = r.get("score", 0)
        entities = r.get("extracted_entities", [])
        entity_str = ", ".join(entities[:5]) if entities else "none"

        parts.append(
            f"[Source {i}] (File: {source}, Page: {page}, "
            f"Relevance: {score:.3f}, Entities: {entity_str})\n{text}"
        )

    return "\n\n".join(parts)


def create_search_tool(retriever: HybridRetriever):
    """Create a ``search_guidelines`` tool bound to *retriever*.

    The returned tool is a LangChain ``@tool``-decorated callable that
    the agent can invoke by name.  It runs the hybrid retriever (NER +
    semantic search) and returns formatted text the LLM can reason over.
    """

    @tool
    def search_guidelines(query: str) -> str:
        """Search medical guidelines database for information about diseases,
        drugs, symptoms, treatments, or procedures.

        Use this tool to find evidence-based medical information from clinical
        practice guidelines and PubMed papers.

        For **comparison questions**, call this tool separately for each item
        being compared (e.g. once for "aspirin" and once for "ibuprofen").

        Args:
            query: A focused medical search query, e.g. "metformin side effects",
                   "hypertension treatment guidelines", "aspirin contraindications".
        """
        logger.info("Agent tool call: search_guidelines(%r)", query)
        result = retriever.search(query=query, top_k=5, entity_filter=True)
        return _format_search_results(result["results"])

    return search_guidelines
