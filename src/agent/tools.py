"""Tools used by the LangGraph medical agent.

Task 5.2 adds three concrete tools:
- ``search_guidelines(query)``: wraps existing hybrid retrieval.
- ``lookup_drug_interaction(drug1, drug2)``: uses RxNav interaction API.
- ``summarize_section(doc_id, section)``: finds section-relevant snippets.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import httpx

logger = logging.getLogger(__name__)


def _citation_from_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a retrieval hit to the agent's citation shape."""
    text = result.get("text", "")
    return {
        "source_file": result.get("source_file", ""),
        "page_number": result.get("page_number", 0),
        "chunk_id": result.get("chunk_id", ""),
        "score": float(result.get("score", 0.0)),
        "text_preview": text[:280],
        "extracted_entities": result.get("extracted_entities", []),
    }


def _format_search_results(results: List[Dict[str, Any]]) -> str:
    """Format retrieval search hits into a numbered context text block for LLM reasoning."""
    if not results:
        return "No relevant clinical guideline sections found."
    lines = []
    for i, r in enumerate(results, 1):
        src = r.get("source_file", "unknown")
        page = r.get("page_number", 0)
        score = r.get("score", 0.0)
        text = (r.get("text") or "").strip()
        lines.append(f"[Source {i}] ({src}, page {page}, score: {score:.3f}):\n{text}")
    return "\n\n".join(lines)


def build_tools(retriever: Any, timeout_s: float = 12.0) -> Dict[str, Any]:
    """Build concrete tool callables bound to runtime dependencies.

    Args:
        retriever: The existing ``HybridRetriever`` instance.
        timeout_s: Timeout used for outbound drug-interaction API calls.

    Returns:
        Dict mapping tool name to callable.
    """

    def search_guidelines(query: str, top_k: int = 5) -> Dict[str, Any]:
        """Search indexed medical guidelines and return citations."""
        search = retriever.search(query=query, top_k=top_k, entity_filter=True)
        results = search.get("results", [])
        citations = [_citation_from_result(r) for r in results]
        snippets = [r.get("text", "")[:400] for r in results]
        return {
            "query": query,
            "result_count": len(results),
            "snippets": snippets,
            "citations": citations,
        }

    def lookup_drug_interaction(drug1: str, drug2: str) -> Dict[str, Any]:
        """Lookup pairwise drug interaction evidence via RxNav API."""
        cleaned_1 = drug1.strip()
        cleaned_2 = drug2.strip()
        if not cleaned_1 or not cleaned_2:
            return {
                "drug1": cleaned_1,
                "drug2": cleaned_2,
                "found": False,
                "message": "Both drug names are required.",
                "citations": [],
            }

        url = "https://rxnav.nlm.nih.gov/REST/interaction/interaction.json"
        params = {"rxcui": f"{cleaned_1}+{cleaned_2}"}

        try:
            with httpx.Client(timeout=timeout_s) as client:
                response = client.get(url, params=params)
                response.raise_for_status()
        except Exception as exc:
            logger.warning("Drug interaction lookup failed for %s/%s: %s", cleaned_1, cleaned_2, exc)
            return {
                "drug1": cleaned_1,
                "drug2": cleaned_2,
                "found": False,
                "message": "Interaction service unavailable.",
                "citations": [],
            }

        data = response.json()
        groups = data.get("interactionTypeGroup") or []
        if not groups:
            return {
                "drug1": cleaned_1,
                "drug2": cleaned_2,
                "found": False,
                "message": "No known interaction found in RxNav.",
                "citations": [],
            }

        interaction_texts: List[str] = []
        for group in groups:
            for item in group.get("interactionType", []):
                for pair in item.get("interactionPair", []):
                    desc = pair.get("description")
                    if desc:
                        interaction_texts.append(desc)

        deduped = list(dict.fromkeys(interaction_texts))
        summary = deduped[0] if deduped else "Interaction data available but no description provided."

        return {
            "drug1": cleaned_1,
            "drug2": cleaned_2,
            "found": True,
            "summary": summary,
            "details": deduped[:5],
            "citations": [
                {
                    "source_file": "rxnav_nlm_api",
                    "page_number": 0,
                    "chunk_id": "rxnav_interaction",
                    "score": 1.0,
                    "text_preview": summary[:280],
                    "extracted_entities": [f"Chemical:{cleaned_1}", f"Chemical:{cleaned_2}"],
                }
            ],
        }

    def summarize_section(doc_id: str, section: str, top_k: int = 8) -> Dict[str, Any]:
        """Summarize a section from a specific source document."""
        query = f"{doc_id} {section}"
        search = retriever.search(query=query, top_k=top_k, entity_filter=False)
        results = [r for r in search.get("results", []) if r.get("source_file") == doc_id]

        section_l = section.lower().strip()
        if section_l:
            section_matches = [r for r in results if section_l in (r.get("text") or "").lower()]
            if section_matches:
                results = section_matches

        picks = results[:3]
        bullets = [f"- {r.get('text', '').strip()[:220]}" for r in picks if r.get("text")]

        summary = "\n".join(bullets) if bullets else "No section summary available from indexed chunks."
        citations = [_citation_from_result(r) for r in picks]

        return {
            "doc_id": doc_id,
            "section": section,
            "summary": summary,
            "result_count": len(results),
            "citations": citations,
        }

    return {
        "search_guidelines": search_guidelines,
        "lookup_drug_interaction": lookup_drug_interaction,
        "summarize_section": summarize_section,
    }
