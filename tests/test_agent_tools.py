"""Tests for Task 5.2 LangGraph agent tools."""

from typing import Any, Dict

from src.agent.graph import is_complex_query
from src.agent.tools import build_tools


class _DummyRetriever:
    def __init__(self, response: Dict[str, Any]):
        self.response = response

    def search(self, query: str, top_k: int = 5, entity_filter: bool = True):
        return self.response


class _DummyResponse:
    def __init__(self, payload: Dict[str, Any]):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _DummyClient:
    def __init__(self, payload: Dict[str, Any]):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get(self, url: str, params: Dict[str, str]):
        return _DummyResponse(self.payload)


def test_search_guidelines_tool_returns_citations():
    retriever = _DummyRetriever(
        {
            "results": [
                {
                    "chunk_id": "c1",
                    "text": "Aspirin can increase bleeding risk.",
                    "score": 0.91,
                    "source_file": "guideline.pdf",
                    "page_number": 4,
                    "extracted_entities": ["Chemical:aspirin"],
                }
            ]
        }
    )
    tools = build_tools(retriever)

    out = tools["search_guidelines"]("aspirin risks")

    assert out["result_count"] == 1
    assert out["citations"][0]["source_file"] == "guideline.pdf"


def test_summarize_section_filters_by_doc_id():
    retriever = _DummyRetriever(
        {
            "results": [
                {
                    "chunk_id": "a1",
                    "text": "Dosage: Start with 5mg daily.",
                    "score": 0.81,
                    "source_file": "docA.pdf",
                    "page_number": 2,
                    "extracted_entities": [],
                },
                {
                    "chunk_id": "b1",
                    "text": "Dosage differs in doc B.",
                    "score": 0.8,
                    "source_file": "docB.pdf",
                    "page_number": 3,
                    "extracted_entities": [],
                },
            ]
        }
    )
    tools = build_tools(retriever)

    out = tools["summarize_section"]("docA.pdf", "dosage")

    assert out["result_count"] == 1
    assert "5mg" in out["summary"]
    assert out["citations"][0]["chunk_id"] == "a1"


def test_lookup_drug_interaction_parses_rxnav(monkeypatch):
    payload = {
        "interactionTypeGroup": [
            {
                "interactionType": [
                    {
                        "interactionPair": [
                            {"description": "May increase serum concentration of one drug."}
                        ]
                    }
                ]
            }
        ]
    }

    import src.agent.tools as tool_module

    monkeypatch.setattr(tool_module.httpx, "Client", lambda timeout: _DummyClient(payload))

    retriever = _DummyRetriever({"results": []})
    tools = build_tools(retriever)
    out = tools["lookup_drug_interaction"]("warfarin", "amiodarone")

    assert out["found"] is True
    assert "serum concentration" in out["summary"]
    assert out["citations"][0]["source_file"] == "rxnav_nlm_api"


def test_is_complex_query_triggers_on_comparison_or_entities():
    assert is_complex_query("Compare drug A and drug B", []) is True
    entities = [object(), object()]
    assert is_complex_query("What is treatment?", entities) is True
