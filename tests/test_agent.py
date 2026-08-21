"""Unit tests for LangGraph Agent Graph, Nodes, and Routing."""

from typing import Any, Dict, List
import pytest

from src.agent.graph import build_agent_graph, is_complex_query, run_agent
from src.agent.nodes import create_agent_node, create_tool_executor
from src.agent.tools import _format_search_results, build_tools


class MockLLM:
    """Mock LLM client for deterministic testing."""

    def __init__(self, response_text: str = "This is a grounded clinical response."):
        self.response_text = response_text
        self.model = "mock-llm"

    def is_available(self) -> bool:
        return True

    def generate_answer(self, question: str, context_chunks: List[Dict], system_prompt: str = None) -> str:
        return self.response_text

    def chat_completion(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
        return self.response_text


class MockRetriever:
    """Mock HybridRetriever returning predictable search hits."""

    def __init__(self):
        self.results = [
            {
                "chunk_id": "c1",
                "text": "Metformin is first line therapy for type 2 diabetes.",
                "source_file": "EPMC_Diabetes_Type2_Review.pdf",
                "page_number": 3,
                "score": 0.92,
                "extracted_entities": ["Chemical:metformin", "Disease:type 2 diabetes"],
            }
        ]

    def search(self, query: str, top_k: int = 5, entity_filter: bool = True):
        return {"results": self.results, "count": len(self.results)}


def test_is_complex_query():
    """Test complexity detection with and without entities."""
    # Simple query, no entities
    assert is_complex_query("What are the symptoms of hypertension?") is False

    # Comparison query triggering regex
    assert is_complex_query("Compare metformin vs insulin for glycemic control") is True
    assert is_complex_query("What are the side effects of lisinopril?") is True
    assert is_complex_query("What is the drug interaction between aspirin and warfarin?") is True

    # Query with multiple extracted entities
    assert is_complex_query("Tell me about this condition", entities=["Disease:diabetes", "Chemical:metformin"]) is True
    assert is_complex_query("Tell me about this condition", entities=["Disease:diabetes"]) is False


def test_format_search_results():
    """Test formatting retrieval results for prompt injection."""
    empty_res = _format_search_results([])
    assert "No relevant clinical guideline sections found." in empty_res

    results = [
        {
            "source_file": "test_guide.pdf",
            "page_number": 5,
            "score": 0.88,
            "text": "Blood pressure should be kept under 130/80.",
        }
    ]
    formatted = _format_search_results(results)
    assert "[Source 1]" in formatted
    assert "test_guide.pdf" in formatted
    assert "page 5" in formatted
    assert "score: 0.880" in formatted
    assert "Blood pressure should be kept under 130/80." in formatted


def test_build_and_run_agent_graph():
    """Test end-to-end agent graph construction and execution."""
    llm = MockLLM("Metformin effectively reduces HbA1c based on clinical guidelines.")
    retriever = MockRetriever()

    graph = build_agent_graph(llm, retriever)
    assert graph is not None

    result = run_agent(graph, "What is the recommended treatment for type 2 diabetes?")

    assert "answer" in result
    assert "citations" in result
    assert "steps" in result
    assert len(result["answer"]) > 0
    assert result["steps"] >= 1
