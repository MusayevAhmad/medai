"""Integration tests for FastAPI endpoints."""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from app.main import app
from src.inference import Entity


@pytest.fixture
def client():
    """Create test client with lifespan context."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    """Test GET /health returns expected system status structure."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] in ("ok", "degraded")
    assert "ner_loaded" in data
    assert "llm_available" in data
    assert "collection_count" in data


def test_entities_endpoint_mocked(client):
    """Test POST /entities with mocked NER model."""
    mock_entities = [
        Entity(label="Disease", text="diabetes", confidence=0.95, start=14, end=22),
        Entity(label="Chemical", text="metformin", confidence=0.98, start=37, end=46),
    ]

    with patch("app.main.get_ner") as mock_get_ner:
        mock_ner = MagicMock()
        mock_ner.predict_entities.return_value = mock_entities
        mock_get_ner.return_value = mock_ner

        payload = {"text": "Diagnosed with diabetes and took metformin."}
        response = client.post("/entities", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert data["entities"][0]["text"] == "diabetes"
        assert data["entities"][0]["label"] == "Disease"
        assert data["entities"][1]["text"] == "metformin"
        assert data["entities"][1]["label"] == "Chemical"


def test_search_endpoint_mocked(client):
    """Test POST /search endpoint with mocked retriever."""
    mock_search_res = {
        "results": [
            {
                "chunk_id": "chk_1",
                "text": "ACE inhibitors reduce blood pressure in hypertensive patients.",
                "source_file": "EPMC_Hypertension_Management.pdf",
                "page_number": 2,
                "score": 0.89,
                "extracted_entities": ["Disease:hypertension", "Chemical:ACE inhibitors"],
            }
        ],
        "count": 1,
    }

    with patch("app.main.get_retriever") as mock_get_retriever:
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = mock_search_res
        mock_get_retriever.return_value = mock_retriever

        payload = {"query": "hypertension treatment", "top_k": 3}
        response = client.post("/search", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert data["results"][0]["chunk_id"] == "chk_1"
        assert "ACE inhibitors" in data["results"][0]["text"]


def test_query_endpoint_mocked(client):
    """Test POST /query endpoint full RAG flow with mocked components."""
    mock_entities = [Entity(label="Disease", text="asthma", confidence=0.94, start=0, end=6)]
    mock_search_res = {
        "results": [
            {
                "chunk_id": "chk_asthma_1",
                "text": "Inhaled corticosteroids are cornerstone therapies for persistent asthma.",
                "source_file": "EPMC_Asthma_Pathophysiology.pdf",
                "page_number": 4,
                "score": 0.91,
                "extracted_entities": ["Disease:asthma"],
            }
        ],
        "count": 1,
    }

    with patch("app.main.get_ner") as mock_get_ner, \
         patch("app.main.get_retriever") as mock_get_retriever, \
         patch("app.main.get_llm") as mock_get_llm:

        mock_ner = MagicMock()
        mock_ner.predict_entities.return_value = mock_entities
        mock_get_ner.return_value = mock_ner

        mock_retriever = MagicMock()
        mock_retriever.search.return_value = mock_search_res
        mock_get_retriever.return_value = mock_retriever

        mock_llm = MagicMock()
        mock_llm.generate_answer.return_value = "Inhaled corticosteroids are recommended [Source 1]."
        mock_llm.model = "mock-llama"
        mock_get_llm.return_value = mock_llm

        payload = {
            "question": "What is the standard treatment for asthma?",
            "top_k": 3,
            "use_agent": False,
        }
        response = client.post("/query", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert "Inhaled corticosteroids" in data["answer"]
        assert len(data["citations"]) == 1
        assert data["citations"][0]["source_file"] == "EPMC_Asthma_Pathophysiology.pdf"
