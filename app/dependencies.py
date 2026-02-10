"""
Shared dependencies for the FastAPI application.

Manages singleton instances of the NER model, vector store, retriever,
and LLM client. Uses module-level state initialised at startup.
"""

import glob
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Module-level singletons (set by init_dependencies)
_ner: Optional["MedicalNER"] = None
_store: Optional["QdrantStore"] = None
_retriever: Optional["HybridRetriever"] = None
_llm: Optional["LLMClient"] = None


def _find_latest_model() -> str:
    """Find the most recent training run's final_model directory."""
    pattern = "outputs/models/run_*/final_model"
    candidates = sorted(glob.glob(pattern), reverse=True)
    if not candidates:
        raise FileNotFoundError(
            f"No trained model found matching '{pattern}'. "
            "Run `python src/train.py --config config.yaml` first."
        )
    return candidates[0]


def init_dependencies(
    model_path: Optional[str] = None,
    collection_name: str = "bio_guidelines",
    qdrant_path: str = "data/qdrant_db",
    llm_base_url: str = "http://localhost:11434/v1",
    llm_model: str = "llama3.2",
) -> None:
    """Initialise all shared dependencies.

    Called once at FastAPI startup. Heavy resources (model loading,
    Qdrant connection) are created here so requests stay fast.

    Args:
        model_path: Path to trained NER model. Auto-detects latest if None.
        collection_name: Qdrant collection name.
        qdrant_path: Local Qdrant storage path (persistent, no Docker).
        llm_base_url: Base URL for the OpenAI-compatible LLM API.
        llm_model: LLM model name.
    """
    global _ner, _store, _retriever, _llm

    # NER model
    if model_path is None:
        model_path = _find_latest_model()
    logger.info("Loading NER model from %s", model_path)

    from src.inference import MedicalNER
    _ner = MedicalNER(model_path=model_path, verbose=True)

    # Vector store
    logger.info("Connecting to Qdrant (path=%s, collection=%s)", qdrant_path, collection_name)
    from src.vector_store import QdrantStore
    _store = QdrantStore(
        collection_name=collection_name,
        qdrant_path=qdrant_path,
    )
    logger.info("Qdrant collection '%s' has %d chunks", collection_name, _store.count())

    # Hybrid retriever
    from src.retrieve import HybridRetriever
    _retriever = HybridRetriever(ner=_ner, store=_store)

    # LLM client
    from src.llm import LLMClient
    _llm = LLMClient(base_url=llm_base_url, model=llm_model)
    logger.info("LLM client configured: %s (model=%s)", llm_base_url, llm_model)


def get_ner() -> "MedicalNER":
    """Return the shared NER model instance."""
    if _ner is None:
        raise RuntimeError("Dependencies not initialised. Call init_dependencies() first.")
    return _ner


def get_store() -> "QdrantStore":
    """Return the shared vector store instance."""
    if _store is None:
        raise RuntimeError("Dependencies not initialised. Call init_dependencies() first.")
    return _store


def get_retriever() -> "HybridRetriever":
    """Return the shared hybrid retriever instance."""
    if _retriever is None:
        raise RuntimeError("Dependencies not initialised. Call init_dependencies() first.")
    return _retriever


def get_llm() -> "LLMClient":
    """Return the shared LLM client instance."""
    if _llm is None:
        raise RuntimeError("Dependencies not initialised. Call init_dependencies() first.")
    return _llm
