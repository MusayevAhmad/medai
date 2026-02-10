"""
Hybrid Retrieval Module

Combines NER entity extraction with vector similarity search for
entity-aware document retrieval.

Pipeline:
    1. Extract entities from the user query via MedicalNER
    2. Build entity filter keys (e.g. "Disease:fever")
    3. Run filtered semantic search in Qdrant
    4. Fall back to unfiltered search if no filtered results

Usage:
    from src.retrieve import HybridRetriever

    retriever = HybridRetriever(ner=ner, store=store)
    results = retriever.search("What treats fever in children?")
"""

import logging
from typing import Dict, List, Optional

from src.inference import Entity, MedicalNER
from src.vector_store import QdrantStore

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Entity-aware retrieval combining NER and vector search.

    The retriever extracts medical entities from the query, converts them
    into filter keys, and uses ``QdrantStore.search_with_filters()`` to
    prioritise chunks that mention the same entities.

    If the filtered search returns no results, it falls back to a pure
    semantic search so the user always gets something useful.

    Args:
        ner: Initialised MedicalNER instance for query entity extraction.
        store: Initialised QdrantStore with ingested documents.
        ner_threshold: Minimum NER confidence to include an entity.
    """

    def __init__(
        self,
        ner: MedicalNER,
        store: QdrantStore,
        ner_threshold: float = 0.3,
    ) -> None:
        self.ner = ner
        self.store = store
        self.ner_threshold = ner_threshold

    def search(
        self,
        query: str,
        top_k: int = 5,
        entity_filter: bool = True,
        score_threshold: float = 0.0,
    ) -> Dict:
        """Run hybrid entity-filtered retrieval.

        Args:
            query: Natural-language question or search query.
            top_k: Number of results to return.
            entity_filter: If True, use NER entities to filter results.
            score_threshold: Minimum similarity score to include.

        Returns:
            Dict with keys:
                - ``query_entities``: List of Entity objects from the query
                - ``entity_keys``: Filter strings used (e.g. "Disease:fever")
                - ``results``: List of search result dicts
                - ``filtered``: Whether entity filtering was applied
        """
        # Step 1: Extract entities from query
        query_entities = self.ner.predict_entities(
            query, threshold=self.ner_threshold
        )

        # Step 2: Build entity filter keys
        entity_keys = [f"{e.label}:{e.text}" for e in query_entities]

        # Step 3: Search with filters (if entities found and filtering enabled)
        filtered = False
        results: List[Dict] = []

        if entity_filter and entity_keys:
            results = self.store.search_with_filters(
                query=query,
                entity_keys=entity_keys,
                top_k=top_k,
                score_threshold=score_threshold,
            )
            filtered = True
            logger.info(
                "Filtered search: %d results for entities %s",
                len(results),
                entity_keys,
            )

        # Step 4: Fall back to unfiltered search if no results
        if not results:
            if filtered:
                logger.info("No filtered results, falling back to semantic search")
            results = self.store.search_with_filters(
                query=query,
                entity_keys=None,
                top_k=top_k,
                score_threshold=score_threshold,
            )
            filtered = False

        return {
            "query_entities": query_entities,
            "entity_keys": entity_keys,
            "results": results,
            "filtered": filtered,
        }
