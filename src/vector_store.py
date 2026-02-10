#!/usr/bin/env python3
"""
Qdrant Vector Store

Manages a Qdrant collection for storing and retrieving document chunks
with their embeddings and metadata.

Supports three connection modes:
    - In-memory  (default, for testing / dev)
    - Local path (persistent file-based storage)
    - Remote URL (Docker or Qdrant Cloud)

Usage:
    from src.vector_store import QdrantStore
    from src.ingest import process_pdf
    from pathlib import Path

    store = QdrantStore(collection_name="bio_guidelines")
    chunks = process_pdf(Path("data/raw_pdfs/guideline.pdf"))
    store.add_chunks(chunks)
"""

from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

from src.ingest import Chunk


# Default embedding model as specified in the roadmap
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class QdrantStore:
    """Vector store backed by Qdrant for document chunk storage and retrieval.

    The store handles:
        - Embedding generation via a SentenceTransformer model
        - Collection creation with the correct vector dimensions
        - Upserting chunks with full metadata (source, page, entities)

    Args:
        collection_name: Name of the Qdrant collection.
        embedding_model: HuggingFace model id or local path for
            sentence-transformers.  Defaults to ``all-MiniLM-L6-v2``.
        qdrant_url: URL for a remote Qdrant instance
            (e.g. ``http://localhost:6333``).  Mutually exclusive with
            *qdrant_path*.
        qdrant_path: Local filesystem path for persistent on-disk storage.
            Mutually exclusive with *qdrant_url*.
        force_recreate: If *True*, drop and recreate the collection on init.

    Note:
        If neither *qdrant_url* nor *qdrant_path* is provided the client
        runs **in-memory** (great for unit tests but data is lost on exit).

    Example:
        >>> store = QdrantStore("test_collection")
        >>> store.add_chunks(chunks)
        10
        >>> store.count()
        10
    """

    def __init__(
        self,
        collection_name: str,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        qdrant_url: Optional[str] = None,
        qdrant_path: Optional[str] = None,
        force_recreate: bool = False,
    ) -> None:
        self.collection_name = collection_name

        # --- Qdrant client ---------------------------------------------------
        if qdrant_url is not None:
            self.client = QdrantClient(url=qdrant_url)
        elif qdrant_path is not None:
            self.client = QdrantClient(path=qdrant_path)
        else:
            # Pure in-memory mode
            self.client = QdrantClient(location=":memory:")

        # --- Embedding model --------------------------------------------------
        self.embedder = SentenceTransformer(embedding_model)
        self.vector_size: int = self.embedder.get_sentence_embedding_dimension()

        # --- Ensure collection exists -----------------------------------------
        self._ensure_collection(force_recreate=force_recreate)

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def _ensure_collection(self, force_recreate: bool = False) -> None:
        """Create the collection if it does not already exist.

        Args:
            force_recreate: Drop the existing collection first.
        """
        existing = [c.name for c in self.client.get_collections().collections]

        if force_recreate and self.collection_name in existing:
            self.client.delete_collection(self.collection_name)
            existing.remove(self.collection_name)

        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def delete_collection(self) -> None:
        """Delete the underlying Qdrant collection."""
        self.client.delete_collection(self.collection_name)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: List[Chunk], batch_size: int = 64) -> int:
        """Embed and upsert chunks into the collection.

        Each chunk is stored as a Qdrant point with:
            - **id**: derived from ``chunk.chunk_id`` (or a UUID fallback)
            - **vector**: dense embedding of ``chunk.text``
            - **payload**: all chunk metadata needed for citation tracking

        Args:
            chunks: List of ``Chunk`` objects from the ingestion pipeline.
            batch_size: Number of chunks to embed at once.

        Returns:
            Total number of points upserted.
        """
        if not chunks:
            return 0

        total_upserted = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.text for c in batch]

            # Embed
            embeddings = self.embedder.encode(texts, show_progress_bar=False)

            # Build Qdrant points
            points: List[PointStruct] = []
            for chunk, vector in zip(batch, embeddings):
                payload = {
                    "text": chunk.text,
                    "chunk_id": chunk.chunk_id,
                    "source_file": chunk.source_file,
                    "page_number": chunk.page_number,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "extracted_entities": chunk.metadata.get(
                        "extracted_entities", []
                    ),
                }
                points.append(
                    PointStruct(
                        id=uuid4().hex,
                        vector=vector.tolist(),
                        payload=payload,
                    )
                )

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            total_upserted += len(points)

        return total_upserted

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the number of points in the collection."""
        info = self.client.get_collection(self.collection_name)
        return info.points_count

    def get_collection_info(self) -> Dict:
        """Return basic collection metadata as a plain dict."""
        info = self.client.get_collection(self.collection_name)
        return {
            "name": self.collection_name,
            "points_count": info.points_count,
            "vector_size": self.vector_size,
            "status": str(info.status),
        }

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        with_payload: bool = True,
    ) -> List[Dict]:
        """Retrieve the top-k most similar chunks for a query string.

        Args:
            query: Natural-language query text.
            top_k: Number of matches to return.
            with_payload: Whether to return stored metadata.

        Returns:
            List of result dicts sorted by score (higher is more similar).
        """
        if not query:
            return []

        query_vector = self.embedder.encode(
            [query],
            show_progress_bar=False,
        )[0].tolist()

        if hasattr(self.client, "search"):
            raw_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                with_payload=with_payload,
            )
        else:
            # Newer qdrant-client versions (0.11+) expose query_points
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                with_payload=with_payload,
            )
            raw_results = response.points

        return self._format_results(raw_results, with_payload)

    def search_with_filters(
        self,
        query: str,
        entity_keys: Optional[List[str]] = None,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[Dict]:
        """Retrieve chunks using semantic search with optional entity filtering.

        When *entity_keys* is provided, results are restricted to chunks
        whose ``extracted_entities`` payload contains at least one of the
        given entity strings (e.g. ``["Disease:fever", "Chemical:aspirin"]``).

        Args:
            query: Natural-language query text.
            entity_keys: Entity strings to filter on (format ``"Label:text"``).
                         If empty or *None*, no filtering is applied.
            top_k: Number of results to return.
            score_threshold: Minimum cosine similarity score.

        Returns:
            Filtered list of result dicts sorted by score.
        """
        if not query:
            return []

        query_vector = self.embedder.encode(
            [query], show_progress_bar=False
        )[0].tolist()

        # Build Qdrant filter
        query_filter = None
        if entity_keys:
            query_filter = Filter(
                should=[
                    FieldCondition(
                        key="extracted_entities",
                        match=MatchAny(any=entity_keys),
                    )
                ]
            )

        if hasattr(self.client, "search"):
            raw_results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=top_k,
                score_threshold=score_threshold if score_threshold > 0 else None,
                with_payload=True,
            )
        else:
            from qdrant_client.models import Query
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=top_k,
                score_threshold=score_threshold if score_threshold > 0 else None,
                with_payload=True,
            )
            raw_results = response.points

        return self._format_results(raw_results, with_payload=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_results(raw_results, with_payload: bool = True) -> List[Dict]:
        """Convert raw Qdrant hits into plain dicts."""
        formatted: List[Dict] = []
        for hit in raw_results:
            payload = hit.payload or {}
            formatted.append(
                {
                    "chunk_id": payload.get("chunk_id"),
                    "text": payload.get("text"),
                    "score": hit.score,
                    "source_file": payload.get("source_file"),
                    "page_number": payload.get("page_number"),
                    "extracted_entities": payload.get("extracted_entities", []),
                    "payload": payload if with_payload else {},
                }
            )
        return formatted
