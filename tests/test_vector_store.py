#!/usr/bin/env python3
"""
Unit tests for the Qdrant vector store module.

All tests use in-memory Qdrant and a mock embedding model so they
run fast and require no external services.
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock

from src.ingest import Chunk
from src.vector_store import QdrantStore, DEFAULT_EMBEDDING_MODEL


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FAKE_VECTOR_SIZE = 384  # matches all-MiniLM-L6-v2


def _make_chunk(
    text: str = "Sample chunk text",
    chunk_id: str = "abc123",
    source_file: str = "test.pdf",
    page_number: int = 1,
    start_char: int = 0,
    end_char: int = 17,
    entities: list = None,
    extracted_entities: list = None,
) -> Chunk:
    """Create a Chunk with sensible defaults for testing."""
    return Chunk(
        text=text,
        chunk_id=chunk_id,
        source_file=source_file,
        page_number=page_number,
        start_char=start_char,
        end_char=end_char,
        entities=entities or [],
        metadata={"extracted_entities": extracted_entities or []},
    )


def _make_chunks(n: int = 10) -> list[Chunk]:
    """Generate *n* distinct test chunks."""
    return [
        _make_chunk(
            text=f"Chunk number {i} about medical topic {i}",
            chunk_id=f"chunk_{i:03d}",
            page_number=(i // 3) + 1,
            start_char=i * 100,
            end_char=i * 100 + 40,
            extracted_entities=[f"Disease:condition_{i}"] if i % 2 == 0 else [],
        )
        for i in range(n)
    ]


def _fake_encode(texts, **kwargs):
    """Return deterministic fake embeddings of the correct shape."""
    return np.random.default_rng(42).random((len(texts), FAKE_VECTOR_SIZE)).astype(np.float32)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_store():
    """Create a QdrantStore backed by in-memory Qdrant with mocked embeddings.

    Patches SentenceTransformer so no model weights are downloaded.
    """
    with patch("src.vector_store.SentenceTransformer") as MockST:
        mock_embedder = MagicMock()
        mock_embedder.get_sentence_embedding_dimension.return_value = FAKE_VECTOR_SIZE
        mock_embedder.encode.side_effect = _fake_encode
        MockST.return_value = mock_embedder

        store = QdrantStore(
            collection_name="test_collection",
            embedding_model="fake-model",
        )

    return store


# ---------------------------------------------------------------------------
# Tests – initialisation & collection management
# ---------------------------------------------------------------------------

class TestQdrantStoreInit:
    """Tests for store creation and collection management."""

    def test_creates_collection_on_init(self, mock_store: QdrantStore):
        """The collection should exist immediately after construction."""
        info = mock_store.get_collection_info()
        assert info["name"] == "test_collection"
        assert info["vector_size"] == FAKE_VECTOR_SIZE

    def test_count_starts_at_zero(self, mock_store: QdrantStore):
        """A fresh collection should have zero points."""
        assert mock_store.count() == 0

    def test_force_recreate_resets_collection(self):
        """force_recreate=True should drop existing data."""
        with patch("src.vector_store.SentenceTransformer") as MockST:
            mock_embedder = MagicMock()
            mock_embedder.get_sentence_embedding_dimension.return_value = FAKE_VECTOR_SIZE
            mock_embedder.encode.side_effect = _fake_encode
            MockST.return_value = mock_embedder

            # First store – insert data
            store1 = QdrantStore(collection_name="reset_test")
            chunks = _make_chunks(5)
            store1.add_chunks(chunks)
            assert store1.count() == 5

            # Second store – force recreate on same client memory
            # We need to reuse the same qdrant client to test this
            store2 = QdrantStore.__new__(QdrantStore)
            store2.collection_name = "reset_test"
            store2.client = store1.client
            store2.embedder = mock_embedder
            store2.vector_size = FAKE_VECTOR_SIZE
            store2._ensure_collection(force_recreate=True)
            assert store2.count() == 0

    def test_delete_collection(self, mock_store: QdrantStore):
        """delete_collection should remove it from Qdrant."""
        mock_store.delete_collection()
        collections = [
            c.name for c in mock_store.client.get_collections().collections
        ]
        assert "test_collection" not in collections


# ---------------------------------------------------------------------------
# Tests – add_chunks
# ---------------------------------------------------------------------------

class TestAddChunks:
    """Tests for upserting chunks into the store."""

    def test_add_single_chunk(self, mock_store: QdrantStore):
        """Inserting one chunk should increase count to 1."""
        chunks = [_make_chunk()]
        inserted = mock_store.add_chunks(chunks)

        assert inserted == 1
        assert mock_store.count() == 1

    def test_add_ten_chunks(self, mock_store: QdrantStore):
        """The roadmap asks us to verify 10 chunks can be inserted."""
        chunks = _make_chunks(10)
        inserted = mock_store.add_chunks(chunks)

        assert inserted == 10
        assert mock_store.count() == 10

    def test_add_empty_list(self, mock_store: QdrantStore):
        """Passing an empty list should be a no-op."""
        inserted = mock_store.add_chunks([])
        assert inserted == 0
        assert mock_store.count() == 0

    def test_add_chunks_in_batches(self, mock_store: QdrantStore):
        """Chunks exceeding batch_size should still all be inserted."""
        chunks = _make_chunks(20)
        inserted = mock_store.add_chunks(chunks, batch_size=7)

        assert inserted == 20
        assert mock_store.count() == 20

    def test_payload_stored_correctly(self, mock_store: QdrantStore):
        """Verify stored payloads contain the required metadata fields."""
        chunk = _make_chunk(
            text="Patient with diabetes.",
            chunk_id="meta_test",
            source_file="guideline.pdf",
            page_number=3,
            start_char=100,
            end_char=122,
            extracted_entities=["Disease:diabetes"],
        )
        mock_store.add_chunks([chunk])

        # Scroll to retrieve the stored point
        records, _ = mock_store.client.scroll(
            collection_name="test_collection",
            limit=1,
            with_payload=True,
        )

        assert len(records) == 1
        payload = records[0].payload

        assert payload["text"] == "Patient with diabetes."
        assert payload["chunk_id"] == "meta_test"
        assert payload["source_file"] == "guideline.pdf"
        assert payload["page_number"] == 3
        assert payload["start_char"] == 100
        assert payload["end_char"] == 122
        assert payload["extracted_entities"] == ["Disease:diabetes"]

    def test_embedder_called_with_texts(self, mock_store: QdrantStore):
        """The embedder should receive the chunk texts."""
        chunks = _make_chunks(3)
        mock_store.add_chunks(chunks)

        call_args = mock_store.embedder.encode.call_args
        texts_passed = call_args[0][0]  # first positional arg
        assert len(texts_passed) == 3
        assert all(isinstance(t, str) for t in texts_passed)


# ---------------------------------------------------------------------------
# Tests – get_collection_info
# ---------------------------------------------------------------------------

class TestCollectionInfo:
    """Tests for the info helper."""

    def test_info_keys(self, mock_store: QdrantStore):
        """Info dict should contain the expected keys."""
        info = mock_store.get_collection_info()
        assert "name" in info
        assert "points_count" in info
        assert "vector_size" in info
        assert "status" in info

    def test_info_reflects_inserts(self, mock_store: QdrantStore):
        """points_count should update after adding chunks."""
        mock_store.add_chunks(_make_chunks(5))
        info = mock_store.get_collection_info()
        assert info["points_count"] == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
