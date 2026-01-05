"""Basic tests for VectorStore."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from pangu_agent.library.embeddings.vector_store import VectorStore


@pytest.fixture
def vector_store(tmp_path: Path):
    """Create a temporary vector store."""
    return VectorStore(storage_path=tmp_path / ".vector_store")


@pytest.fixture
def sample_embeddings():
    """Create sample normalized embeddings."""
    np.random.seed(42)
    emb1 = np.random.randn(512).astype(np.float32)
    emb2 = np.random.randn(512).astype(np.float32)
    emb3 = np.random.randn(512).astype(np.float32)

    # Normalize
    emb1 /= np.linalg.norm(emb1)
    emb2 /= np.linalg.norm(emb2)
    emb3 /= np.linalg.norm(emb3)

    return emb1, emb2, emb3


def test_vector_store_initialization(vector_store):
    """Test vector store initializes correctly."""
    assert vector_store is not None
    assert vector_store.count() == 0


def test_add_and_count(vector_store, sample_embeddings):
    """Test adding embeddings."""
    emb1, emb2, emb3 = sample_embeddings

    vector_store.add("file1", emb1, {"path": "doc1.pdf", "hash": "abc123"})
    assert vector_store.count() == 1

    vector_store.add("file2", emb2, {"path": "doc2.pdf", "hash": "def456"})
    assert vector_store.count() == 2


def test_search(vector_store, sample_embeddings):
    """Test searching for similar embeddings."""
    emb1, emb2, emb3 = sample_embeddings

    # Add embeddings
    vector_store.add("file1", emb1, {"path": "doc1.pdf"})
    vector_store.add("file2", emb2, {"path": "doc2.pdf"})
    vector_store.add("file3", emb3, {"path": "doc3.pdf"})

    # Search with emb1 (should find itself first)
    results = vector_store.search(emb1, top_k=2)

    assert len(results) == 2
    assert results[0].file_id == "file1"
    assert results[0].score > 0.99  # Should be almost 1.0 (itself)
    assert "path" in results[0].metadata


def test_delete(vector_store, sample_embeddings):
    """Test deleting embeddings."""
    emb1, emb2, _ = sample_embeddings

    vector_store.add("file1", emb1, {"path": "doc1.pdf"})
    vector_store.add("file2", emb2, {"path": "doc2.pdf"})
    assert vector_store.count() == 2

    vector_store.delete("file1")
    assert vector_store.count() == 1

    # Search should not find deleted file
    results = vector_store.search(emb1, top_k=5)
    assert all(r.file_id != "file1" for r in results)


def test_get_metadata(vector_store, sample_embeddings):
    """Test retrieving metadata."""
    emb1, _, _ = sample_embeddings

    metadata = {"path": "doc1.pdf", "hash": "abc123", "author": "Smith"}
    vector_store.add("file1", emb1, metadata)

    retrieved = vector_store.get("file1")
    assert retrieved is not None
    assert retrieved["path"] == "doc1.pdf"
    assert retrieved["author"] == "Smith"


def test_update_metadata(vector_store, sample_embeddings):
    """Test updating metadata without changing embedding."""
    emb1, _, _ = sample_embeddings

    vector_store.add("file1", emb1, {"path": "old_path.pdf", "hash": "abc"})

    # Update metadata
    vector_store.update_metadata("file1", {"path": "new_path.pdf"})

    # Check updated metadata
    metadata = vector_store.get("file1")
    assert metadata["path"] == "new_path.pdf"
    assert metadata["hash"] == "abc"  # Should keep old fields

    # Embedding should still be the same
    results = vector_store.search(emb1, top_k=1)
    assert results[0].file_id == "file1"
    assert results[0].score > 0.99


def test_search_empty_store(vector_store):
    """Test searching in empty store returns empty results."""
    query = np.random.randn(512).astype(np.float32)
    query /= np.linalg.norm(query)

    results = vector_store.search(query, top_k=5)
    assert len(results) == 0


def test_upsert_behavior(vector_store, sample_embeddings):
    """Test that adding same file_id updates the embedding."""
    emb1, emb2, _ = sample_embeddings

    # Add file1 with emb1
    vector_store.add("file1", emb1, {"version": 1})
    assert vector_store.count() == 1

    # Update file1 with emb2
    vector_store.add("file1", emb2, {"version": 2})
    assert vector_store.count() == 1  # Should still be 1

    # Search with emb2 should find file1
    results = vector_store.search(emb2, top_k=1)
    assert results[0].file_id == "file1"
    assert results[0].metadata["version"] == 2
