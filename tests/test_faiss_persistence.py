"""Tests for FAISS persistence (save/load) in src/vector_store.py"""

import os

import numpy as np
import pytest

from src.vector_store import FaissVectorStore


@pytest.fixture
def populated_store():
    """Create a FAISS store with 3 vectors and metadata."""
    store = FaissVectorStore()
    ids = ["doc-001", "doc-002", "doc-003"]
    embeddings = np.random.RandomState(42).randn(3, 128).astype(np.float32)
    metadata = [
        {"title": "First", "source": "clauses_json"},
        {"title": "Second", "source": "cuad"},
        {"title": "Third", "source": "statutes"},
    ]
    store.upsert(ids, embeddings, metadata)
    return store, ids, embeddings, metadata


class TestFaissPersistence:
    def test_save_creates_both_files(self, populated_store, tmp_path):
        store, *_ = populated_store
        base = str(tmp_path / "test_index")
        store.save(base)
        assert os.path.exists(f"{base}.index")
        assert os.path.exists(f"{base}.meta.json")

    def test_load_nonexistent_returns_false(self):
        store = FaissVectorStore()
        assert store.load("/nonexistent/path/index") is False

    def test_save_load_roundtrip(self, populated_store, tmp_path):
        store, ids, embeddings, metadata = populated_store
        base = str(tmp_path / "roundtrip")
        store.save(base)

        new_store = FaissVectorStore()
        assert new_store.load(base) is True
        assert new_store.total_vectors == 3

        # Verify search returns correct results with scores
        query = embeddings[0:1].copy()
        import faiss
        faiss.normalize_L2(query)
        results = new_store.search(query, top_k=1)
        assert len(results) == 1
        assert results[0]["id"] == ids[0]
        assert results[0]["metadata"]["title"] == metadata[0]["title"]

    def test_deleted_ids_survive_roundtrip(self, populated_store, tmp_path):
        store, ids, *_ = populated_store
        store.delete([ids[1]])
        base = str(tmp_path / "deleted")
        store.save(base)

        new_store = FaissVectorStore()
        new_store.load(base)

        # Search should not return deleted doc
        query = np.random.RandomState(99).randn(1, 128).astype(np.float32)
        import faiss
        faiss.normalize_L2(query)
        results = new_store.search(query, top_k=10)
        result_ids = [r["id"] for r in results]
        assert ids[1] not in result_ids

    def test_save_creates_nested_directories(self, tmp_path):
        store = FaissVectorStore()
        embeddings = np.random.RandomState(7).randn(1, 64).astype(np.float32)
        store.upsert(["x"], embeddings, [{"k": "v"}])

        nested = str(tmp_path / "a" / "b" / "c" / "idx")
        store.save(nested)
        assert os.path.exists(f"{nested}.index")
        assert os.path.exists(f"{nested}.meta.json")
