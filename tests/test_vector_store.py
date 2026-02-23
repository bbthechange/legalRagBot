"""Tests for src/vector_store.py"""

import os
from unittest.mock import patch

import numpy as np
import pytest

from src.vector_store import FaissVectorStore, create_vector_store


def _make_vectors(n, dim=128):
    """Create n random normalized vectors."""
    rng = np.random.RandomState(42)
    vecs = rng.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


class TestFaissVectorStore:
    def test_upsert_count_and_total(self):
        store = FaissVectorStore()
        vecs = _make_vectors(5)
        ids = [f"id-{i}" for i in range(5)]
        meta = [{"type": "NDA"} for _ in range(5)]

        count = store.upsert(ids, vecs, meta)
        assert count == 5
        assert store.total_vectors == 5

    def test_search_returns_sorted_results_with_keys(self):
        store = FaissVectorStore()
        vecs = _make_vectors(5)
        ids = [f"id-{i}" for i in range(5)]
        meta = [{"type": "NDA"} for _ in range(5)]
        store.upsert(ids, vecs, meta)

        query = vecs[0:1].copy()
        results = store.search(query, top_k=3)
        assert len(results) == 3
        for r in results:
            assert "id" in r
            assert "score" in r
            assert "metadata" in r

        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_top_k_1(self):
        store = FaissVectorStore()
        vecs = _make_vectors(5)
        ids = [f"id-{i}" for i in range(5)]
        meta = [{"type": "NDA"} for _ in range(5)]
        store.upsert(ids, vecs, meta)

        query = vecs[0:1].copy()
        results = store.search(query, top_k=1)
        assert len(results) == 1

    def test_empty_store_search(self):
        store = FaissVectorStore()
        results = store.search(np.zeros((1, 128), dtype=np.float32))
        assert results == []

    def test_filter_by_type(self):
        store = FaissVectorStore()
        vecs = _make_vectors(5)
        ids = [f"id-{i}" for i in range(5)]
        meta = [
            {"type": "NDA"},
            {"type": "Employment"},
            {"type": "NDA"},
            {"type": "Service Agreement"},
            {"type": "NDA"},
        ]
        store.upsert(ids, vecs, meta)

        query = vecs[0:1].copy()
        results = store.search(query, top_k=5, filters={"type": "NDA"})
        assert len(results) > 0
        for r in results:
            assert r["metadata"]["type"] == "NDA"

    def test_filter_matching_nothing(self):
        store = FaissVectorStore()
        vecs = _make_vectors(5)
        ids = [f"id-{i}" for i in range(5)]
        meta = [{"type": "NDA"} for _ in range(5)]
        store.upsert(ids, vecs, meta)

        query = vecs[0:1].copy()
        results = store.search(query, top_k=5, filters={"type": "NonExistent"})
        assert results == []

    def test_overfetch_with_filter(self):
        store = FaissVectorStore()
        vecs = _make_vectors(10)
        ids = [f"id-{i}" for i in range(10)]
        meta = [{"type": "NDA" if i % 2 == 0 else "Employment"} for i in range(10)]
        store.upsert(ids, vecs, meta)

        query = vecs[0:1].copy()
        results = store.search(query, top_k=3, filters={"type": "NDA"})
        assert len(results) <= 3
        for r in results:
            assert r["metadata"]["type"] == "NDA"

    def test_delete_removes_from_search(self):
        store = FaissVectorStore()
        vecs = _make_vectors(5)
        ids = [f"id-{i}" for i in range(5)]
        meta = [{"type": "NDA"} for _ in range(5)]
        store.upsert(ids, vecs, meta)

        deleted = store.delete(["id-0"])
        assert deleted == 1

        query = vecs[0:1].copy()
        results = store.search(query, top_k=5)
        result_ids = [r["id"] for r in results]
        assert "id-0" not in result_ids
        # total_vectors unchanged (FAISS doesn't remove from index)
        assert store.total_vectors == 5

    def test_delete_nonexistent(self):
        store = FaissVectorStore()
        vecs = _make_vectors(3)
        ids = ["a", "b", "c"]
        meta = [{} for _ in range(3)]
        store.upsert(ids, vecs, meta)

        deleted = store.delete(["nonexistent"])
        assert deleted == 0


class TestVectorStoreFactory:
    def test_faiss_factory(self):
        store = create_vector_store("faiss")
        assert isinstance(store, FaissVectorStore)

    def test_pinecone_no_api_key(self):
        with patch.dict(os.environ, {"PINECONE_API_KEY": ""}):
            with pytest.raises(ValueError, match="PINECONE_API_KEY"):
                create_vector_store("pinecone")

    def test_unknown_provider(self):
        with pytest.raises(ValueError, match="Unknown vector store provider"):
            create_vector_store("unknown")
