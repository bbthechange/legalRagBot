"""Tests for multi-source document support (end-to-end)."""

from unittest.mock import patch, MagicMock

import pytest

from src.embeddings import _load_clauses_json, load_documents, load_clause_database
from src.retrieval import search_similar_clauses, format_retrieval_results


class TestLoadClausesJson:
    def test_converts_all_15_docs(self):
        docs = _load_clauses_json("data/clauses.json")
        assert len(docs) == 15

        for doc in docs:
            assert "doc_id" in doc
            assert "source" in doc
            assert doc["source"] == "clauses_json"
            assert "doc_type" in doc
            assert doc["doc_type"] == "clause"
            assert "title" in doc
            assert "text" in doc
            assert "metadata" in doc
            assert isinstance(doc["metadata"], dict)


class TestLoadDocumentsWithUnifiedSchema:
    def test_load_from_documents(self, mock_provider, sample_unified_documents):
        mock_store = MagicMock()
        mock_store.upsert.return_value = 3
        mock_store.total_vectors = 0

        with patch("src.embeddings.create_provider", return_value=mock_provider), \
             patch("src.embeddings.create_vector_store", return_value=mock_store):
            db = load_documents(documents=sample_unified_documents)
            assert len(db["documents"]) == 3
            assert db["documents"] is db["clauses"]

    def test_return_dict_has_both_keys(self, mock_provider, sample_unified_documents):
        mock_store = MagicMock()
        mock_store.upsert.return_value = 3
        mock_store.total_vectors = 0

        with patch("src.embeddings.create_provider", return_value=mock_provider), \
             patch("src.embeddings.create_vector_store", return_value=mock_store):
            db = load_documents(documents=sample_unified_documents)
            assert "documents" in db
            assert "clauses" in db
            assert db["documents"] is db["clauses"]


class TestBackwardCompat:
    def test_load_clause_database_returns_expected_keys(self, mock_provider):
        mock_store = MagicMock()
        mock_store.upsert.return_value = 15
        mock_store.total_vectors = 0

        with patch("src.embeddings.create_provider", return_value=mock_provider), \
             patch("src.embeddings.create_vector_store", return_value=mock_store):
            db = load_clause_database()
            assert "store" in db
            assert "clauses" in db
            assert "provider" in db


class TestRetrievalFromMetadata:
    def test_search_returns_metadata_fields(self, loaded_multi_source_db):
        results = search_similar_clauses(
            "confidentiality agreement", loaded_multi_source_db, top_k=3
        )
        assert len(results) > 0
        for r in results:
            clause = r["clause"]
            assert "id" in clause
            assert "title" in clause
            assert "source" in clause
            assert "doc_type" in clause


class TestFormatMixedDocTypes:
    def test_format_with_mixed_types(self, loaded_multi_source_db):
        results = search_similar_clauses(
            "data protection", loaded_multi_source_db, top_k=3
        )
        formatted = format_retrieval_results(results)
        assert isinstance(formatted, str)
        assert "Source [" in formatted
