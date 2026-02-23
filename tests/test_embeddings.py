"""Tests for src/embeddings.py"""

import logging
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from src.embeddings import get_embeddings, infer_practice_area


class TestGetEmbeddings:
    def test_returns_correct_shape(self, mock_provider):
        texts = ["hello world", "test text"]
        result = get_embeddings(texts, mock_provider)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 2
        assert result.shape[1] == 1536


class TestInferPracticeArea:
    def test_nda_maps_to_ip(self):
        assert infer_practice_area("NDA") == "intellectual_property"

    def test_employment_maps_to_employment_labor(self):
        assert infer_practice_area("Employment") == "employment_labor"

    def test_service_agreement_maps_to_commercial(self):
        assert infer_practice_area("Service Agreement") == "commercial_contracts"

    def test_unknown_maps_to_general(self):
        assert infer_practice_area("Unknown Type") == "general"


class TestLoadClauseDatabase:
    def test_returns_dict_with_correct_keys_and_counts(self, mock_provider):
        mock_store = MagicMock()
        mock_store.upsert.return_value = 15
        mock_store.total_vectors = 0

        with patch("src.embeddings.create_provider", return_value=mock_provider), \
             patch("src.embeddings.create_vector_store", return_value=mock_store):
            from src.embeddings import load_clause_database
            db = load_clause_database()

            assert "store" in db
            assert "clauses" in db
            assert "provider" in db
            assert len(db["clauses"]) == 15

            # Verify upsert was called with 15 items
            call_args = mock_store.upsert.call_args
            assert len(call_args[0][0]) == 15  # 15 ids
            assert call_args[0][1].shape[0] == 15  # 15 embeddings


class TestLoadDocuments:
    def test_load_from_data_path(self, mock_provider):
        mock_store = MagicMock()
        mock_store.upsert.return_value = 15
        mock_store.total_vectors = 0

        with patch("src.embeddings.create_provider", return_value=mock_provider), \
             patch("src.embeddings.create_vector_store", return_value=mock_store):
            from src.embeddings import load_documents
            db = load_documents(data_path="data/clauses.json")

            assert "store" in db
            assert "documents" in db
            assert "clauses" in db
            assert "provider" in db
            assert db["documents"] is db["clauses"]
            assert len(db["documents"]) == 15

    def test_load_from_documents_list(self, mock_provider, sample_unified_documents):
        mock_store = MagicMock()
        mock_store.upsert.return_value = 3
        mock_store.total_vectors = 0

        with patch("src.embeddings.create_provider", return_value=mock_provider), \
             patch("src.embeddings.create_vector_store", return_value=mock_store):
            from src.embeddings import load_documents
            db = load_documents(documents=sample_unified_documents)

            assert len(db["documents"]) == 3
            assert db["documents"][0]["doc_id"] == "uni-001"

    def test_backward_compat(self, mock_provider):
        mock_store = MagicMock()
        mock_store.upsert.return_value = 15
        mock_store.total_vectors = 0

        with patch("src.embeddings.create_provider", return_value=mock_provider), \
             patch("src.embeddings.create_vector_store", return_value=mock_store):
            from src.embeddings import load_clause_database
            db = load_clause_database()

            assert "store" in db
            assert "clauses" in db
            assert "provider" in db

    def test_rejects_neither_arg(self):
        from src.embeddings import load_documents
        with pytest.raises(ValueError, match="Must provide either"):
            load_documents()

    def test_validates_documents(self, mock_provider, caplog):
        mock_store = MagicMock()
        mock_store.upsert.return_value = 1
        mock_store.total_vectors = 0

        invalid_doc = {
            "doc_id": "bad-001",
            "source": "invalid_source",
            "doc_type": "clause",
            "title": "Test",
            "text": "Test text",
        }

        with patch("src.embeddings.create_provider", return_value=mock_provider), \
             patch("src.embeddings.create_vector_store", return_value=mock_store), \
             caplog.at_level(logging.WARNING):
            from src.embeddings import load_documents
            db = load_documents(documents=[invalid_doc])
            assert len(db["documents"]) == 1
            assert any("validation" in r.message for r in caplog.records)
