"""Tests for src/schemas.py"""

import pytest

from src.schemas import (
    validate_document,
    normalize_document,
    VALID_SOURCES,
    VALID_DOC_TYPES,
)


def _valid_doc():
    return {
        "doc_id": "test-001",
        "source": "clauses_json",
        "doc_type": "clause",
        "title": "Test Clause",
        "text": "This is a test clause.",
    }


class TestValidateDocument:
    def test_valid_doc_passes(self):
        assert validate_document(_valid_doc()) == []

    def test_missing_field_error(self):
        doc = _valid_doc()
        del doc["doc_id"]
        errors = validate_document(doc)
        assert any("Missing required field: doc_id" in e for e in errors)

    def test_empty_field_error(self):
        doc = _valid_doc()
        doc["title"] = ""
        errors = validate_document(doc)
        assert any("Empty required field: title" in e for e in errors)

    def test_unknown_source_error(self):
        doc = _valid_doc()
        doc["source"] = "invalid_source"
        errors = validate_document(doc)
        assert any("Unknown source" in e for e in errors)

    def test_unknown_doc_type_error(self):
        doc = _valid_doc()
        doc["doc_type"] = "invalid_type"
        errors = validate_document(doc)
        assert any("Unknown doc_type" in e for e in errors)

    def test_all_valid_sources_accepted(self):
        for source in VALID_SOURCES:
            doc = _valid_doc()
            doc["source"] = source
            assert validate_document(doc) == [], f"Source {source} rejected"

    def test_all_valid_doc_types_accepted(self):
        for doc_type in VALID_DOC_TYPES:
            doc = _valid_doc()
            doc["doc_type"] = doc_type
            assert validate_document(doc) == [], f"Doc type {doc_type} rejected"


class TestNormalizeDocument:
    def test_fills_missing_metadata_with_none(self):
        doc = _valid_doc()
        result = normalize_document(doc)
        assert "metadata" in result
        assert result["metadata"]["clause_type"] is None
        assert result["metadata"]["category"] is None
        assert result["metadata"]["risk_level"] is None

    def test_preserves_existing_values(self):
        doc = _valid_doc()
        doc["metadata"] = {"clause_type": "NDA", "risk_level": "high"}
        result = normalize_document(doc)
        assert result["metadata"]["clause_type"] == "NDA"
        assert result["metadata"]["risk_level"] == "high"
        assert result["metadata"]["category"] is None

    def test_returns_new_dict(self):
        doc = _valid_doc()
        result = normalize_document(doc)
        assert result is not doc
        assert result["metadata"] is not doc.get("metadata", {})
