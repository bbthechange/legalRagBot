"""Tests for BaseIngestor — the abstract ingest pipeline."""

import logging
import pytest
from src.ingest.base import BaseIngestor


class SimpleIngestor(BaseIngestor):
    """Concrete implementation for testing."""
    source_name = "test_source"

    def __init__(self, raw_data):
        self._raw_data = raw_data

    def load_raw(self):
        return self._raw_data

    def transform(self, raw_data):
        return raw_data


def _valid_doc(doc_id="test-001"):
    return {
        "doc_id": doc_id,
        "source": "clauses_json",
        "doc_type": "clause",
        "title": "Test Clause",
        "text": "This is a test clause text.",
        "metadata": {
            "clause_type": "nda",
            "category": "confidentiality",
            "risk_level": "low",
            "notes": "Test note",
            "practice_area": "general",
        },
    }


def test_ingest_returns_valid_documents():
    """BaseIngestor.ingest() works through load → transform → validate → normalize."""
    raw = [_valid_doc("doc-001"), _valid_doc("doc-002")]
    ingestor = SimpleIngestor(raw)
    result = ingestor.ingest()
    assert len(result) == 2
    assert all(d["doc_id"] in ("doc-001", "doc-002") for d in result)


def test_ingest_filters_invalid_documents(caplog):
    """Invalid documents (missing required fields) are filtered out with a warning."""
    invalid = {"doc_id": "bad-doc", "source": "clauses_json"}  # missing fields
    valid = _valid_doc("good-doc")

    ingestor = SimpleIngestor([invalid, valid])
    with caplog.at_level(logging.WARNING):
        result = ingestor.ingest()

    assert len(result) == 1
    assert result[0]["doc_id"] == "good-doc"
    assert any("Invalid doc" in r.message for r in caplog.records)


def test_ingest_normalizes_valid_documents():
    """Valid documents are passed through normalize_document()."""
    doc = {
        "doc_id": "norm-001",
        "source": "clauses_json",
        "doc_type": "clause",
        "title": "Normalization Test",
        "text": "Some text here.",
        "metadata": {"clause_type": "nda"},
    }
    ingestor = SimpleIngestor([doc])
    result = ingestor.ingest()

    assert len(result) == 1
    # normalize_document fills in all metadata keys with None if absent
    assert "practice_area" in result[0]["metadata"]


def test_ingest_logs_error_count(caplog):
    """Error count is logged when invalid documents are skipped."""
    invalids = [
        {"doc_id": f"bad-{i}", "source": "clauses_json"}  # missing required fields
        for i in range(3)
    ]
    valid = _valid_doc("good-001")

    ingestor = SimpleIngestor(invalids + [valid])
    with caplog.at_level(logging.WARNING):
        result = ingestor.ingest()

    assert len(result) == 1
    warning_messages = [r.message for r in caplog.records if r.levelname == "WARNING"]
    assert any("3 invalid documents skipped" in msg for msg in warning_messages)


def test_ingest_empty_source():
    """Empty source returns empty list without errors."""
    ingestor = SimpleIngestor([])
    result = ingestor.ingest()
    assert result == []
