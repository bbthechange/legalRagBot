"""Tests for the StatuteIngestor."""

import json

import pytest

from src.ingest.statutes import StatuteIngestor


@pytest.fixture
def statute_dir(tmp_path):
    """Create a temporary statute directory with two test statutes."""
    ca = {
        "jurisdiction": "California",
        "jurisdiction_abbr": "CA",
        "statute_citation": "Cal. Civ. Code 1798.29",
        "effective_date": "2024-01-01",
        "personal_information_definition": [
            "SSN, driver's license number",
            "Financial account number with access code",
        ],
        "breach_definition": "Unauthorized acquisition of computerized data",
        "encryption_safe_harbor": True,
        "encryption_safe_harbor_details": "No notification if data was encrypted",
        "notification_timeline": "Without unreasonable delay",
        "notification_timeline_days": None,
        "notification_recipients": {
            "individuals": True,
            "attorney_general": True,
            "ag_threshold": 500,
        },
        "notification_content_requirements": ["Entity name", "Data types"],
        "private_right_of_action": True,
        "penalties": "Up to $7,500 per violation",
        "special_provisions": ["CCPA private right of action"],
    }
    fl = {
        "jurisdiction": "Florida",
        "jurisdiction_abbr": "FL",
        "statute_citation": "Fla. Stat. 501.171",
        "effective_date": "2014-07-01",
        "personal_information_definition": [
            "SSN, driver's license number",
        ],
        "breach_definition": "Unauthorized access of data",
        "encryption_safe_harbor": True,
        "encryption_safe_harbor_details": "Encrypted data exempt unless key also taken",
        "notification_timeline": "No later than 30 days",
        "notification_timeline_days": 30,
        "notification_recipients": {
            "individuals": True,
            "attorney_general": True,
            "ag_threshold": 500,
        },
        "notification_content_requirements": ["Date of breach", "Data types"],
        "private_right_of_action": False,
        "penalties": "Civil penalties per day",
        "special_provisions": ["30-day deadline"],
    }
    (tmp_path / "california.json").write_text(json.dumps(ca))
    (tmp_path / "florida.json").write_text(json.dumps(fl))
    return tmp_path


def test_load_raw_reads_files(statute_dir):
    """load_raw() reads statute JSON files and returns statute-level dicts."""
    ingestor = StatuteIngestor(data_dir=str(statute_dir))
    raw = ingestor.load_raw()
    assert len(raw) == 2
    jurisdictions = {s["jurisdiction"] for s in raw}
    assert "California" in jurisdictions
    assert "Florida" in jurisdictions


def test_transform_creates_multiple_docs_per_state(statute_dir):
    """transform() produces multiple documents per state (summary + provisions)."""
    ingestor = StatuteIngestor(data_dir=str(statute_dir))
    raw = ingestor.load_raw()
    docs = ingestor.transform(raw)
    # Each state: summary + PI definition + timeline + safe harbor = 4 docs
    assert len(docs) == 8  # 4 per state * 2 states


def test_transform_correct_source_and_doc_type(statute_dir):
    """All documents have source='statutes' and doc_type='statute'."""
    ingestor = StatuteIngestor(data_dir=str(statute_dir))
    raw = ingestor.load_raw()
    docs = ingestor.transform(raw)
    for doc in docs:
        assert doc["source"] == "statutes"
        assert doc["doc_type"] == "statute"


def test_transform_metadata_includes_required_fields(statute_dir):
    """Metadata includes jurisdiction, citation, and practice_area."""
    ingestor = StatuteIngestor(data_dir=str(statute_dir))
    raw = ingestor.load_raw()
    docs = ingestor.transform(raw)
    for doc in docs:
        meta = doc["metadata"]
        assert "jurisdiction" in meta
        assert "citation" in meta
        assert meta["practice_area"] == "privacy"


def test_transform_summary_doc_ids(statute_dir):
    """Summary documents have expected doc_id format."""
    ingestor = StatuteIngestor(data_dir=str(statute_dir))
    raw = ingestor.load_raw()
    docs = ingestor.transform(raw)
    summary_ids = [d["doc_id"] for d in docs if d["doc_id"].endswith("-summary")]
    assert "statute-ca-summary" in summary_ids
    assert "statute-fl-summary" in summary_ids


def test_transform_timeline_includes_days(statute_dir):
    """Timeline documents include day counts when available."""
    ingestor = StatuteIngestor(data_dir=str(statute_dir))
    raw = ingestor.load_raw()
    docs = ingestor.transform(raw)
    fl_timeline = [d for d in docs if d["doc_id"] == "statute-fl-timeline"]
    assert len(fl_timeline) == 1
    assert "30 days" in fl_timeline[0]["text"]


def test_empty_directory_returns_empty(tmp_path):
    """Empty statute directory returns empty list."""
    ingestor = StatuteIngestor(data_dir=str(tmp_path))
    raw = ingestor.load_raw()
    assert raw == []
    docs = ingestor.transform(raw)
    assert docs == []


def test_ingest_validates_documents(statute_dir):
    """Full ingest() pipeline validates and normalizes documents."""
    ingestor = StatuteIngestor(data_dir=str(statute_dir))
    docs = ingestor.ingest()
    assert len(docs) == 8
    for doc in docs:
        # Normalized documents have top-level keys
        assert "doc_id" in doc
        assert "source" in doc
        assert "doc_type" in doc
        assert "title" in doc
        assert "text" in doc
        assert "metadata" in doc
