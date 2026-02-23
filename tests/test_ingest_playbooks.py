"""Tests for the PlaybookIngestor."""

import json

import pytest

from src.ingest.playbooks import PlaybookIngestor


@pytest.fixture
def playbook_dir(tmp_path):
    """Create a temporary playbook directory with a test playbook."""
    playbook = {
        "playbook_id": "test-playbook",
        "name": "Test Playbook",
        "description": "A test playbook.",
        "clauses": [
            {
                "clause_type": "limitation_of_liability",
                "preferred_position": "Mutual cap equal to 12 months fees.",
                "fallback_position": "Cap equal to fees paid in prior 12 months.",
                "walk_away": "Cap less than 3 months fees.",
                "risk_factors": ["Asymmetric cap"],
                "notes": "Check per-claim vs aggregate.",
            },
            {
                "clause_type": "indemnification",
                "preferred_position": "Vendor indemnifies for IP infringement.",
                "fallback_position": "IP indemnification only.",
                "walk_away": "No indemnification.",
                "risk_factors": ["Sole remedy clauses"],
                "notes": "Ensure survives termination.",
            },
        ],
    }
    (tmp_path / "test.json").write_text(json.dumps(playbook))
    return tmp_path


def test_load_raw_reads_files(playbook_dir):
    """load_raw() reads playbook files and returns playbook-level dicts."""
    ingestor = PlaybookIngestor(data_dir=str(playbook_dir))
    raw = ingestor.load_raw()
    assert len(raw) == 1  # One playbook file
    assert raw[0]["playbook_id"] == "test-playbook"
    assert len(raw[0]["clauses"]) == 2


def test_transform_one_doc_per_clause(playbook_dir):
    """transform() produces one document per clause."""
    ingestor = PlaybookIngestor(data_dir=str(playbook_dir))
    raw = ingestor.load_raw()
    docs = ingestor.transform(raw)
    assert len(docs) == 2
    assert docs[0]["doc_id"] == "playbook-test-playbook-limitation_of_liability"
    assert docs[1]["doc_id"] == "playbook-test-playbook-indemnification"


def test_transform_correct_source_and_doc_type(playbook_dir):
    """transform() sets source='common_paper' and doc_type='playbook'."""
    ingestor = PlaybookIngestor(data_dir=str(playbook_dir))
    raw = ingestor.load_raw()
    docs = ingestor.transform(raw)
    for doc in docs:
        assert doc["source"] == "common_paper"
        assert doc["doc_type"] == "playbook"
        assert doc["metadata"]["practice_area"] == "commercial_contracts"
        assert doc["metadata"]["risk_level"] is None


def test_transform_text_contains_all_positions(playbook_dir):
    """transform() text field contains preferred, fallback, and walk-away positions."""
    ingestor = PlaybookIngestor(data_dir=str(playbook_dir))
    raw = ingestor.load_raw()
    docs = ingestor.transform(raw)
    text = docs[0]["text"]
    assert "Preferred:" in text
    assert "Fallback:" in text
    assert "Walk-away:" in text
    assert "12 months fees" in text
