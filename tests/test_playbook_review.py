"""Tests for the playbook review pipeline."""

import json

import pytest

from src.playbook_review import (
    load_playbook,
    find_playbook_position,
    review_contract,
    _build_contract_summary,
)


@pytest.fixture
def sample_playbook(tmp_path):
    """Create a minimal playbook file for testing."""
    playbook = {
        "playbook_id": "test-review",
        "name": "Test Review Playbook",
        "description": "A test playbook for review tests.",
        "clauses": [
            {
                "clause_type": "termination",
                "preferred_position": "30 day cure period with pro-rata refund.",
                "fallback_position": "30 day cure, no refund.",
                "walk_away": "No termination rights.",
                "risk_factors": ["Long cure periods"],
                "notes": "Check data return timelines.",
            },
            {
                "clause_type": "indemnification",
                "preferred_position": "Mutual indemnification for IP.",
                "fallback_position": "Vendor IP indemnification only.",
                "walk_away": "No indemnification.",
                "risk_factors": ["Sole remedy"],
                "notes": "Ensure survives termination.",
            },
        ],
    }
    path = tmp_path / "test-review.json"
    path.write_text(json.dumps(playbook))
    return str(path)


def test_load_playbook(sample_playbook):
    """load_playbook reads and parses the JSON file."""
    pb = load_playbook(sample_playbook)
    assert pb["playbook_id"] == "test-review"
    assert pb["name"] == "Test Review Playbook"
    assert len(pb["clauses"]) == 2


def test_find_playbook_position_found(sample_playbook):
    """find_playbook_position returns the matching clause position."""
    pb = load_playbook(sample_playbook)
    pos = find_playbook_position("termination", pb)
    assert pos is not None
    assert pos["clause_type"] == "termination"
    assert "30 day" in pos["preferred_position"]


def test_find_playbook_position_not_found(sample_playbook):
    """find_playbook_position returns None for unknown clause types."""
    pb = load_playbook(sample_playbook)
    pos = find_playbook_position("nonexistent_clause", pb)
    assert pos is None


def test_review_contract_with_mocked_extraction(
    sample_playbook, loaded_multi_source_db, monkeypatch
):
    """review_contract returns structured results with patched extract_clauses."""
    fake_clauses = [
        {
            "text": "Either party may terminate on 30 days notice.",
            "position": 0,
            "heading": "TERMINATION",
            "clause_type": "termination",
            "confidence": "high",
        },
        {
            "text": "Vendor shall indemnify customer for all IP claims.",
            "position": 1,
            "heading": "INDEMNIFICATION",
            "clause_type": "indemnification",
            "confidence": "high",
        },
    ]
    monkeypatch.setattr(
        "src.playbook_review.extract_clauses", lambda text, provider: fake_clauses
    )

    result = review_contract("dummy contract text " * 5, sample_playbook, loaded_multi_source_db)

    assert result["playbook"] == "Test Review Playbook"
    assert result["total_clauses"] == 2
    assert len(result["clause_analyses"]) == 2
    assert result["review_status"] == "pending_review"
    assert "disclaimer" in result


def test_build_contract_summary_counts():
    """_build_contract_summary correctly counts alignments."""
    analyses = [
        {"clause_type": "a", "analysis": {"alignment": "preferred", "risk_level": "low"}},
        {"clause_type": "b", "analysis": {"alignment": "fallback", "risk_level": "medium"}},
        {"clause_type": "c", "analysis": {"alignment": "preferred", "risk_level": "low"}},
    ]
    summary = _build_contract_summary(analyses, {})
    assert summary["total_clauses"] == 3
    assert summary["alignment_counts"]["preferred"] == 2
    assert summary["alignment_counts"]["fallback"] == 1
    assert summary["overall_risk"] == "low"


def test_build_contract_summary_walk_away_is_high_risk():
    """_build_contract_summary marks overall risk as high when walk_away present."""
    analyses = [
        {"clause_type": "a", "analysis": {"alignment": "preferred", "risk_level": "low"}},
        {
            "clause_type": "b",
            "heading": "LIABILITY",
            "analysis": {"alignment": "walk_away", "risk_level": "high", "analysis": "Bad clause"},
        },
    ]
    summary = _build_contract_summary(analyses, {})
    assert summary["overall_risk"] == "high"
    assert len(summary["critical_issues"]) == 1
    assert summary["critical_issues"][0]["clause_type"] == "b"
