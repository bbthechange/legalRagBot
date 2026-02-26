"""Tests for the playbook review pipeline."""

import json

import pytest

from src.playbook_review import (
    load_playbook,
    find_playbook_position,
    review_contract,
    _build_contract_summary,
    _similarity_label,
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

    # Verify clause analyses have the expected top-level keys (flat, not nested)
    for a in result["clause_analyses"]:
        assert "playbook_match" in a
        assert "extracted_text" in a
        assert "position_in_contract" in a
        assert "retrieval_sources" in a
        assert isinstance(a["retrieval_sources"], list)
        for src in a["retrieval_sources"]:
            assert "id" in src
            assert "title" in src
            assert src["similarity"] in ("Strong", "Moderate", "Weak")

    # Verify summary uses the spec's flat keys
    summary = result["summary"]
    assert "total_clauses_reviewed" in summary
    assert "preferred_match" in summary
    assert "fallback_match" in summary
    assert "walk_away_triggered" in summary
    assert "not_in_playbook" in summary
    assert "overall_risk" in summary


class TestSimilarityLabel:
    def test_strong(self):
        assert _similarity_label(0.80) == "Strong"
        assert _similarity_label(0.95) == "Strong"
        assert _similarity_label(1.0) == "Strong"

    def test_moderate(self):
        assert _similarity_label(0.60) == "Moderate"
        assert _similarity_label(0.75) == "Moderate"

    def test_weak(self):
        assert _similarity_label(0.59) == "Weak"
        assert _similarity_label(0.0) == "Weak"


def test_build_contract_summary_counts():
    """_build_contract_summary correctly counts matches."""
    analyses = [
        {"playbook_match": "preferred", "risk_level": "low"},
        {"playbook_match": "fallback", "risk_level": "medium"},
        {"playbook_match": "preferred", "risk_level": "low"},
    ]
    summary = _build_contract_summary(analyses, {})
    assert summary["total_clauses_reviewed"] == 3
    assert summary["preferred_match"] == 2
    assert summary["fallback_match"] == 1
    assert summary["overall_risk"] == "low"


def test_build_contract_summary_walk_away_is_high_risk():
    """_build_contract_summary marks overall risk as high when walk_away present."""
    analyses = [
        {"playbook_match": "preferred", "risk_level": "low"},
        {
            "playbook_match": "walk_away",
            "risk_level": "high",
            "gaps": [{"issue": "Liability cap below threshold", "severity": "high"}],
        },
    ]
    summary = _build_contract_summary(analyses, {})
    assert summary["overall_risk"] == "high"
    assert summary["walk_away_triggered"] == 1
    assert len(summary["critical_issues"]) == 1
    assert "Liability cap" in summary["critical_issues"][0]
