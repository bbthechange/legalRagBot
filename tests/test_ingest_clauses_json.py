"""Tests for ClausesJsonIngestor."""

import json
import pytest
from src.ingest.clauses_json import ClausesJsonIngestor


def test_clauses_json_ingest_count(tmp_path):
    """ClausesJsonIngestor().ingest() returns the correct number of documents."""
    clauses = [
        {
            "id": f"test-{i:03d}",
            "type": "NDA",
            "category": "confidentiality",
            "title": f"Test Clause {i}",
            "text": f"Clause text for document {i}.",
            "risk_level": "low",
            "notes": f"Note {i}",
        }
        for i in range(5)
    ]
    data_file = tmp_path / "clauses.json"
    data_file.write_text(json.dumps(clauses))

    ingestor = ClausesJsonIngestor(data_path=str(data_file))
    result = ingestor.ingest()
    assert len(result) == 5


def test_clauses_json_source_and_doc_type(tmp_path):
    """All documents have correct source and doc_type."""
    clauses = [
        {
            "id": "c-001",
            "type": "NDA",
            "category": "confidentiality",
            "title": "NDA Clause",
            "text": "Confidential information clause text.",
            "risk_level": "low",
            "notes": "Standard NDA",
        }
    ]
    data_file = tmp_path / "clauses.json"
    data_file.write_text(json.dumps(clauses))

    ingestor = ClausesJsonIngestor(data_path=str(data_file))
    result = ingestor.ingest()

    assert len(result) == 1
    assert result[0]["source"] == "clauses_json"
    assert result[0]["doc_type"] == "clause"


def test_clauses_json_doc_ids_match_source(tmp_path):
    """All doc_ids match the original clause IDs from the JSON file."""
    clauses = [
        {
            "id": f"clause-{i}",
            "type": "Employment",
            "category": "restrictive_covenant",
            "title": f"Employment Clause {i}",
            "text": f"Employment clause text number {i}.",
            "risk_level": "medium",
            "notes": f"Employment note {i}",
        }
        for i in range(3)
    ]
    data_file = tmp_path / "clauses.json"
    data_file.write_text(json.dumps(clauses))

    ingestor = ClausesJsonIngestor(data_path=str(data_file))
    result = ingestor.ingest()

    result_ids = {d["doc_id"] for d in result}
    expected_ids = {f"clause-{i}" for i in range(3)}
    assert result_ids == expected_ids


def test_clauses_json_metadata_fields(tmp_path):
    """Metadata includes clause_type, category, risk_level, notes, practice_area."""
    clauses = [
        {
            "id": "meta-001",
            "type": "Service Agreement",
            "category": "payment",
            "title": "Service Agreement Clause",
            "text": "Payment terms and conditions apply to this agreement.",
            "risk_level": "high",
            "notes": "Review carefully",
        }
    ]
    data_file = tmp_path / "clauses.json"
    data_file.write_text(json.dumps(clauses))

    ingestor = ClausesJsonIngestor(data_path=str(data_file))
    result = ingestor.ingest()

    assert len(result) == 1
    meta = result[0]["metadata"]
    assert meta["clause_type"] == "Service Agreement"
    assert meta["category"] == "payment"
    assert meta["risk_level"] == "high"
    assert meta["notes"] == "Review carefully"
    assert meta["practice_area"] == "commercial_contracts"


def test_clauses_json_practice_area_mapping(tmp_path):
    """Practice areas are correctly mapped from clause type."""
    clauses = [
        {
            "id": "nda-001",
            "type": "NDA",
            "category": "confidentiality",
            "title": "NDA",
            "text": "NDA text here.",
            "risk_level": "low",
            "notes": "",
        },
        {
            "id": "emp-001",
            "type": "Employment",
            "category": "employment",
            "title": "Employment",
            "text": "Employment text here.",
            "risk_level": "low",
            "notes": "",
        },
        {
            "id": "svc-001",
            "type": "Service Agreement",
            "category": "services",
            "title": "Service Agreement",
            "text": "Service agreement text here.",
            "risk_level": "low",
            "notes": "",
        },
        {
            "id": "oth-001",
            "type": "Other",
            "category": "misc",
            "title": "Other Clause",
            "text": "Other clause text here.",
            "risk_level": "low",
            "notes": "",
        },
    ]
    data_file = tmp_path / "clauses.json"
    data_file.write_text(json.dumps(clauses))

    ingestor = ClausesJsonIngestor(data_path=str(data_file))
    result = ingestor.ingest()

    by_id = {d["doc_id"]: d for d in result}
    assert by_id["nda-001"]["metadata"]["practice_area"] == "intellectual_property"
    assert by_id["emp-001"]["metadata"]["practice_area"] == "employment_labor"
    assert by_id["svc-001"]["metadata"]["practice_area"] == "commercial_contracts"
    assert by_id["oth-001"]["metadata"]["practice_area"] == "general"


def test_clauses_json_real_file():
    """Integration: ClausesJsonIngestor loads the real data/clauses.json."""
    ingestor = ClausesJsonIngestor(data_path="data/clauses.json")
    result = ingestor.ingest()
    assert len(result) == 15
    for doc in result:
        assert doc["source"] == "clauses_json"
        assert doc["doc_type"] == "clause"
        assert doc["doc_id"]
        assert doc["text"]
