"""Tests for citation grounding and draft framing (Unit 2)."""

import json
from unittest.mock import patch

import pytest

from src.retrieval import format_retrieval_results
from src.rag_pipeline import analyze_clause, STRATEGIES
from src.generation import (
    build_basic_prompt,
    build_structured_prompt,
    build_few_shot_prompt,
)


class TestFormatRetrievalResultsIncludesIDs:
    def test_source_id_in_output(self):
        results = [
            {
                "clause": {
                    "id": "nda-001",
                    "title": "Standard Mutual NDA",
                    "type": "NDA",
                    "category": "confidentiality",
                    "text": "Both parties agree to keep information confidential.",
                    "risk_level": "low",
                    "notes": "Standard clause.",
                    "doc_type": "clause",
                    "source": "",
                    "jurisdiction": "",
                    "citation": "",
                    "position": "",
                },
                "score": 0.892,
            }
        ]
        output = format_retrieval_results(results)
        assert "[nda-001]" in output

    def test_multiple_sources_have_correct_ids(self):
        results = [
            {
                "clause": {
                    "id": "emp-001",
                    "title": "Reasonable Non-Compete",
                    "type": "Employment",
                    "category": "non_compete",
                    "text": "12 months, 50-mile radius.",
                    "risk_level": "medium",
                    "notes": "",
                    "doc_type": "clause",
                    "source": "",
                    "jurisdiction": "",
                    "citation": "",
                    "position": "",
                },
                "score": 0.85,
            },
            {
                "clause": {
                    "id": "emp-002",
                    "title": "Overly Broad Non-Compete",
                    "type": "Employment",
                    "category": "non_compete",
                    "text": "3 years, worldwide.",
                    "risk_level": "high",
                    "notes": "",
                    "doc_type": "clause",
                    "source": "",
                    "jurisdiction": "",
                    "citation": "",
                    "position": "",
                },
                "score": 0.80,
            },
        ]
        output = format_retrieval_results(results)
        assert "[emp-001]" in output
        assert "[emp-002]" in output
        assert "Similar Clause" not in output


class TestPipelineOutputHasSources:
    def test_sources_key_present(self, loaded_faiss_db):
        result = analyze_clause("test confidential clause", loaded_faiss_db)
        assert "sources" in result
        assert isinstance(result["sources"], list)

    def test_sources_have_required_fields(self, loaded_faiss_db):
        result = analyze_clause("test confidential clause", loaded_faiss_db)
        for src in result["sources"]:
            assert "id" in src
            assert "title" in src
            assert "score" in src
            assert "risk_level" in src

    def test_sources_score_is_positive(self, loaded_faiss_db):
        with patch("src.rag_pipeline.search_similar_clauses") as mock_search, \
             patch("src.rag_pipeline.generate_analysis") as mock_gen:
            mock_search.return_value = [
                {
                    "clause": {"id": "test-001", "title": "T", "type": "NDA",
                               "category": "c", "text": "t", "risk_level": "low",
                               "notes": "n"},
                    "score": 0.85,
                },
            ]
            mock_gen.return_value = '{"risk_level": "low"}'
            result = analyze_clause("test confidential clause", loaded_faiss_db)
            for src in result["sources"]:
                assert src["score"] > 0


class TestPipelineOutputHasDraftFraming:
    def test_review_status_is_pending_review(self, loaded_faiss_db):
        result = analyze_clause("test clause", loaded_faiss_db)
        assert result["review_status"] == "pending_review"

    def test_disclaimer_contains_draft(self, loaded_faiss_db):
        result = analyze_clause("test clause", loaded_faiss_db)
        assert "DRAFT" in result["disclaimer"]

    def test_disclaimer_mentions_attorney(self, loaded_faiss_db):
        result = analyze_clause("test clause", loaded_faiss_db)
        assert "attorney" in result["disclaimer"].lower() or "Attorney" in result["disclaimer"]


class TestPipelineAnalysisIsParsed:
    def test_analysis_is_dict_when_json_returned(self, loaded_faiss_db):
        # MockProvider.chat() returns valid JSON, so analysis should be a dict
        result = analyze_clause("test clause", loaded_faiss_db)
        assert isinstance(result["analysis"], dict)

    def test_analysis_parse_failure_returns_raw_response(self, loaded_faiss_db, monkeypatch):
        non_json_response = "This is plain text analysis, not JSON."
        monkeypatch.setattr(
            loaded_faiss_db["provider"], "chat",
            lambda *a, **kw: non_json_response,
        )
        result = analyze_clause("test clause", loaded_faiss_db)
        assert isinstance(result["analysis"], dict)
        assert result["analysis"].get("parse_error") is True
        assert "raw_response" in result["analysis"]


class TestSourcesMatchRetrievedClauses:
    def test_sources_correspond_to_actual_retrieved_clauses(self, loaded_faiss_db):
        with patch("src.rag_pipeline.search_similar_clauses") as mock_search, \
             patch("src.rag_pipeline.generate_analysis") as mock_gen:
            mock_search.return_value = [
                {
                    "clause": {
                        "id": "nda-001", "title": "Standard NDA", "type": "NDA",
                        "category": "c", "text": "t", "risk_level": "low",
                        "notes": "n",
                    },
                    "score": 0.9,
                },
                {
                    "clause": {
                        "id": "nda-002", "title": "Unilateral NDA", "type": "NDA",
                        "category": "c", "text": "t", "risk_level": "medium",
                        "notes": "n",
                    },
                    "score": 0.8,
                },
            ]
            mock_gen.return_value = '{"risk_level": "low"}'

            result = analyze_clause("test clause", loaded_faiss_db)

            source_ids = [s["id"] for s in result["sources"]]
            assert "nda-001" in source_ids
            assert "nda-002" in source_ids
            assert len(result["sources"]) == 2


class TestPromptBuildersIncludeCitationInstructions:
    def test_basic_prompt_has_citation_instruction(self):
        messages = build_basic_prompt("test clause", "context")
        system_content = messages[0]["content"].lower()
        assert "source" in system_content or "cite" in system_content

    def test_structured_prompt_has_citation_instruction(self):
        messages = build_structured_prompt("test clause", "context")
        system_content = messages[0]["content"].lower()
        assert "source" in system_content or "cite" in system_content

    def test_few_shot_prompt_has_citation_instruction(self):
        messages = build_few_shot_prompt("test clause", "context")
        # System message is first
        system_content = messages[0]["content"].lower()
        assert "source" in system_content or "cite" in system_content

    def test_few_shot_example_has_inline_citations(self):
        messages = build_few_shot_prompt("test clause", "context")
        # The assistant example message should have inline citations
        assistant_messages = [m for m in messages if m["role"] == "assistant"]
        assert len(assistant_messages) > 0
        example_content = assistant_messages[0]["content"]
        assert "[emp-001]" in example_content or "[emp-002]" in example_content

    def test_few_shot_example_has_sources_used_field(self):
        messages = build_few_shot_prompt("test clause", "context")
        assistant_messages = [m for m in messages if m["role"] == "assistant"]
        assert len(assistant_messages) > 0
        parsed = json.loads(assistant_messages[0]["content"])
        assert "sources_used" in parsed
        assert isinstance(parsed["sources_used"], list)
        assert len(parsed["sources_used"]) > 0

    def test_few_shot_example_has_confidence_fields(self):
        messages = build_few_shot_prompt("test clause", "context")
        assistant_messages = [m for m in messages if m["role"] == "assistant"]
        parsed = json.loads(assistant_messages[0]["content"])
        assert "confidence" in parsed
        assert "confidence_rationale" in parsed
        assert parsed["confidence"] in ("high", "medium", "low")
