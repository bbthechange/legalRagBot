"""Tests for src/retrieval.py"""

import pytest

from src.retrieval import search_similar_clauses, format_retrieval_results


class TestSearchSimilarClauses:
    def test_returns_results_with_clause_and_score(self, loaded_faiss_db):
        results = search_similar_clauses(
            "confidentiality agreement", loaded_faiss_db, top_k=3
        )
        assert len(results) > 0
        for r in results:
            assert "clause" in r
            assert "score" in r

    def test_clause_has_expected_keys(self, loaded_faiss_db):
        results = search_similar_clauses(
            "confidentiality agreement", loaded_faiss_db, top_k=1
        )
        assert len(results) >= 1
        clause = results[0]["clause"]
        expected_keys = {"id", "title", "type", "category", "text", "risk_level", "notes"}
        assert expected_keys.issubset(set(clause.keys()))

    def test_results_ordered_by_score(self, loaded_faiss_db):
        results = search_similar_clauses(
            "non-compete employment", loaded_faiss_db, top_k=5
        )
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_respected(self, loaded_faiss_db):
        results = search_similar_clauses(
            "liability clause", loaded_faiss_db, top_k=2
        )
        assert len(results) <= 2

    def test_filter_by_type(self, loaded_faiss_db):
        results = search_similar_clauses(
            "agreement clause", loaded_faiss_db, top_k=5, filters={"type": "NDA"}
        )
        for r in results:
            assert r["clause"]["type"] == "NDA"

    def test_reconstructs_from_metadata(self, loaded_faiss_db):
        """All results are reconstructed from vector store metadata."""
        results = search_similar_clauses(
            "confidentiality agreement", loaded_faiss_db, top_k=3
        )
        for r in results:
            clause = r["clause"]
            assert "id" in clause
            assert "title" in clause
            assert "type" in clause
            assert "text" in clause
            assert "source" in clause
            assert "doc_type" in clause


class TestFormatRetrievalResults:
    def test_produces_formatted_string(self, loaded_faiss_db):
        results = search_similar_clauses(
            "test clause", loaded_faiss_db, top_k=2
        )
        formatted = format_retrieval_results(results)
        assert isinstance(formatted, str)
        assert "Source [" in formatted
        assert "similarity:" in formatted
        assert "Type:" in formatted
        assert "Risk Level:" in formatted

    def test_format_statute_type(self, loaded_multi_source_db):
        results = search_similar_clauses(
            "data protection law", loaded_multi_source_db, top_k=3
        )
        statute_results = [r for r in results if r["clause"]["doc_type"] == "statute"]
        assert len(statute_results) > 0, "Expected at least one statute result"
        formatted = format_retrieval_results(statute_results)
        assert "Jurisdiction:" in formatted
        assert "Jurisdiction: UK" in formatted

    def test_format_playbook_type(self, loaded_multi_source_db):
        results = search_similar_clauses(
            "indemnification playbook", loaded_multi_source_db, top_k=3
        )
        playbook_results = [r for r in results if r["clause"]["doc_type"] == "playbook"]
        assert len(playbook_results) > 0, "Expected at least one playbook result"
        formatted = format_retrieval_results(playbook_results)
        assert "Playbook:" in formatted
