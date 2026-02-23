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

    def test_metadata_fallback(self, loaded_faiss_db):
        """When a clause ID isn't in db['clauses'], reconstruct from metadata."""
        original_clauses = loaded_faiss_db["clauses"]
        removed = original_clauses[0]
        loaded_faiss_db["clauses"] = [
            c for c in original_clauses if c["id"] != removed["id"]
        ]

        results = search_similar_clauses(
            f"{removed['title']}: {removed['text']}", loaded_faiss_db, top_k=15
        )

        found = [r for r in results if r["clause"]["id"] == removed["id"]]
        assert len(found) > 0, f"Expected to find {removed['id']} via metadata fallback"
        clause = found[0]["clause"]
        assert clause["title"] == removed["title"]
        assert clause["text"] == removed["text"]

        # Restore
        loaded_faiss_db["clauses"] = original_clauses


class TestFormatRetrievalResults:
    def test_produces_formatted_string(self, loaded_faiss_db):
        results = search_similar_clauses(
            "test clause", loaded_faiss_db, top_k=2
        )
        formatted = format_retrieval_results(results)
        assert isinstance(formatted, str)
        assert "Similar Clause" in formatted
        assert "similarity:" in formatted
        assert "Type:" in formatted
        assert "Risk Level:" in formatted
