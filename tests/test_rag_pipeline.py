"""Tests for src/rag_pipeline.py"""

from unittest.mock import patch

from src.rag_pipeline import STRATEGIES, analyze_clause


class TestStrategies:
    def test_has_expected_keys(self):
        assert set(STRATEGIES.keys()) == {"basic", "structured", "few_shot"}
        for fn in STRATEGIES.values():
            assert callable(fn)


class TestAnalyzeClause:
    def test_returns_expected_dict(self, loaded_faiss_db):
        with patch("src.rag_pipeline.search_similar_clauses") as mock_search, \
             patch("src.rag_pipeline.generate_analysis") as mock_gen:
            mock_search.return_value = [
                {
                    "clause": {
                        "id": "test", "title": "T", "type": "NDA",
                        "category": "c", "text": "t", "risk_level": "low",
                        "notes": "n",
                    },
                    "score": 0.9,
                },
            ]
            mock_gen.return_value = '{"risk_level": "low"}'

            result = analyze_clause("test clause", loaded_faiss_db)

            assert "analysis" in result
            assert "retrieved_clauses" in result
            assert "strategy" in result
            assert "model" in result

    def test_works_with_each_strategy(self, loaded_faiss_db):
        for strategy_name in STRATEGIES:
            with patch("src.rag_pipeline.search_similar_clauses") as mock_search, \
                 patch("src.rag_pipeline.generate_analysis") as mock_gen:
                mock_search.return_value = []
                mock_gen.return_value = "analysis"

                result = analyze_clause(
                    "test clause", loaded_faiss_db, strategy=strategy_name
                )
                assert result["strategy"] == strategy_name

    def test_default_strategy_is_few_shot(self, loaded_faiss_db):
        with patch("src.rag_pipeline.search_similar_clauses") as mock_search, \
             patch("src.rag_pipeline.generate_analysis") as mock_gen:
            mock_search.return_value = []
            mock_gen.return_value = "analysis"

            result = analyze_clause("test clause", loaded_faiss_db)
            assert result["strategy"] == "few_shot"
