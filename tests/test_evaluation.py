"""Tests for src/evaluation.py"""

import json
from unittest.mock import patch

import pytest

from src.evaluation import TEST_CASES, JUDGE_PROMPT, evaluate_retrieval, evaluate_generation


class TestTestCases:
    def test_has_four_cases(self):
        assert len(TEST_CASES) == 4

    def test_each_case_has_required_keys(self):
        required = {"name", "clause", "expected_similar_ids", "expected_risk_level", "must_identify"}
        for tc in TEST_CASES:
            assert required.issubset(set(tc.keys())), f"{tc['name']} missing keys"

    def test_expected_ids_exist_in_clauses(self, sample_clauses):
        clause_ids = {c["id"] for c in sample_clauses}
        for tc in TEST_CASES:
            for eid in tc["expected_similar_ids"]:
                assert eid in clause_ids, (
                    f"{eid} not found in clauses.json (test case: {tc['name']})"
                )


class TestEvaluateRetrieval:
    def test_runs_without_error(self, loaded_faiss_db):
        results = evaluate_retrieval(loaded_faiss_db, top_k=3)
        assert "test_cases" in results
        assert "recall_at_k" in results
        assert "mrr" in results

    def test_metrics_are_floats_in_range(self, loaded_faiss_db):
        results = evaluate_retrieval(loaded_faiss_db, top_k=3)
        assert isinstance(results["recall_at_k"], float)
        assert isinstance(results["mrr"], float)
        assert 0.0 <= results["recall_at_k"] <= 1.0
        assert 0.0 <= results["mrr"] <= 1.0


class TestEvaluateGeneration:
    def test_with_mocked_judge(self, loaded_faiss_db, monkeypatch):
        judge_response = json.dumps({
            "risk_accuracy": 4,
            "issue_coverage": 3,
            "actionability": 4,
            "grounding": 3,
            "total": 14,
            "notes": "Good analysis",
        })

        with patch("src.evaluation.analyze_clause") as mock_analyze:
            mock_analyze.return_value = {
                "analysis": "test analysis",
                "retrieved_clauses": [],
                "strategy": "few_shot",
                "model": "mock",
            }

            monkeypatch.setattr(
                loaded_faiss_db["provider"], "chat",
                lambda *a, **kw: judge_response,
            )

            results = evaluate_generation(loaded_faiss_db, strategy="few_shot")

            assert "test_cases" in results
            assert "strategy" in results
            assert "avg_scores" in results
            assert results["strategy"] == "few_shot"


class TestJudgePrompt:
    def test_has_required_placeholders(self):
        assert "{expected_risk}" in JUDGE_PROMPT
        assert "{must_identify}" in JUDGE_PROMPT
        assert "{analysis}" in JUDGE_PROMPT
