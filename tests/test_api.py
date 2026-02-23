"""
API endpoint tests.

Uses FastAPI TestClient for in-process testing.
Mocks the database to avoid real LLM/embedding API calls.
"""
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient


@pytest.fixture
def client(loaded_faiss_db):
    """
    Create a test client with a pre-loaded mock database.
    Patches get_db() to return the mock DB instead of calling load_clause_database().
    """
    from src.api import app, get_db

    def override_get_db():
        return loaded_faiss_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


# --- Health endpoint ---

class TestHealth:
    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_response_has_required_fields(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "status" in data
        assert "provider" in data
        assert "document_count" in data
        assert "available_strategies" in data

    def test_document_count_matches_store(self, client, loaded_faiss_db):
        resp = client.get("/health")
        data = resp.json()
        assert data["document_count"] == loaded_faiss_db["store"].total_vectors

    def test_status_is_healthy(self, client):
        resp = client.get("/health")
        assert resp.json()["status"] == "healthy"

    def test_available_strategies_nonempty(self, client):
        resp = client.get("/health")
        strategies = resp.json()["available_strategies"]
        assert isinstance(strategies, list)
        assert len(strategies) > 0

    def test_health_requires_no_auth(self, client, monkeypatch):
        monkeypatch.setenv("API_KEY", "secret-key")
        # No X-API-Key header — should still succeed
        resp = client.get("/health")
        assert resp.status_code == 200


# --- Analyze endpoint ---

class TestAnalyze:
    def test_valid_request_returns_200(self, client):
        resp = client.post("/analyze", json={
            "clause_text": "Employee agrees not to compete for 2 years worldwide."
        })
        assert resp.status_code == 200

    def test_response_has_required_fields(self, client):
        resp = client.post("/analyze", json={
            "clause_text": "Employee agrees not to compete for 2 years worldwide."
        })
        data = resp.json()
        assert "analysis" in data
        assert "sources" in data
        assert "review_status" in data
        assert "disclaimer" in data
        assert "strategy" in data
        assert "model" in data
        assert "top_k" in data

    def test_short_clause_text_returns_422(self, client):
        resp = client.post("/analyze", json={"clause_text": "short"})
        assert resp.status_code == 422

    def test_empty_clause_text_returns_422(self, client):
        resp = client.post("/analyze", json={"clause_text": ""})
        assert resp.status_code == 422

    def test_missing_clause_text_returns_422(self, client):
        resp = client.post("/analyze", json={})
        assert resp.status_code == 422

    def test_unknown_strategy_returns_400(self, client):
        resp = client.post("/analyze", json={
            "clause_text": "Employee agrees not to compete for 2 years worldwide.",
            "strategy": "nonexistent_strategy"
        })
        assert resp.status_code == 400
        assert "nonexistent_strategy" in resp.json()["detail"]

    def test_unknown_strategy_message_includes_available(self, client):
        resp = client.post("/analyze", json={
            "clause_text": "Employee agrees not to compete for 2 years worldwide.",
            "strategy": "bad_strategy"
        })
        detail = resp.json()["detail"]
        assert "Available" in detail or "available" in detail

    def test_review_status_is_pending_review(self, client):
        resp = client.post("/analyze", json={
            "clause_text": "Employee agrees not to compete for 2 years worldwide."
        })
        assert resp.json()["review_status"] == "pending_review"

    def test_disclaimer_contains_draft(self, client):
        resp = client.post("/analyze", json={
            "clause_text": "Employee agrees not to compete for 2 years worldwide."
        })
        assert "DRAFT" in resp.json()["disclaimer"]

    def test_sources_is_list_with_expected_fields(self, client):
        resp = client.post("/analyze", json={
            "clause_text": "Employee agrees not to compete for 2 years worldwide.",
            "top_k": 2
        })
        sources = resp.json()["sources"]
        assert isinstance(sources, list)
        for s in sources:
            assert "id" in s
            assert "title" in s
            assert "score" in s

    def test_custom_strategy_is_used(self, client):
        resp = client.post("/analyze", json={
            "clause_text": "Employee agrees not to compete for 2 years worldwide.",
            "strategy": "basic"
        })
        assert resp.status_code == 200
        assert resp.json()["strategy"] == "basic"

    def test_top_k_above_max_returns_422(self, client):
        resp = client.post("/analyze", json={
            "clause_text": "Employee agrees not to compete for 2 years worldwide.",
            "top_k": 11
        })
        assert resp.status_code == 422

    def test_top_k_zero_returns_422(self, client):
        resp = client.post("/analyze", json={
            "clause_text": "Employee agrees not to compete for 2 years worldwide.",
            "top_k": 0
        })
        assert resp.status_code == 422


# --- Search endpoint ---

class TestSearch:
    def test_valid_query_returns_200(self, client):
        resp = client.post("/search", json={"query": "non-compete clause"})
        assert resp.status_code == 200

    def test_response_has_required_fields(self, client):
        resp = client.post("/search", json={"query": "non-compete clause"})
        data = resp.json()
        assert "results" in data
        assert "query" in data
        assert "total_results" in data

    def test_query_echoed_in_response(self, client):
        resp = client.post("/search", json={"query": "non-compete clause"})
        assert resp.json()["query"] == "non-compete clause"

    def test_top_k_limits_results(self, client):
        resp = client.post("/search", json={"query": "non-compete clause", "top_k": 1})
        results = resp.json()["results"]
        assert len(results) <= 1

    def test_total_results_matches_results_length(self, client):
        resp = client.post("/search", json={"query": "non-compete clause"})
        data = resp.json()
        assert data["total_results"] == len(data["results"])

    def test_filters_applied(self, client):
        resp = client.post("/search", json={
            "query": "confidentiality",
            "top_k": 5,
            "filters": {"risk_level": "low"}
        })
        assert resp.status_code == 200
        for result in resp.json()["results"]:
            assert result["risk_level"] == "low"

    def test_empty_query_returns_422(self, client):
        resp = client.post("/search", json={"query": ""})
        assert resp.status_code == 422

    def test_too_short_query_returns_422(self, client):
        resp = client.post("/search", json={"query": "ab"})
        assert resp.status_code == 422

    def test_missing_query_returns_422(self, client):
        resp = client.post("/search", json={})
        assert resp.status_code == 422

    def test_top_k_above_max_returns_422(self, client):
        resp = client.post("/search", json={"query": "non-compete clause", "top_k": 21})
        assert resp.status_code == 422

    def test_result_fields_present(self, client):
        resp = client.post("/search", json={"query": "non-compete clause", "top_k": 3})
        for result in resp.json()["results"]:
            assert "id" in result
            assert "title" in result
            assert "text" in result
            assert "score" in result


# --- Auth tests ---

class TestAuth:
    def test_analyze_without_key_when_api_key_set_returns_401(self, client, monkeypatch):
        monkeypatch.setenv("API_KEY", "secret-key")
        resp = client.post("/analyze", json={
            "clause_text": "Employee agrees not to compete for 2 years worldwide."
        })
        assert resp.status_code == 401

    def test_analyze_with_wrong_key_returns_401(self, client, monkeypatch):
        monkeypatch.setenv("API_KEY", "secret-key")
        resp = client.post("/analyze", json={
            "clause_text": "Employee agrees not to compete for 2 years worldwide."
        }, headers={"X-API-Key": "wrong-key"})
        assert resp.status_code == 401

    def test_analyze_with_correct_key_returns_200(self, client, monkeypatch):
        monkeypatch.setenv("API_KEY", "secret-key")
        resp = client.post("/analyze", json={
            "clause_text": "Employee agrees not to compete for 2 years worldwide."
        }, headers={"X-API-Key": "secret-key"})
        assert resp.status_code == 200

    def test_analyze_without_api_key_env_allows_all(self, client, monkeypatch):
        monkeypatch.delenv("API_KEY", raising=False)
        resp = client.post("/analyze", json={
            "clause_text": "Employee agrees not to compete for 2 years worldwide."
        })
        assert resp.status_code == 200

    def test_search_without_key_when_api_key_set_returns_401(self, client, monkeypatch):
        monkeypatch.setenv("API_KEY", "secret-key")
        resp = client.post("/search", json={"query": "non-compete clause"})
        assert resp.status_code == 401

    def test_search_with_correct_key_returns_200(self, client, monkeypatch):
        monkeypatch.setenv("API_KEY", "secret-key")
        resp = client.post("/search", json={"query": "non-compete clause"},
                           headers={"X-API-Key": "secret-key"})
        assert resp.status_code == 200


# --- Error handling ---

class TestErrorHandling:
    def test_malformed_json_returns_422(self, client):
        resp = client.post(
            "/analyze",
            content=b"not valid json",
            headers={"Content-Type": "application/json"}
        )
        assert resp.status_code == 422

    def test_server_error_returns_500(self, loaded_faiss_db):
        """Mock provider.chat to raise, verify 500 is returned."""
        from src.api import app, get_db

        def override_get_db():
            return loaded_faiss_db

        app.dependency_overrides[get_db] = override_get_db

        try:
            # raise_server_exceptions=False so TestClient returns 500 instead of re-raising
            error_client = TestClient(app, raise_server_exceptions=False)
            with patch("src.rag_pipeline.generate_analysis", side_effect=RuntimeError("LLM unavailable")):
                resp = error_client.post("/analyze", json={
                    "clause_text": "Employee agrees not to compete for 2 years worldwide."
                })
            assert resp.status_code == 500
        finally:
            app.dependency_overrides.clear()
