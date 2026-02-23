"""Tests for the /breach-analysis API endpoint."""

import json

import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient


@pytest.fixture
def client(loaded_multi_source_db):
    """
    Create a test client with a pre-loaded mock database.
    Patches get_db() to return the mock DB.
    """
    from src.api import app, get_db

    def override_get_db():
        return loaded_multi_source_db

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


class TestBreachAnalysisEndpoint:
    def test_valid_request_returns_200(self, client):
        """POST /breach-analysis with valid params returns 200."""
        resp = client.post("/breach-analysis", json={
            "data_types_compromised": ["ssn"],
            "affected_states": ["CA"],
        })
        assert resp.status_code == 200

    def test_response_has_required_fields(self, client):
        """Response includes breach_params, summary, state_analyses, disclaimer."""
        resp = client.post("/breach-analysis", json={
            "data_types_compromised": ["ssn"],
            "affected_states": ["CA"],
        })
        data = resp.json()
        assert "breach_params" in data
        assert "summary" in data
        assert "state_analyses" in data
        assert "disclaimer" in data
        assert "review_status" in data

    def test_review_status_is_pending(self, client):
        """Response has review_status='pending_review'."""
        resp = client.post("/breach-analysis", json={
            "data_types_compromised": ["ssn"],
            "affected_states": ["CA"],
        })
        assert resp.json()["review_status"] == "pending_review"

    def test_disclaimer_contains_draft(self, client):
        """Disclaimer contains 'DRAFT'."""
        resp = client.post("/breach-analysis", json={
            "data_types_compromised": ["ssn"],
            "affected_states": ["CA"],
        })
        assert "DRAFT" in resp.json()["disclaimer"]

    def test_empty_states_returns_422(self, client):
        """Empty affected_states returns 422."""
        resp = client.post("/breach-analysis", json={
            "data_types_compromised": ["ssn"],
            "affected_states": [],
        })
        assert resp.status_code == 422

    def test_empty_data_types_returns_422(self, client):
        """Empty data_types_compromised returns 422."""
        resp = client.post("/breach-analysis", json={
            "data_types_compromised": [],
            "affected_states": ["CA"],
        })
        assert resp.status_code == 422

    def test_missing_required_fields_returns_422(self, client):
        """Missing required fields returns 422."""
        resp = client.post("/breach-analysis", json={})
        assert resp.status_code == 422

    def test_summary_has_total_jurisdictions(self, client):
        """Summary includes total_jurisdictions count."""
        resp = client.post("/breach-analysis", json={
            "data_types_compromised": ["ssn"],
            "affected_states": ["CA", "NY"],
        })
        data = resp.json()
        assert data["summary"]["total_jurisdictions"] >= 1

    def test_optional_fields_have_defaults(self, client):
        """Optional fields use defaults when not provided."""
        resp = client.post("/breach-analysis", json={
            "data_types_compromised": ["ssn"],
            "affected_states": ["CA"],
        })
        data = resp.json()
        params = data["breach_params"]
        assert params["encryption_status"] == "unknown"
        assert params["entity_type"] == "for_profit"
