"""Tests for the query router module."""

import json
from unittest.mock import patch

from src.query_router import route_query


class TestRouteQuery:
    """Tests for route_query()."""

    def test_contract_question_routes_to_contract_review(self, mock_provider):
        """Contract questions should route to contract_review type."""
        router_response = json.dumps({
            "query_type": "contract_review",
            "filters": {"source": "clauses_json", "doc_type": "clause"},
            "search_strategy": "hybrid",
            "rewritten_query": None,
            "explanation": "Contract clause question",
        })
        mock_provider.chat = lambda *a, **kw: router_response

        result = route_query("What is a standard non-compete duration?", mock_provider)

        assert result["query_type"] == "contract_review"
        assert result["filters"]["source"] == "clauses_json"

    def test_breach_question_routes_with_jurisdiction(self, mock_provider):
        """Breach questions with state should include jurisdiction filter."""
        router_response = json.dumps({
            "query_type": "breach_response",
            "filters": {
                "source": "statutes",
                "doc_type": "statute",
                "jurisdiction": "CA",
                "clause_type": None,
            },
            "search_strategy": "hybrid",
            "rewritten_query": "California data breach notification deadline requirements",
            "explanation": "Breach notification question about California",
        })
        mock_provider.chat = lambda *a, **kw: router_response

        result = route_query(
            "What is California's breach notification deadline?", mock_provider
        )

        assert result["query_type"] == "breach_response"
        assert result["filters"]["jurisdiction"] == "CA"
        # clause_type was None, should be removed
        assert "clause_type" not in result["filters"]

    def test_cross_cutting_question(self, mock_provider):
        """Ambiguous questions should route as cross_cutting."""
        router_response = json.dumps({
            "query_type": "cross_cutting",
            "filters": {},
            "search_strategy": "semantic",
            "rewritten_query": None,
            "explanation": "Spans multiple areas",
        })
        mock_provider.chat = lambda *a, **kw: router_response

        result = route_query(
            "Compare contract liability caps across industries", mock_provider
        )

        assert result["query_type"] == "cross_cutting"
        assert result["search_strategy"] == "semantic"

    def test_router_failure_falls_back_to_semantic(self, mock_provider):
        """When the router returns garbage, fall back to semantic search."""
        mock_provider.chat = lambda *a, **kw: "this is not valid json at all"

        result = route_query("Some question", mock_provider)

        assert result["query_type"] == "general_legal"
        assert result["search_strategy"] == "semantic"
        assert result["filters"] == {}

    def test_null_string_cleanup(self, mock_provider):
        """String "null" values in filters should be removed."""
        router_response = json.dumps({
            "query_type": "contract_review",
            "filters": {
                "source": "clauses_json",
                "doc_type": "null",
                "jurisdiction": "null",
                "clause_type": "null",
            },
            "search_strategy": "semantic",
            "rewritten_query": None,
            "explanation": "Test",
        })
        mock_provider.chat = lambda *a, **kw: router_response

        result = route_query("What is a standard NDA?", mock_provider)

        assert "doc_type" not in result["filters"]
        assert "jurisdiction" not in result["filters"]
        assert "clause_type" not in result["filters"]
        assert result["filters"]["source"] == "clauses_json"
