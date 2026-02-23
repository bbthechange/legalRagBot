"""Tests for the unified knowledge base search pipeline."""

import json
from unittest.mock import patch

from src.kb_search import search_knowledge_base


class TestSearchKnowledgeBase:
    """Tests for search_knowledge_base()."""

    def test_full_pipeline_returns_expected_shape(self, loaded_multi_source_db):
        """Pipeline should return dict with answer, routing, sources, review_status, disclaimer."""
        db = loaded_multi_source_db
        # Mock both router and generation to return valid responses
        router_response = json.dumps({
            "query_type": "general_legal",
            "filters": {},
            "search_strategy": "semantic",
            "rewritten_query": None,
            "explanation": "General question",
        })
        kb_answer = json.dumps({
            "answer": "Based on [uni-001], the answer is...",
            "sources_used": [{"id": "uni-001", "title": "Test NDA Clause", "relevance": "Directly relevant"}],
            "confidence": "medium",
            "caveats": ["Limited sources"],
            "related_queries": ["What about mutual NDAs?"],
        })

        call_count = {"n": 0}
        original_chat = db["provider"].chat

        def mock_chat(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return router_response
            return kb_answer

        db["provider"].chat = mock_chat

        result = search_knowledge_base("What is a standard NDA?", db)

        assert "answer" in result
        assert "routing" in result
        assert "sources" in result
        assert result["review_status"] == "pending_review"
        assert "DRAFT" in result["disclaimer"]

    def test_with_router_disabled(self, loaded_multi_source_db):
        """use_router=False should skip routing and do pure semantic search."""
        db = loaded_multi_source_db
        kb_answer = json.dumps({
            "answer": "The answer is...",
            "sources_used": [],
            "confidence": "medium",
            "caveats": [],
            "related_queries": [],
        })
        db["provider"].chat = lambda *a, **kw: kb_answer

        result = search_knowledge_base("test query", db, use_router=False)

        assert result["routing"]["query_type"] == "general_legal"
        assert result["routing"]["search_strategy"] == "semantic"
        assert result["routing"]["filters"] == {}

    def test_empty_results_returns_no_info_answer(self, loaded_multi_source_db):
        """When no documents are found, answer should say 'I don't have enough information'."""
        db = loaded_multi_source_db

        with patch("src.kb_search.search_similar_clauses", return_value=[]):
            # Still need router to work
            router_response = json.dumps({
                "query_type": "general_legal",
                "filters": {},
                "search_strategy": "semantic",
                "rewritten_query": None,
                "explanation": "General",
            })
            db["provider"].chat = lambda *a, **kw: router_response

            result = search_knowledge_base("something obscure", db)

        assert "don't have enough information" in result["answer"]["answer"]
        assert result["sources"] == []
        assert result["answer"]["confidence"] == "low"

    def test_sources_match_retrieval(self, loaded_multi_source_db):
        """Sources in response should correspond to actual retrieved documents."""
        db = loaded_multi_source_db
        router_response = json.dumps({
            "query_type": "general_legal",
            "filters": {},
            "search_strategy": "semantic",
            "rewritten_query": None,
            "explanation": "General",
        })
        kb_answer = json.dumps({
            "answer": "Answer",
            "sources_used": [],
            "confidence": "medium",
            "caveats": [],
            "related_queries": [],
        })

        call_count = {"n": 0}

        def mock_chat(*args, **kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return router_response
            return kb_answer

        db["provider"].chat = mock_chat

        result = search_knowledge_base("confidential information", db, top_k=3)

        # Every source should have required fields
        for source in result["sources"]:
            assert "id" in source
            assert "title" in source
            assert "score" in source
            assert isinstance(source["score"], float)

    def test_draft_framing_present(self, loaded_multi_source_db):
        """review_status should be pending_review and disclaimer should mention DRAFT."""
        db = loaded_multi_source_db
        kb_answer = json.dumps({
            "answer": "Answer",
            "sources_used": [],
            "confidence": "medium",
            "caveats": [],
            "related_queries": [],
        })
        db["provider"].chat = lambda *a, **kw: kb_answer

        result = search_knowledge_base("test", db, use_router=False)

        assert result["review_status"] == "pending_review"
        assert "DRAFT" in result["disclaimer"]
