"""Tests for the knowledge_base_qa prompt strategy."""

from src.generation import build_knowledge_base_qa_prompt
from src.rag_pipeline import STRATEGIES


class TestBuildKnowledgeBaseQaPrompt:
    """Tests for build_knowledge_base_qa_prompt()."""

    def test_returns_messages_list(self):
        """Should return a list of message dicts with system and user roles."""
        messages = build_knowledge_base_qa_prompt(
            "What is a standard NDA?",
            "Retrieved context here",
        )

        assert isinstance(messages, list)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_system_prompt_contains_citation_instructions(self):
        """System prompt should instruct model to cite sources."""
        messages = build_knowledge_base_qa_prompt("test query", "test context")

        system_content = messages[0]["content"]
        assert "Cite" in system_content or "cite" in system_content
        assert "[" in system_content  # bracket citation format mentioned
        assert "sources_used" in system_content

    def test_user_message_contains_query(self):
        """User message should contain the query text."""
        query = "What is a reasonable non-compete duration?"
        messages = build_knowledge_base_qa_prompt(query, "context")

        assert messages[1]["content"] == query

    def test_system_prompt_contains_context(self):
        """System prompt should include the retrieved context."""
        context = "NDA clause text with risk level low"
        messages = build_knowledge_base_qa_prompt("query", context)

        assert context in messages[0]["content"]


class TestKnowledgeBaseQaStrategy:
    """Tests for the strategy registration."""

    def test_strategy_registered(self):
        """knowledge_base_qa should be registered in STRATEGIES dict."""
        assert "knowledge_base_qa" in STRATEGIES

    def test_strategy_callable(self):
        """The registered strategy should be callable."""
        assert callable(STRATEGIES["knowledge_base_qa"])

    def test_strategy_is_correct_function(self):
        """The registered strategy should be build_knowledge_base_qa_prompt."""
        assert STRATEGIES["knowledge_base_qa"] is build_knowledge_base_qa_prompt
