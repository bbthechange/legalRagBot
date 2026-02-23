"""Tests for src/generation.py"""

from unittest.mock import MagicMock

from src.generation import (
    build_basic_prompt,
    build_structured_prompt,
    build_few_shot_prompt,
    generate_analysis,
)


class TestBuildBasicPrompt:
    def test_returns_two_messages(self):
        messages = build_basic_prompt("test clause", "test context")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "test clause" in messages[1]["content"]
        assert "test context" in messages[1]["content"]


class TestBuildStructuredPrompt:
    def test_returns_two_messages_with_json(self):
        messages = build_structured_prompt("test clause", "test context")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "JSON" in messages[0]["content"]
        assert "test clause" in messages[1]["content"]
        assert "test context" in messages[1]["content"]


class TestBuildFewShotPrompt:
    def test_returns_four_messages(self):
        messages = build_few_shot_prompt("test clause", "test context")
        assert len(messages) == 4
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"
        assert messages[3]["role"] == "user"
        assert "test clause" in messages[3]["content"]
        assert "test context" in messages[3]["content"]


class TestGenerateAnalysis:
    def test_delegates_to_provider_chat(self, mock_provider):
        messages = [{"role": "user", "content": "test"}]
        result = generate_analysis(messages, mock_provider)
        assert isinstance(result, str)

    def test_passes_parameters(self):
        provider = MagicMock()
        provider.chat.return_value = "analysis result"

        messages = [{"role": "user", "content": "test"}]
        result = generate_analysis(
            messages, provider, model="custom-model",
            temperature=0.5, max_tokens=500
        )

        provider.chat.assert_called_once_with(
            messages, model="custom-model", temperature=0.5, max_tokens=500
        )
        assert result == "analysis result"
