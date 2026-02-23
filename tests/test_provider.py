"""Tests for src/provider.py"""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.provider import create_provider, OpenAIProvider, AzureOpenAIProvider, BedrockProvider


class TestCreateProvider:
    def test_create_openai_provider(self):
        """create_provider with LLM_PROVIDER=openai mocks OpenAI client."""
        mock_openai_module = MagicMock()
        mock_client = MagicMock()
        mock_openai_module.OpenAI.return_value = mock_client

        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            with patch.dict(os.environ, {"LLM_PROVIDER": "openai"}):
                provider = create_provider()
                assert provider.provider_name == "OpenAI"
                assert provider.client == mock_client

    def test_unknown_provider_raises(self):
        """Unknown provider name raises ValueError."""
        with patch.dict(os.environ, {"LLM_PROVIDER": "unknown_provider"}):
            with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
                create_provider()

    def test_default_provider_is_openai(self):
        """No LLM_PROVIDER env var defaults to openai."""
        mock_openai_module = MagicMock()
        mock_openai_module.OpenAI.return_value = MagicMock()

        env = os.environ.copy()
        env.pop("LLM_PROVIDER", None)
        with patch.dict("sys.modules", {"openai": mock_openai_module}):
            with patch.dict(os.environ, env, clear=True):
                provider = create_provider()
                assert provider.provider_name == "OpenAI"

    def test_azure_missing_env_vars_raises(self):
        """Azure provider without endpoint/key raises ValueError."""
        mock_openai_module = MagicMock()

        with patch.dict("sys.modules", {"openai": mock_openai_module}), \
             patch.dict(os.environ, {
                 "LLM_PROVIDER": "azure",
                 "AZURE_OPENAI_ENDPOINT": "",
                 "AZURE_OPENAI_API_KEY": "",
             }):
            with pytest.raises(ValueError, match="AZURE_OPENAI_ENDPOINT"):
                create_provider()

    def test_azure_missing_deployment_names_raises(self):
        """Azure provider without deployment names raises ValueError."""
        mock_openai_module = MagicMock()
        mock_openai_module.AzureOpenAI.return_value = MagicMock()

        with patch.dict("sys.modules", {"openai": mock_openai_module}), \
             patch.dict(os.environ, {
                 "LLM_PROVIDER": "azure",
                 "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
                 "AZURE_OPENAI_API_KEY": "test-key",
                 "AZURE_EMBEDDING_DEPLOYMENT": "",
                 "AZURE_CHAT_DEPLOYMENT": "",
             }):
            with pytest.raises(ValueError, match="AZURE_EMBEDDING_DEPLOYMENT"):
                create_provider()

    def test_interface_contract(self):
        """All provider classes have embed, chat, provider_name, embedding_model, chat_model."""
        for cls in [OpenAIProvider, AzureOpenAIProvider, BedrockProvider]:
            assert callable(getattr(cls, "embed", None)), f"{cls.__name__} missing embed"
            assert callable(getattr(cls, "chat", None)), f"{cls.__name__} missing chat"
            assert hasattr(cls, "provider_name"), f"{cls.__name__} missing provider_name"

        # OpenAI has class-level model attributes; Azure/Bedrock set them in __init__
        assert hasattr(OpenAIProvider, "embedding_model")
        assert hasattr(OpenAIProvider, "chat_model")
