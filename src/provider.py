"""
LLM Provider Abstraction — Multi-Cloud Support

Supports OpenAI (direct), Azure OpenAI, and AWS Bedrock as interchangeable
LLM backends. Each provider exposes the same two methods: embed() and chat().

Selection via LLM_PROVIDER env var (default: "openai").
"""

import logging
import os
import json
import numpy as np

from src.retry import retry_with_backoff

logger = logging.getLogger(__name__)


class OpenAIProvider:
    """Direct OpenAI API provider."""

    provider_name = "OpenAI"
    embedding_model = "text-embedding-3-small"
    chat_model = "gpt-4o-mini"

    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI()
        logger.info("Initialized OpenAI provider")

    # TODO: narrow to specific exception types (429, 500+)
    @retry_with_backoff(max_retries=3, base_delay=1.0, retryable_exceptions=(Exception,))
    def embed(self, texts: list[str]) -> np.ndarray:
        logger.debug("Embedding %d texts via OpenAI", len(texts))
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings, dtype="float32")

    @retry_with_backoff(max_retries=3, base_delay=1.0, retryable_exceptions=(Exception,))
    def chat(self, messages: list[dict], model: str | None = None,
             temperature: float = 0.2, max_tokens: int = 1500) -> str:
        logger.debug("Chat request to OpenAI model=%s", model or self.chat_model)
        response = self.client.chat.completions.create(
            model=model or self.chat_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


class AzureOpenAIProvider:
    """Azure OpenAI Service provider."""

    provider_name = "Azure OpenAI"

    def __init__(self):
        from openai import AzureOpenAI

        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")

        if not endpoint or not api_key:
            raise ValueError(
                "Azure OpenAI requires AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY env vars"
            )

        self.client = AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )
        self.embedding_model = os.environ.get("AZURE_EMBEDDING_DEPLOYMENT")
        self.chat_model = os.environ.get("AZURE_CHAT_DEPLOYMENT")

        if not self.embedding_model or not self.chat_model:
            raise ValueError(
                "Azure OpenAI requires AZURE_EMBEDDING_DEPLOYMENT and AZURE_CHAT_DEPLOYMENT env vars"
            )
        logger.info("Initialized Azure OpenAI provider (endpoint=%s)", endpoint)

    @retry_with_backoff(max_retries=3, base_delay=1.0, retryable_exceptions=(Exception,))
    def embed(self, texts: list[str]) -> np.ndarray:
        logger.debug("Embedding %d texts via Azure OpenAI", len(texts))
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=texts,
        )
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings, dtype="float32")

    @retry_with_backoff(max_retries=3, base_delay=1.0, retryable_exceptions=(Exception,))
    def chat(self, messages: list[dict], model: str | None = None,
             temperature: float = 0.2, max_tokens: int = 1500) -> str:
        logger.debug("Chat request to Azure OpenAI model=%s", model or self.chat_model)
        response = self.client.chat.completions.create(
            model=model or self.chat_model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


class BedrockProvider:
    """AWS Bedrock provider using Titan for embeddings and Converse API for chat."""

    provider_name = "AWS Bedrock"

    def __init__(self):
        import boto3

        region = os.environ.get("AWS_REGION", "us-east-1")
        self.bedrock = boto3.client("bedrock-runtime", region_name=region)
        self.embedding_model = os.environ.get(
            "BEDROCK_EMBEDDING_MODEL", "amazon.titan-embed-text-v2:0"
        )
        self.chat_model = os.environ.get(
            "BEDROCK_CHAT_MODEL", "anthropic.claude-3-haiku-20240307-v1:0"
        )
        logger.info("Initialized Bedrock provider (region=%s)", region)

    @retry_with_backoff(max_retries=3, base_delay=1.0, retryable_exceptions=(Exception,))
    def embed(self, texts: list[str]) -> np.ndarray:
        logger.debug("Embedding %d texts via Bedrock", len(texts))
        embeddings = []
        for text in texts:
            response = self.bedrock.invoke_model(
                modelId=self.embedding_model,
                body=json.dumps({"inputText": text}),
                contentType="application/json",
            )
            result = json.loads(response["body"].read())
            embeddings.append(result["embedding"])
        return np.array(embeddings, dtype="float32")

    @retry_with_backoff(max_retries=3, base_delay=1.0, retryable_exceptions=(Exception,))
    def chat(self, messages: list[dict], model: str | None = None,
             temperature: float = 0.2, max_tokens: int = 1500) -> str:
        logger.debug("Chat request to Bedrock model=%s", model or self.chat_model)
        system_parts = []
        converse_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_parts.append({"text": msg["content"]})
            else:
                converse_messages.append({
                    "role": msg["role"],
                    "content": [{"text": msg["content"]}],
                })

        kwargs = {
            "modelId": model or self.chat_model,
            "messages": converse_messages,
            "inferenceConfig": {
                "temperature": temperature,
                "maxTokens": max_tokens,
            },
        }
        if system_parts:
            kwargs["system"] = system_parts

        response = self.bedrock.converse(**kwargs)
        return response["output"]["message"]["content"][0]["text"]


_PROVIDERS = {
    "openai": OpenAIProvider,
    "azure": AzureOpenAIProvider,
    "bedrock": BedrockProvider,
}


def create_provider():
    """
    Create an LLM provider based on the LLM_PROVIDER env var.
    Defaults to "openai" if not set.
    """
    name = os.environ.get("LLM_PROVIDER", "openai").lower()
    if name not in _PROVIDERS:
        available = ", ".join(_PROVIDERS)
        raise ValueError(
            f"Unknown LLM_PROVIDER '{name}'. Choose from: {available}"
        )
    return _PROVIDERS[name]()
