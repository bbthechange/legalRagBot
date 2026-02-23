"""Tests for batched get_embeddings() in src/embeddings.py."""

import numpy as np
import pytest
from unittest.mock import MagicMock, call

from src.embeddings import get_embeddings


def _make_provider(embedding_dim=8):
    """Create a mock provider whose .embed() returns random embeddings."""
    provider = MagicMock()

    def embed_side_effect(texts):
        return np.random.rand(len(texts), embedding_dim).astype(np.float32)

    provider.embed.side_effect = embed_side_effect
    return provider


def test_small_batch_single_call():
    """get_embeddings() with 5 texts and batch_size=100 → single provider.embed call."""
    provider = _make_provider()
    texts = [f"text {i}" for i in range(5)]

    result = get_embeddings(texts, provider, batch_size=100)

    provider.embed.assert_called_once_with(texts)
    assert result.shape == (5, 8)


def test_large_batch_multiple_calls():
    """get_embeddings() with 250 texts and batch_size=100 → 3 provider.embed calls."""
    provider = _make_provider(embedding_dim=16)
    texts = [f"text {i}" for i in range(250)]

    result = get_embeddings(texts, provider, batch_size=100)

    assert provider.embed.call_count == 3
    # Verify batch sizes: 100, 100, 50
    call_args = provider.embed.call_args_list
    assert len(call_args[0][0][0]) == 100
    assert len(call_args[1][0][0]) == 100
    assert len(call_args[2][0][0]) == 50


def test_result_shape_matches_input():
    """Result shape is (n_texts, embedding_dim) regardless of batching."""
    provider = _make_provider(embedding_dim=32)
    texts = [f"clause text number {i}" for i in range(73)]

    result = get_embeddings(texts, provider, batch_size=25)

    assert result.shape == (73, 32)


def test_exact_batch_boundary():
    """get_embeddings() with exactly batch_size texts → single call."""
    provider = _make_provider()
    texts = [f"text {i}" for i in range(100)]

    result = get_embeddings(texts, provider, batch_size=100)

    provider.embed.assert_called_once()
    assert result.shape == (100, 8)


def test_single_text():
    """get_embeddings() with a single text returns shape (1, dim)."""
    provider = _make_provider(embedding_dim=4)
    texts = ["single text"]

    result = get_embeddings(texts, provider, batch_size=100)

    assert result.shape == (1, 4)
    provider.embed.assert_called_once()


def test_batched_output_is_concatenated():
    """
    Batched output matches non-batched output when using deterministic embeddings.
    """
    embedding_dim = 8
    texts = [f"text {i}" for i in range(10)]

    # Use fixed seed for determinism
    rng = np.random.default_rng(42)
    fixed_embeddings = rng.random((10, embedding_dim)).astype(np.float32)

    # Non-batched: returns all at once
    provider_single = MagicMock()
    provider_single.embed.return_value = fixed_embeddings
    result_single = get_embeddings(texts, provider_single, batch_size=100)

    # Batched: returns 2 chunks of 5
    provider_batched = MagicMock()
    provider_batched.embed.side_effect = [
        fixed_embeddings[:5],
        fixed_embeddings[5:],
    ]
    result_batched = get_embeddings(texts, provider_batched, batch_size=5)

    np.testing.assert_array_almost_equal(result_single, result_batched)


def test_batch_size_of_one():
    """batch_size=1 calls provider.embed once per text."""
    provider = _make_provider(embedding_dim=4)
    texts = [f"text {i}" for i in range(4)]

    result = get_embeddings(texts, provider, batch_size=1)

    assert provider.embed.call_count == 4
    assert result.shape == (4, 4)
