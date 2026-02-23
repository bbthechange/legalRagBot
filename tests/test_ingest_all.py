"""Tests for the ingest_all orchestrator module."""

import pytest
from src.ingest.ingest_all import register_ingestors


def test_register_ingestors_no_args_returns_all():
    """register_ingestors() with no args returns all available ingestors."""
    ingestors = register_ingestors()
    assert "clauses_json" in ingestors
    assert "cuad" in ingestors
    assert "common_paper" in ingestors
    assert len(ingestors) == 3


def test_register_ingestors_clauses_json_only():
    """register_ingestors(sources=['clauses_json']) returns only that ingestor."""
    ingestors = register_ingestors(sources=["clauses_json"])
    assert "clauses_json" in ingestors
    assert "cuad" not in ingestors
    assert len(ingestors) == 1


def test_register_ingestors_cuad_only():
    """register_ingestors(sources=['cuad']) returns only CUAD ingestor."""
    ingestors = register_ingestors(sources=["cuad"])
    assert "cuad" in ingestors
    assert "clauses_json" not in ingestors
    assert len(ingestors) == 1


def test_register_ingestors_unknown_source_returns_empty():
    """register_ingestors(sources=['unknown']) returns empty dict."""
    ingestors = register_ingestors(sources=["unknown_source"])
    assert ingestors == {}


def test_register_ingestors_factories_are_callable():
    """Registered ingestor values are callable factories."""
    ingestors = register_ingestors()
    for name, factory in ingestors.items():
        assert callable(factory), f"Ingestor '{name}' factory is not callable"


def test_register_ingestors_max_cuad_passed_through():
    """max_cuad parameter is forwarded to CuadIngestor."""
    ingestors = register_ingestors(sources=["cuad"], max_cuad=42)
    cuad_ingestor = ingestors["cuad"]()
    assert cuad_ingestor.max_docs == 42


def test_register_ingestors_multiple_known_sources():
    """register_ingestors with multiple valid sources returns all of them."""
    ingestors = register_ingestors(sources=["clauses_json", "cuad"])
    assert "clauses_json" in ingestors
    assert "cuad" in ingestors
    assert len(ingestors) == 2


def test_register_ingestors_common_paper_only():
    """register_ingestors(sources=['common_paper']) returns only playbook ingestor."""
    ingestors = register_ingestors(sources=["common_paper"])
    assert "common_paper" in ingestors
    assert len(ingestors) == 1
