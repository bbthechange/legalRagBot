"""Tests for the contract chunker module."""

import json

import pytest

from src.contract_chunker import chunk_contract, classify_clause_type, extract_clauses


class TestChunkContract:
    """Tests for chunk_contract()."""

    def test_numbered_sections(self):
        """Splits on numbered sections like 1. and 1.1."""
        text = (
            "1. LIMITATION OF LIABILITY\n"
            "The total liability shall not exceed the fees paid.\n\n"
            "2. INDEMNIFICATION\n"
            "Vendor shall indemnify customer against IP claims.\n\n"
            "3. TERMINATION\n"
            "Either party may terminate with 30 days notice."
        )
        chunks = chunk_contract(text)
        assert len(chunks) == 3
        assert "LIMITATION OF LIABILITY" in chunks[0]["text"]
        assert "INDEMNIFICATION" in chunks[1]["text"]
        assert "TERMINATION" in chunks[2]["text"]

    def test_all_caps_headings(self):
        """Splits on ALL CAPS headings."""
        text = (
            "CONFIDENTIALITY\n"
            "All information shared shall remain confidential for five years.\n\n"
            "GOVERNING LAW\n"
            "This agreement is governed by the laws of Delaware."
        )
        chunks = chunk_contract(text)
        assert len(chunks) == 2
        assert "confidential" in chunks[0]["text"]
        assert "Delaware" in chunks[1]["text"]

    def test_no_sections_returns_single_chunk(self):
        """When no section markers found, returns entire text as one chunk."""
        text = "This is a contract clause with no section markers but enough text to be valid."
        chunks = chunk_contract(text)
        assert len(chunks) == 1
        assert chunks[0]["position"] == 0

    def test_tiny_fragments_skipped(self):
        """Fragments shorter than MIN_CHUNK_LENGTH are discarded."""
        text = (
            "1. SHORT\n"
            "tiny\n\n"
            "2. VALID SECTION\n"
            "This is a valid section with enough text to pass the minimum length check."
        )
        chunks = chunk_contract(text)
        # The first chunk "1. SHORT\ntiny" may be short, second should be kept
        for chunk in chunks:
            assert len(chunk["text"]) >= 20

    def test_chunks_have_required_keys(self):
        """Each chunk has text, position, and heading keys."""
        text = (
            "1. FIRST SECTION\n"
            "Content of the first section goes here.\n\n"
            "2. SECOND SECTION\n"
            "Content of the second section goes here."
        )
        chunks = chunk_contract(text)
        for chunk in chunks:
            assert "text" in chunk
            assert "position" in chunk
            assert "heading" in chunk


class TestClassifyClauseType:
    """Tests for classify_clause_type() with mock provider."""

    def test_classify_returns_clause_type(self, mock_provider, monkeypatch):
        """classify_clause_type returns a dict with clause_type and confidence."""
        def mock_chat(messages, model=None, temperature=0.2, max_tokens=200):
            return json.dumps({"clause_type": "indemnification", "confidence": "high"})

        monkeypatch.setattr(mock_provider, "chat", mock_chat)
        result = classify_clause_type("Vendor shall indemnify customer.", mock_provider)
        assert result["clause_type"] == "indemnification"
        assert result["confidence"] == "high"


class TestExtractClauses:
    """Tests for extract_clauses() pipeline."""

    def test_extract_returns_all_keys(self, mock_provider, monkeypatch):
        """extract_clauses returns dicts with text, position, heading, clause_type, confidence."""
        def mock_chat(messages, model=None, temperature=0.2, max_tokens=200):
            return json.dumps({"clause_type": "termination", "confidence": "medium"})

        monkeypatch.setattr(mock_provider, "chat", mock_chat)

        text = (
            "1. TERMINATION\n"
            "Either party may terminate this agreement on 30 days written notice."
        )
        results = extract_clauses(text, mock_provider)
        assert len(results) >= 1
        for r in results:
            assert "text" in r
            assert "position" in r
            assert "heading" in r
            assert "clause_type" in r
            assert "confidence" in r
