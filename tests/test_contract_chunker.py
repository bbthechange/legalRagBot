"""Tests for the contract chunker module."""

import json

import pytest

from src.contract_chunker import (
    chunk_contract, classify_clause_type, extract_clauses,
    _normalize_text, MIN_CHUNK_LENGTH, MAX_CHUNK_LENGTH,
)


class TestNormalizeText:
    """Tests for _normalize_text()."""

    def test_windows_line_endings(self):
        """Converts \\r\\n to \\n."""
        assert _normalize_text("line1\r\nline2\r\n") == "line1\nline2\n"

    def test_non_breaking_spaces(self):
        """Converts non-breaking spaces (\\u00a0) to regular spaces."""
        assert _normalize_text("hello\u00a0world") == "hello world"

    def test_smart_quotes(self):
        """Converts smart quotes to ASCII equivalents."""
        text = "\u201cHello\u201d and \u2018world\u2019"
        assert _normalize_text(text) == '"Hello" and \'world\''

    def test_consecutive_blank_lines_collapsed(self):
        """Collapses 3+ consecutive newlines to 2."""
        text = "para1\n\n\n\npara2\n\n\n\n\npara3"
        result = _normalize_text(text)
        assert "\n\n\n" not in result
        assert "para1\n\npara2\n\npara3" == result


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
        assert chunks[0]["heading"] is None

    def test_tiny_fragments_skipped(self):
        """Fragments shorter than MIN_CHUNK_LENGTH are discarded."""
        text = (
            "1. SHORT\n"
            "tiny\n\n"
            "2. VALID SECTION\n"
            "This is a valid section with enough text to pass the minimum length check."
        )
        chunks = chunk_contract(text)
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

    # --- New tests below ---

    def test_single_line_no_newlines(self):
        """The reported bug: single-line text with no newlines finds sections."""
        text = (
            "1. LIMITATION OF LIABILITY  The total liability shall not exceed the fees paid.  "
            "2. INDEMNIFICATION  Vendor shall indemnify customer against all IP claims.  "
            "3. TERMINATION  Either party may terminate with 30 days written notice."
        )
        chunks = chunk_contract(text)
        assert len(chunks) == 3
        assert "liability" in chunks[0]["text"].lower()

    def test_windows_crlf_line_endings(self):
        """Windows \\r\\n line endings produce correct section count."""
        text = (
            "1. LIMITATION OF LIABILITY\r\n"
            "The total liability shall not exceed the fees paid.\r\n\r\n"
            "2. INDEMNIFICATION\r\n"
            "Vendor shall indemnify customer against IP claims.\r\n\r\n"
            "3. TERMINATION\r\n"
            "Either party may terminate with 30 days notice."
        )
        chunks = chunk_contract(text)
        assert len(chunks) == 3

    def test_non_breaking_spaces_in_markers(self):
        """Non-breaking spaces in section markers are handled."""
        # \u00a0 between "1." and text
        text = (
            "1.\u00a0LIMITATION OF LIABILITY\n"
            "The total liability shall not exceed the fees paid.\n\n"
            "2.\u00a0INDEMNIFICATION\n"
            "Vendor shall indemnify customer against IP claims."
        )
        chunks = chunk_contract(text)
        assert len(chunks) == 2

    def test_preamble_captured(self):
        """Text before first section marker is captured as PREAMBLE."""
        text = (
            "This Agreement is entered into between Party A and Party B on January 1, 2025.\n\n"
            "1. LIMITATION OF LIABILITY\n"
            "The total liability shall not exceed the fees paid.\n\n"
            "2. TERMINATION\n"
            "Either party may terminate with 30 days notice."
        )
        chunks = chunk_contract(text)
        assert chunks[0]["heading"] == "PREAMBLE"
        assert "Party A" in chunks[0]["text"]
        assert len(chunks) == 3

    def test_all_caps_abbreviations_not_matched(self):
        """Short ALL CAPS like LLC, USA are NOT matched as headings."""
        text = (
            "1. PARTIES\n"
            "This agreement is between ACME LLC and WIDGET USA for services.\n\n"
            "2. TERMINATION\n"
            "Either party may terminate with 30 days notice."
        )
        chunks = chunk_contract(text)
        # Should get 2 chunks (the numbered sections), not extra from LLC/USA
        assert len(chunks) == 2

    def test_all_caps_company_name_inline_not_matched(self):
        """ALL CAPS company names inline are not false-matched as headings."""
        text = (
            "1. PARTIES\n"
            "ACME CORP. is the service provider.\n"
            "WIDGET LLC. is the customer.\n\n"
            "2. TERMINATION\n"
            "Either party may terminate with 30 days notice."
        )
        chunks = chunk_contract(text)
        # Should be 2 numbered sections, not extra splits on company names
        assert len(chunks) == 2

    def test_all_caps_company_name_own_line_not_matched(self):
        """ALL CAPS company names alone on their own line are not false-matched."""
        text = (
            "1. PARTIES\n"
            "The following entities are parties to this agreement:\n\n"
            "ACME CORP.\n"
            "New York, NY\n\n"
            "2. TERMINATION\n"
            "Either party may terminate with 30 days notice."
        )
        chunks = chunk_contract(text)
        # Should be 2 numbered sections, not extra from "ACME CORP."
        assert len(chunks) == 2

    def test_all_caps_real_headings_detected(self):
        """Real ALL CAPS headings at line boundaries are still detected."""
        text = (
            "CONFIDENTIALITY\n"
            "All information shared shall remain confidential for five years.\n\n"
            "GOVERNING LAW\n"
            "This agreement is governed by the laws of Delaware.\n\n"
            "TERMINATION\n"
            "Either party may terminate with 30 days written notice."
        )
        chunks = chunk_contract(text)
        assert len(chunks) == 3

    def test_large_chunk_split(self):
        """Chunks exceeding MAX_CHUNK_LENGTH are split with (cont.) heading."""
        # Build a chunk that exceeds 3000 chars
        sentence = "The parties agree to the terms described herein. "
        large_section = sentence * 80  # ~3920 chars
        text = (
            "1. LARGE SECTION\n"
            f"{large_section}\n\n"
            "2. SMALL SECTION\n"
            "This is a normal-sized section with enough text to be valid."
        )
        chunks = chunk_contract(text)
        # The large section should be split into 2+ chunks
        assert len(chunks) >= 3
        # Check for (cont.) heading
        cont_chunks = [c for c in chunks if c["heading"] and "(cont.)" in c["heading"]]
        assert len(cont_chunks) >= 1

    def test_section_clause_patterns(self):
        """Section X and Clause X patterns are matched."""
        text = (
            "Section 1 Definitions and Interpretation\n"
            "The following terms shall have the meanings ascribed below.\n\n"
            "Section 2 Term and Termination\n"
            "This agreement shall commence on the effective date.\n\n"
            "Clause 3 Confidentiality Obligations\n"
            "Each party shall keep information confidential."
        )
        chunks = chunk_contract(text)
        assert len(chunks) == 3

    def test_whereas_now_therefore(self):
        """WHEREAS and NOW THEREFORE recital markers are matched."""
        text = (
            "WHEREAS Company A desires to engage the services of Company B;\n\n"
            "WHEREAS Company B has the expertise and resources to provide such services;\n\n"
            "NOW, THEREFORE, the parties agree as follows:\n\n"
            "1. SERVICES\n"
            "Company B shall provide consulting services."
        )
        chunks = chunk_contract(text)
        # Should find WHEREAS (x2), NOW THEREFORE, and 1. SERVICES
        assert len(chunks) == 4

    def test_positions_sequential(self):
        """Positions are 0-indexed and sequential after all processing."""
        text = (
            "Preamble text that is long enough to meet the minimum length.\n\n"
            "1. FIRST\n"
            "Content of the first section with sufficient length.\n\n"
            "2. SECOND\n"
            "Content of the second section with sufficient length.\n\n"
            "3. THIRD\n"
            "Content of the third section with sufficient length."
        )
        chunks = chunk_contract(text)
        for i, chunk in enumerate(chunks):
            assert chunk["position"] == i


class TestClassifyClauseType:
    """Tests for classify_clause_type() with mock provider."""

    def test_classify_returns_clause_type(self, mock_provider, monkeypatch):
        """classify_clause_type returns a dict with clause_type and confidence."""
        def mock_chat(messages, model=None, temperature=0.0, max_tokens=100):
            return json.dumps({"clause_type": "indemnification", "confidence": "high"})

        monkeypatch.setattr(mock_provider, "chat", mock_chat)
        result = classify_clause_type("Vendor shall indemnify customer.", mock_provider)
        assert result["clause_type"] == "indemnification"
        assert result["confidence"] == "high"


class TestExtractClauses:
    """Tests for extract_clauses() pipeline."""

    def test_extract_returns_all_keys(self, mock_provider, monkeypatch):
        """extract_clauses returns dicts with text, position, heading, clause_type, confidence."""
        def mock_chat(messages, model=None, temperature=0.0, max_tokens=100):
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
