"""Tests for CuadIngestor — only tests transform() with mock data (no HuggingFace download)."""

import pytest
from src.ingest.cuad import CuadIngestor, QUESTION_TO_CLAUSE_TYPE


def _make_row(question, answer_texts, title="ACME_Corp_Agreement.pdf"):
    return {
        "id": "cuad-test-001",
        "title": title,
        "context": "Full contract text...",
        "question": question,
        "answers": {
            "text": answer_texts,
            "answer_start": [0] * len(answer_texts),
        },
    }


GOVERNING_LAW_Q = 'Highlight the parts (if any) of this contract related to "Governing Law"'
INDEMNIFICATION_Q = 'Highlight the parts (if any) of this contract related to "Indemnification"'


class TestExtractClauseType:
    def test_known_question_exact_match(self):
        """Known question strings map to correct clause type labels."""
        ingestor = CuadIngestor()
        # Use a known key from QUESTION_TO_CLAUSE_TYPE
        for question, expected in list(QUESTION_TO_CLAUSE_TYPE.items())[:5]:
            result = ingestor._extract_clause_type(question)
            assert result == expected, f"Failed for question: {question[:50]}"

    def test_governing_law_mapping(self):
        """Governing law question maps correctly."""
        ingestor = CuadIngestor()
        result = ingestor._extract_clause_type(GOVERNING_LAW_Q)
        assert result == "governing_law"

    def test_indemnification_mapping(self):
        """Indemnification question maps correctly."""
        ingestor = CuadIngestor()
        result = ingestor._extract_clause_type(INDEMNIFICATION_Q)
        assert result == "indemnification"

    def test_unknown_question_extracts_quoted_portion(self):
        """Unknown question → extracts quoted portion as fallback."""
        ingestor = CuadIngestor()
        question = 'Highlight the parts related to "Force Majeure" in this contract.'
        result = ingestor._extract_clause_type(question)
        assert result == "force_majeure"

    def test_unknown_question_no_quotes_returns_unknown(self):
        """Question with no quotes and no match → 'unknown'."""
        ingestor = CuadIngestor()
        result = ingestor._extract_clause_type("This is an unrecognized question without quotes.")
        assert result == "unknown"

    def test_quoted_text_with_spaces_normalized(self):
        """Quoted text with spaces is normalized to snake_case."""
        ingestor = CuadIngestor()
        question = 'Highlight the parts related to "Limitation Of Liability" here.'
        result = ingestor._extract_clause_type(question)
        assert result == "limitation_of_liability"


class TestMakeDocId:
    def test_deterministic_same_inputs(self):
        """Same inputs always produce the same doc ID."""
        ingestor = CuadIngestor()
        id1 = ingestor._make_doc_id("Contract.pdf", "governing_law", "The laws of Delaware apply.")
        id2 = ingestor._make_doc_id("Contract.pdf", "governing_law", "The laws of Delaware apply.")
        assert id1 == id2

    def test_different_texts_produce_different_ids(self):
        """Different texts produce different IDs."""
        ingestor = CuadIngestor()
        id1 = ingestor._make_doc_id("Contract.pdf", "governing_law", "Delaware law applies here.")
        id2 = ingestor._make_doc_id("Contract.pdf", "governing_law", "California law applies here.")
        assert id1 != id2

    def test_different_clause_types_produce_different_ids(self):
        """Different clause types produce different IDs for same text."""
        ingestor = CuadIngestor()
        text = "This agreement shall be governed by the laws of the State of Delaware."
        id1 = ingestor._make_doc_id("Contract.pdf", "governing_law", text)
        id2 = ingestor._make_doc_id("Contract.pdf", "indemnification", text)
        assert id1 != id2

    def test_id_starts_with_cuad_prefix(self):
        """Generated IDs start with 'cuad-' prefix."""
        ingestor = CuadIngestor()
        doc_id = ingestor._make_doc_id("Contract.pdf", "governing_law", "Some text here.")
        assert doc_id.startswith("cuad-")


class TestTransform:
    def _make_ingestor(self, max_docs=None):
        return CuadIngestor(max_docs=max_docs)

    def test_transform_with_mock_data(self):
        """transform() with mock CUAD rows produces schema-conforming documents."""
        rows = [
            _make_row(
                GOVERNING_LAW_Q,
                ["This agreement shall be governed by the laws of the State of Delaware."],
            )
        ]
        ingestor = self._make_ingestor()
        result = ingestor.transform(rows)

        assert len(result) == 1
        doc = result[0]
        assert doc["source"] == "cuad"
        assert doc["doc_type"] == "clause"
        assert doc["doc_id"].startswith("cuad-")
        assert "governing_law" in doc["doc_id"].lower()
        assert doc["metadata"]["clause_type"] == "governing_law"
        assert "ACME_Corp_Agreement.pdf" in doc["title"]

    def test_transform_deduplication(self):
        """Duplicate answer texts produce only one document."""
        same_text = "This agreement shall be governed by the laws of Delaware."
        rows = [
            _make_row(GOVERNING_LAW_Q, [same_text]),
            _make_row(GOVERNING_LAW_Q, [same_text], title="Other_Contract.pdf"),
        ]
        ingestor = self._make_ingestor()
        result = ingestor.transform(rows)
        assert len(result) == 1

    def test_transform_skips_empty_answers(self):
        """Rows with empty answers.text produce no documents."""
        rows = [
            _make_row(GOVERNING_LAW_Q, []),
            _make_row(GOVERNING_LAW_Q, ["", "  "]),
        ]
        ingestor = self._make_ingestor()
        result = ingestor.transform(rows)
        assert result == []

    def test_transform_skips_very_short_texts(self):
        """Answer texts shorter than 10 characters are skipped."""
        rows = [_make_row(GOVERNING_LAW_Q, ["Short"])]  # < 10 chars
        ingestor = self._make_ingestor()
        result = ingestor.transform(rows)
        assert result == []

    def test_transform_max_docs_limit(self):
        """max_docs limits the number of output documents."""
        rows = [
            _make_row(
                GOVERNING_LAW_Q,
                [f"Clause text number {i} for the governing law section of this contract." for i in range(10)],
            )
        ]
        ingestor = self._make_ingestor(max_docs=5)
        result = ingestor.transform(rows)
        assert len(result) == 5

    def test_transform_multiple_answers_per_row(self):
        """Multiple answer texts in one row each become a separate document."""
        rows = [
            _make_row(
                INDEMNIFICATION_Q,
                [
                    "Each party shall indemnify and hold harmless the other party from any claims.",
                    "The indemnifying party shall defend at its own cost all claims arising from its breach.",
                ],
            )
        ]
        ingestor = self._make_ingestor()
        result = ingestor.transform(rows)
        assert len(result) == 2

    def test_transform_practice_area_ip(self):
        """IP-related clause types get 'intellectual_property' practice area."""
        rows = [
            _make_row(
                'Highlight the parts (if any) of this contract related to "License Grant"',
                ["Licensor grants Licensee a non-exclusive license to use the software."],
            )
        ]
        ingestor = self._make_ingestor()
        result = ingestor.transform(rows)
        assert len(result) == 1
        assert result[0]["metadata"]["practice_area"] == "intellectual_property"

    def test_transform_practice_area_commercial(self):
        """Commercial clause types get 'commercial_contracts' practice area."""
        rows = [_make_row(INDEMNIFICATION_Q, ["Party A shall indemnify Party B for all losses and damages arising."])]
        ingestor = self._make_ingestor()
        result = ingestor.transform(rows)
        assert len(result) == 1
        assert result[0]["metadata"]["practice_area"] == "commercial_contracts"

    def test_transform_empty_input(self):
        """Empty input returns empty list."""
        ingestor = self._make_ingestor()
        result = ingestor.transform([])
        assert result == []
