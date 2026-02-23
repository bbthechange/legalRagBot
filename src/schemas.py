"""
Unified Document Schema — Source-Agnostic Document Format

Defines the canonical schema all data sources must conform to before
entering the RAG pipeline. Provides validation and normalization helpers.
"""

REQUIRED_FIELDS = {"doc_id", "source", "doc_type", "title", "text"}

VALID_SOURCES = {
    "clauses_json",
    "cuad",
    "common_paper",
    "statutes",
    "opp115",
    "open_terms_archive",
    "legalbench",
}

VALID_DOC_TYPES = {
    "clause",
    "statute",
    "playbook",
    "privacy_policy",
    "terms_of_service",
}


def validate_document(doc: dict) -> list[str]:
    """
    Validate a document against the unified schema.

    Returns a list of error strings (empty list means valid).
    """
    errors = []

    for field in REQUIRED_FIELDS:
        if field not in doc:
            errors.append(f"Missing required field: {field}")
        elif not doc[field]:
            errors.append(f"Empty required field: {field}")

    if doc.get("source") and doc["source"] not in VALID_SOURCES:
        errors.append(f"Unknown source: {doc['source']}")

    if doc.get("doc_type") and doc["doc_type"] not in VALID_DOC_TYPES:
        errors.append(f"Unknown doc_type: {doc['doc_type']}")

    return errors


def normalize_document(doc: dict) -> dict:
    """
    Normalize a document into the canonical schema shape.

    Ensures the document has a nested `metadata` dict with all
    optional fields defaulted to None. Returns a new dict (does
    not mutate the input).
    """
    metadata_keys = [
        "clause_type", "category", "risk_level", "notes",
        "practice_area", "jurisdiction", "citation", "position",
    ]

    existing_metadata = doc.get("metadata", {})

    normalized_metadata = {}
    for key in metadata_keys:
        normalized_metadata[key] = existing_metadata.get(key, None)

    return {
        "doc_id": doc.get("doc_id"),
        "source": doc.get("source"),
        "doc_type": doc.get("doc_type"),
        "title": doc.get("title"),
        "text": doc.get("text"),
        "metadata": normalized_metadata,
    }
