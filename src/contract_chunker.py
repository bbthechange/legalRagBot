"""
Contract Chunker — Split contracts into clauses and classify them.

Splits contract text on numbered sections and ALL CAPS headings,
then uses an LLM to classify each chunk by clause type.
"""

import re
import logging

from src.generation import generate_analysis
from src.output_parser import parse_json_response_or_raw

logger = logging.getLogger(__name__)

KNOWN_CLAUSE_TYPES = [
    "limitation_of_liability",
    "indemnification",
    "data_protection",
    "termination",
    "ip_ownership",
    "confidentiality",
    "confidentiality_scope",
    "confidentiality_exclusions",
    "confidentiality_duration",
    "governing_law",
    "warranty",
    "service_levels",
    "insurance",
    "non_solicitation",
    "permitted_disclosures",
    "return_or_destruction",
    "remedies",
    "force_majeure",
    "assignment",
    "notices",
    "entire_agreement",
    "amendments",
]

# Pattern matches: "1.", "1.1", "1.1.1", "ARTICLE I", or ALL CAPS headings on their own line
_SECTION_PATTERN = re.compile(
    r"(?:^|\n)"
    r"(?:"
    r"\d+(?:\.\d+)*\.?\s+"        # numbered: 1. / 1.1 / 1.1.1
    r"|ARTICLE\s+[IVXLCDM\d]+\.?\s+"  # ARTICLE I / ARTICLE 1
    r"|[A-Z][A-Z\s]{2,}(?:\n|\.)"  # ALL CAPS heading (min 3 chars)
    r")"
)

MIN_CHUNK_LENGTH = 20


def chunk_contract(text: str) -> list[dict]:
    """
    Split contract text into clause chunks.

    Returns a list of dicts with keys: text, position (0-based index), heading.
    Fragments shorter than MIN_CHUNK_LENGTH are discarded.
    """
    splits = list(_SECTION_PATTERN.finditer(text))

    if not splits:
        # No section markers found — return entire text as one chunk
        stripped = text.strip()
        if len(stripped) < MIN_CHUNK_LENGTH:
            return []
        return [{"text": stripped, "position": 0, "heading": ""}]

    chunks = []
    for i, match in enumerate(splits):
        start = match.start()
        end = splits[i + 1].start() if i + 1 < len(splits) else len(text)
        chunk_text = text[start:end].strip()

        if len(chunk_text) < MIN_CHUNK_LENGTH:
            continue

        # Extract heading from the match
        heading = match.group().strip().rstrip(".")

        chunks.append({
            "text": chunk_text,
            "position": len(chunks),
            "heading": heading,
        })

    return chunks


def classify_clause_type(clause_text: str, provider) -> dict:
    """
    Use an LLM to classify a clause chunk into a known clause type.

    Returns a dict with clause_type and confidence keys.
    """
    clause_list = ", ".join(KNOWN_CLAUSE_TYPES)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a legal contract analyst. Classify the following contract clause "
                "into one of these types: " + clause_list + ", or 'other' if none fit.\n\n"
                "Respond with JSON only: {\"clause_type\": \"...\", \"confidence\": \"high|medium|low\"}"
            ),
        },
        {"role": "user", "content": clause_text[:2000]},
    ]

    try:
        raw = generate_analysis(messages, provider, max_tokens=200)
        result = parse_json_response_or_raw(raw)
        if result.get("parse_error"):
            return {"clause_type": "other", "confidence": "low"}
        return {
            "clause_type": result.get("clause_type", "other"),
            "confidence": result.get("confidence", "low"),
        }
    except Exception:
        logger.warning("Clause classification failed, defaulting to 'other'")
        return {"clause_type": "other", "confidence": "low"}


def extract_clauses(contract_text: str, provider) -> list[dict]:
    """
    Full extraction pipeline: chunk the contract, then classify each chunk.

    Returns a list of dicts with keys: text, position, heading, clause_type, confidence.
    """
    chunks = chunk_contract(contract_text)
    results = []
    for chunk in chunks:
        classification = classify_clause_type(chunk["text"], provider)
        results.append({
            "text": chunk["text"],
            "position": chunk["position"],
            "heading": chunk["heading"],
            "clause_type": classification["clause_type"],
            "confidence": classification["confidence"],
        })
    return results
