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
    "limitation_of_liability", "indemnification", "data_protection",
    "termination", "ip_ownership", "confidentiality", "governing_law",
    "warranty", "service_levels", "insurance", "non_compete",
    "non_solicitation", "assignment", "force_majeure", "notices",
    "entire_agreement", "amendment", "severability", "waiver",
    "representations", "payment_terms", "audit_rights",
]

# Pattern matches: "1.", "1.1", "1.1.1", "ARTICLE I", or ALL CAPS headings
_SECTION_PATTERN = re.compile(
    r"(?:^|\n)"
    r"(?:"
    r"\d+(?:\.\d+)*\.?\s+"        # numbered: 1. / 1.1 / 1.1.1
    r"|ARTICLE\s+[IVXLCDM\d]+\.?\s+"  # ARTICLE I / ARTICLE 1
    r"|[A-Z][A-Z\s]{2,}(?:\n|\.)"  # ALL CAPS heading (min 3 chars)
    r")"
)

MIN_CHUNK_LENGTH = 20


CLASSIFY_PROMPT = """You are a legal document analyst. Classify the following contract clause into one of these types:

{clause_types}

If the clause doesn't clearly fit any type, use "other".

Contract clause:
{clause_text}

Respond with ONLY a JSON object:
{{"clause_type": "the_type", "confidence": "high|medium|low"}}"""


def chunk_contract(text: str) -> list[dict]:
    """
    Split contract text into clause-level chunks.

    Returns list of {"text": str, "position": int, "heading": str | None}
    Fragments shorter than MIN_CHUNK_LENGTH are discarded.
    """
    splits = list(_SECTION_PATTERN.finditer(text))

    if not splits:
        stripped = text.strip()
        if len(stripped) < MIN_CHUNK_LENGTH:
            return []
        return [{"text": stripped, "position": 0, "heading": None}]

    chunks = []
    for i, match in enumerate(splits):
        start = match.start()
        end = splits[i + 1].start() if i + 1 < len(splits) else len(text)
        chunk_text = text[start:end].strip()

        if len(chunk_text) < MIN_CHUNK_LENGTH:
            continue

        heading = match.group().strip().rstrip(".")

        chunks.append({
            "text": chunk_text,
            "position": len(chunks),
            "heading": heading if heading else None,
        })

    return chunks


def classify_clause_type(clause_text: str, provider) -> dict:
    """
    Use an LLM to classify a clause's type.

    Returns {"clause_type": str, "confidence": str}
    """
    messages = [
        {"role": "system", "content": "You are a legal document classifier. Return only valid JSON."},
        {"role": "user", "content": CLASSIFY_PROMPT.format(
            clause_types=", ".join(KNOWN_CLAUSE_TYPES),
            clause_text=clause_text[:2000],
        )},
    ]

    try:
        raw = generate_analysis(messages, provider, temperature=0.0, max_tokens=100)
        parsed = parse_json_response_or_raw(raw)

        if isinstance(parsed, dict) and "clause_type" in parsed:
            return parsed
        return {"clause_type": "other", "confidence": "low"}
    except Exception:
        logger.warning("Clause classification failed, defaulting to 'other'")
        return {"clause_type": "other", "confidence": "low"}


def extract_clauses(contract_text: str, provider) -> list[dict]:
    """
    Full clause extraction pipeline: chunk -> classify.

    Returns list of classified clause chunks:
    [{"text": str, "clause_type": str, "confidence": str, "position": int, "heading": str|None}]
    """
    chunks = chunk_contract(contract_text)
    logger.info(f"Chunked contract into {len(chunks)} sections")

    classified = []
    for chunk in chunks:
        classification = classify_clause_type(chunk["text"], provider)
        classified.append({
            "text": chunk["text"],
            "clause_type": classification["clause_type"],
            "confidence": classification.get("confidence", "medium"),
            "position": chunk["position"],
            "heading": chunk["heading"],
        })

    logger.info(f"Classified {len(classified)} clauses: "
                f"{len([c for c in classified if c['clause_type'] != 'other'])} typed, "
                f"{len([c for c in classified if c['clause_type'] == 'other'])} other")
    return classified
