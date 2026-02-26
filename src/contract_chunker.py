"""
Contract Chunker — Split contracts into clauses and classify them.

Splits contract text on numbered sections and ALL CAPS headings,
then uses an LLM to classify each chunk by clause type.
"""

import re
import logging
from concurrent.futures import ThreadPoolExecutor

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

# Section pattern with two groups:
# Group A: Distinctive markers (numbered, ARTICLE, Section, WHEREAS, etc.)
#   — can match after 2+ spaces (handles collapsed newlines from copy-paste)
# Group B: ALL CAPS headings
#   — require strict line boundary to prevent false-matching company names
_SECTION_PATTERN = re.compile(
    r"(?:"
    # --- Group A: Distinctive markers — can match after 2+ spaces ---
    r"(?:^|\n|  +)(?:"
        r"\d+(?:\.\d+)*\.?\s+"                                      # 1. / 1.1 / 1.1.1
        r"|ARTICLE\s+[IVXLCDM\d]+\.?\s+"                            # ARTICLE I / ARTICLE 1
        r"|(?:Section|SECTION|Clause|CLAUSE)\s+\d+(?:\.\d+)*\.?\s+"  # Section 1 / Clause 1.2
        r"|(?:SCHEDULE|EXHIBIT|APPENDIX)\s+[A-Z\d]+\.?\s+"          # SCHEDULE A / EXHIBIT 1
        r"|(?:WHEREAS|NOW,?\s+THEREFORE)[,:]?\s+"                   # Recital markers
    r")"
    r"|"
    # --- Group B: ALL CAPS headings — require blank line or start of text ---
    # Heading must be alone on its line without trailing period (rejects "ACME CORP.")
    r"(?:^|\n\n)(?:[A-Z]{4,}|[A-Z]+\s+[A-Z]+)(?:\s+[A-Z]+)*(?=\s*\n|\s*$)"
    r")"
)

MIN_CHUNK_LENGTH = 20
MAX_CHUNK_LENGTH = 3000
CHUNK_OVERLAP = 200

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


CLASSIFY_PROMPT = """You are a legal document analyst. Classify the following contract clause into one of these types:

{clause_types}

If the clause doesn't clearly fit any type, use "other".

Contract clause:
{clause_text}

Respond with ONLY a JSON object:
{{"clause_type": "the_type", "confidence": "high|medium|low"}}"""


def _normalize_text(text: str) -> str:
    """Normalize whitespace, line endings, and typographic characters."""
    # Windows and bare carriage returns
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Non-breaking spaces (Word/PDF)
    text = text.replace("\u00a0", " ")
    # Smart quotes to ASCII
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    # Em/en dashes to ASCII
    text = text.replace("\u2014", "-").replace("\u2013", "-")
    # Collapse 3+ consecutive newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _extract_heading(match_text: str) -> str | None:
    """Clean up a regex match into a heading string."""
    heading = match_text.strip().rstrip(".")
    heading = re.sub(r"\s+", " ", heading)
    return heading if heading else None


def _hard_split(text: str, heading: str | None) -> list[dict]:
    """Split text at nearest whitespace when no sentence boundaries exist."""
    result = []
    start = 0
    while start < len(text):
        end = start + MAX_CHUNK_LENGTH
        if end >= len(text):
            result.append(text[start:])
            break
        # Find nearest whitespace before the limit
        split_at = text.rfind(" ", start, end)
        if split_at <= start:
            split_at = end  # no whitespace found, hard cut
        result.append(text[start:split_at])
        start = split_at + 1

    chunks = []
    for i, sub_text in enumerate(result):
        sub_heading = heading
        if i > 0 and sub_heading:
            sub_heading = sub_heading + " (cont.)"
        elif i > 0:
            sub_heading = "(cont.)"
        chunks.append({"text": sub_text.strip(), "position": 0, "heading": sub_heading})
    return chunks


def _split_large_chunk(chunk: dict) -> list[dict]:
    """Split an oversized chunk at sentence boundaries with overlap."""
    text = chunk["text"]
    if len(text) <= MAX_CHUNK_LENGTH:
        return [chunk]

    sentences = _SENTENCE_BOUNDARY.split(text)

    # Fallback: if no sentence boundaries, hard-split at nearest whitespace
    if len(sentences) <= 1:
        return _hard_split(text, chunk["heading"])

    sub_chunks = []
    current = ""
    heading = chunk["heading"]

    for sentence in sentences:
        if current and len(current) + len(sentence) + 1 > MAX_CHUNK_LENGTH:
            sub_chunks.append(current)
            # Start next sub-chunk with overlap; truncate if overlap + sentence exceeds max
            overlap = current[-CHUNK_OVERLAP:] if len(current) > CHUNK_OVERLAP else current
            if len(overlap) + 1 + len(sentence) > MAX_CHUNK_LENGTH:
                overlap = ""
            current = (overlap + " " + sentence).lstrip() if overlap else sentence
        else:
            current = (current + " " + sentence).lstrip() if current else sentence

    if current:
        sub_chunks.append(current)

    result = []
    for i, sub_text in enumerate(sub_chunks):
        sub_heading = heading
        if i > 0 and sub_heading:
            sub_heading = sub_heading + " (cont.)"
        elif i > 0:
            sub_heading = "(cont.)"
        result.append({
            "text": sub_text,
            "position": 0,  # reassigned later
            "heading": sub_heading,
        })

    return result


def chunk_contract(text: str) -> list[dict]:
    """
    Split contract text into clause-level chunks.

    Returns list of {"text": str, "position": int, "heading": str | None}
    Fragments shorter than MIN_CHUNK_LENGTH are discarded.
    """
    text = _normalize_text(text)
    splits = list(_SECTION_PATTERN.finditer(text))

    if not splits:
        stripped = text.strip()
        if len(stripped) < MIN_CHUNK_LENGTH:
            return []
        return [{"text": stripped, "position": 0, "heading": None}]

    chunks = []

    # Capture preamble (text before first section marker)
    if splits[0].start() > 0:
        preamble = text[:splits[0].start()].strip()
        if len(preamble) >= MIN_CHUNK_LENGTH:
            chunks.append({"text": preamble, "position": 0, "heading": "PREAMBLE"})

    for i, match in enumerate(splits):
        start = match.start()
        end = splits[i + 1].start() if i + 1 < len(splits) else len(text)
        chunk_text = text[start:end].strip()

        if len(chunk_text) < MIN_CHUNK_LENGTH:
            continue

        heading = _extract_heading(match.group())

        chunks.append({
            "text": chunk_text,
            "position": len(chunks),
            "heading": heading,
        })

    # Split oversized chunks
    final_chunks = []
    for chunk in chunks:
        final_chunks.extend(_split_large_chunk(chunk))

    # Reassign sequential positions
    for i, chunk in enumerate(final_chunks):
        chunk["position"] = i

    return final_chunks


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

    logger.info(f"Classifying {len(chunks)} clauses in parallel...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        classifications = list(executor.map(
            lambda chunk: classify_clause_type(chunk["text"], provider),
            chunks,
        ))

    classified = []
    for chunk, classification in zip(chunks, classifications):
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
