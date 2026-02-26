"""
Playbook-driven contract review pipeline.

Compares extracted contract clauses against the firm's playbook positions
and generates a clause-by-clause report with gap analysis and redline suggestions.
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor

from src.contract_chunker import extract_clauses
from src.retrieval import search_similar_clauses, format_retrieval_results
from src.generation import generate_analysis
from src.output_parser import parse_json_response_or_raw

logger = logging.getLogger(__name__)


def load_playbook(playbook_path: str) -> dict:
    """Load a playbook JSON file."""
    with open(playbook_path) as f:
        return json.load(f)


def find_playbook_position(clause_type: str, playbook: dict) -> dict | None:
    """Find the matching playbook position for a clause type."""
    for entry in playbook.get("clauses", []):
        if entry["clause_type"] == clause_type:
            return entry
    return None


REVIEW_PROMPT = """You are a senior contract attorney reviewing a clause against the firm's playbook.

CLAUSE FROM CONTRACT:
{clause_text}

PLAYBOOK POSITION:
Preferred: {preferred}
Fallback: {fallback}
Walk-away: {walk_away}
Notes: {notes}

SIMILAR CLAUSES FROM KNOWLEDGE BASE:
{retrieved_context}

Analyze how this clause compares to the playbook position. Respond in JSON:
{{
    "clause_type": "{clause_type}",
    "playbook_match": "preferred | fallback | walk_away | not_covered",
    "gaps": [
        {{
            "issue": "Description of the gap",
            "severity": "high | medium | low",
            "playbook_says": "What the playbook requires",
            "clause_says": "What the contract actually says"
        }}
    ],
    "suggested_redline": "Specific language to add, change, or remove",
    "risk_level": "high | medium | low",
    "negotiation_notes": "Practical advice for negotiating this point",
    "sources_used": [{{"id": "source-id", "relevance": "why relevant"}}],
    "confidence": "high | medium | low"
}}"""


def review_clause_against_playbook(
    clause: dict,
    playbook_position: dict,
    db: dict,
) -> dict:
    """Review a single clause against its playbook position."""
    similar = search_similar_clauses(clause["text"], db, top_k=3)
    context = format_retrieval_results(similar)

    messages = [
        {"role": "system", "content": "You are an expert contract review attorney. Return only valid JSON."},
        {"role": "user", "content": REVIEW_PROMPT.format(
            clause_text=clause["text"],
            preferred=playbook_position["preferred_position"],
            fallback=playbook_position["fallback_position"],
            walk_away=playbook_position["walk_away"],
            notes=playbook_position.get("notes", ""),
            retrieved_context=context,
            clause_type=clause["clause_type"],
        )},
    ]

    raw = generate_analysis(messages, db["provider"], temperature=0.1)
    parsed = parse_json_response_or_raw(raw)

    if isinstance(parsed, dict):
        # Ensure required keys have defaults (LLM output may be incomplete)
        parsed.setdefault("clause_type", clause["clause_type"])
        parsed.setdefault("playbook_match", "not_covered")
        parsed.setdefault("gaps", [])
        parsed.setdefault("suggested_redline", "")
        parsed.setdefault("risk_level", "medium")
        parsed.setdefault("negotiation_notes", "")
        parsed["extracted_text"] = clause["text"][:500]
        parsed["position_in_contract"] = clause["position"]
        parsed["heading"] = clause.get("heading")
        parsed["preferred_position"] = playbook_position.get("preferred_position", "")

    return parsed


def review_contract(
    contract_text: str,
    playbook_path: str,
    db: dict,
) -> dict:
    """
    Full contract review pipeline.

    1. Extract and classify clauses from the contract
    2. For each clause, find the matching playbook position
    3. Generate comparison analysis with gap identification and redlines
    4. Produce contract-level summary report

    Returns complete review report.
    """
    playbook = load_playbook(playbook_path)
    logger.info(f"Loaded playbook: {playbook['name']} ({len(playbook['clauses'])} clause types)")

    clauses = extract_clauses(contract_text, db["provider"])
    logger.info(f"Extracted {len(clauses)} clauses from contract")

    # Separate clauses with playbook matches (need API calls) from those without
    clauses_to_review = []  # (index, clause, playbook_pos)
    clause_analyses = [None] * len(clauses)

    for i, clause in enumerate(clauses):
        playbook_pos = find_playbook_position(clause["clause_type"], playbook)

        if playbook_pos:
            clauses_to_review.append((i, clause, playbook_pos))
        else:
            clause_analyses[i] = {
                "clause_type": clause["clause_type"],
                "playbook_match": "not_covered",
                "gaps": [],
                "suggested_redline": "",
                "risk_level": "unknown",
                "negotiation_notes": f"This clause type ({clause['clause_type']}) is not covered by the {playbook['name']} playbook.",
                "extracted_text": clause["text"][:500],
                "position_in_contract": clause["position"],
            }

    # Review matched clauses in parallel
    if clauses_to_review:
        logger.info(f"Reviewing {len(clauses_to_review)} clauses in parallel...")
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [
                (i, executor.submit(review_clause_against_playbook, clause, playbook_pos, db))
                for i, clause, playbook_pos in clauses_to_review
            ]
            for i, future in futures:
                try:
                    clause_analyses[i] = future.result()
                except Exception:
                    clause = next(c for idx, c, _ in clauses_to_review if idx == i)
                    logger.error(f"Failed to review clause {i} ({clause['clause_type']}), marking as error")
                    clause_analyses[i] = {
                        "clause_type": clause["clause_type"],
                        "playbook_match": "not_covered",
                        "gaps": [],
                        "suggested_redline": "",
                        "risk_level": "unknown",
                        "negotiation_notes": "Review failed due to an internal error. Manual review required.",
                        "extracted_text": clause["text"][:500],
                        "position_in_contract": clause["position"],
                    }

    summary = _build_contract_summary(clause_analyses, playbook)

    return {
        "playbook": playbook["name"],
        "total_clauses": len(clauses),
        "summary": summary,
        "clause_analyses": clause_analyses,
        "review_status": "pending_review",
        "disclaimer": (
            "DRAFT CONTRACT REVIEW \u2014 Requires Attorney Review. "
            "This analysis was generated by an AI system. All suggested "
            "redlines must be reviewed and approved by a licensed attorney "
            "before use in negotiations."
        ),
    }


def _build_contract_summary(analyses: list[dict], playbook: dict) -> dict:
    """Build contract-level summary from clause analyses."""
    match_counts = {"preferred": 0, "fallback": 0, "walk_away": 0, "not_covered": 0}
    risk_counts = {"high": 0, "medium": 0, "low": 0}
    critical_issues = []

    for a in analyses:
        if isinstance(a, dict):
            match = a.get("playbook_match", "not_covered")
            if match not in match_counts:
                match = "not_covered"
            match_counts[match] += 1

            risk = a.get("risk_level", "unknown")
            if risk in risk_counts:
                risk_counts[risk] += 1

            if a.get("playbook_match") == "walk_away":
                for gap in a.get("gaps", []):
                    if isinstance(gap, dict) and gap.get("severity") == "high":
                        critical_issues.append(gap.get("issue", "Walk-away triggered"))

    overall_risk = "high" if match_counts["walk_away"] > 0 or risk_counts["high"] > 2 else \
                   "medium" if risk_counts["medium"] > 2 else "low"

    return {
        "total_clauses_reviewed": len(analyses),
        "preferred_match": match_counts["preferred"],
        "fallback_match": match_counts["fallback"],
        "walk_away_triggered": match_counts["walk_away"],
        "not_in_playbook": match_counts["not_covered"],
        "overall_risk": overall_risk,
        "critical_issues": critical_issues[:5],
    }
