"""
Playbook Review Pipeline — Compare contract clauses against firm positions.

Loads a playbook, extracts clauses from a contract, matches each clause
to a playbook position, and generates a clause-by-clause review report
with risk assessments and redline suggestions.
"""

import json
import logging

from src.contract_chunker import extract_clauses
from src.retrieval import search_similar_clauses, format_retrieval_results
from src.generation import generate_analysis
from src.output_parser import parse_json_response_or_raw

logger = logging.getLogger(__name__)


def load_playbook(path: str) -> dict:
    """Load a playbook JSON file."""
    with open(path) as f:
        return json.load(f)


def find_playbook_position(clause_type: str, playbook: dict) -> dict | None:
    """Find the playbook position for a given clause type. Returns None if not found."""
    for clause in playbook.get("clauses", []):
        if clause["clause_type"] == clause_type:
            return clause
    return None


REVIEW_SYSTEM_PROMPT = """You are a senior contract attorney conducting a playbook-based contract review.

You will receive:
1. A contract clause extracted from an agreement under review
2. The firm's playbook position for this clause type (preferred, fallback, walk-away)
3. Similar clauses from the knowledge base for additional context

Analyze the clause against the playbook position and respond with JSON:
{
    "alignment": "preferred" | "fallback" | "walk_away" | "not_covered",
    "risk_level": "low" | "medium" | "high",
    "analysis": "Detailed explanation of how the clause compares to the playbook position",
    "key_issues": ["List of specific issues found"],
    "redline_suggestions": ["Specific language changes to bring clause closer to preferred position"],
    "sources_used": [{"id": "source-id", "relevance": "why relevant"}]
}

Guidelines:
- "preferred" = clause aligns with or is better than the preferred position
- "fallback" = clause is acceptable but weaker than preferred (within fallback range)
- "walk_away" = clause crosses a walk-away threshold and requires escalation
- "not_covered" = clause type not in playbook; flag for manual review
- Be specific about what language triggers each assessment
- Provide actionable redline suggestions with proposed alternative language"""


def review_clause_against_playbook(
    clause: dict,
    playbook_position: dict | None,
    db: dict,
) -> dict:
    """
    Review a single clause against its playbook position.

    Uses RAG to find similar clauses for additional context,
    then generates an LLM analysis comparing the clause to the playbook.
    """
    # Retrieve similar clauses for context
    results = search_similar_clauses(clause["text"], db, top_k=3)
    context = format_retrieval_results(results)

    if playbook_position:
        playbook_context = (
            f"FIRM PLAYBOOK POSITION ({playbook_position['clause_type']}):\n"
            f"Preferred: {playbook_position['preferred_position']}\n"
            f"Fallback: {playbook_position['fallback_position']}\n"
            f"Walk-away: {playbook_position['walk_away']}\n"
            f"Risk factors: {', '.join(playbook_position.get('risk_factors', []))}\n"
            f"Notes: {playbook_position.get('notes', '')}"
        )
    else:
        playbook_context = "No playbook position available for this clause type. Flag for manual review."

    messages = [
        {"role": "system", "content": REVIEW_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"CONTRACT CLAUSE (classified as: {clause['clause_type']}, "
                f"confidence: {clause['confidence']}):\n"
                f"{clause['text']}\n\n"
                f"{playbook_context}\n\n"
                f"SIMILAR CLAUSES FROM KNOWLEDGE BASE:\n{context}"
            ),
        },
    ]

    raw = generate_analysis(messages, db["provider"], max_tokens=2000)
    analysis = parse_json_response_or_raw(raw)

    return {
        "clause_type": clause["clause_type"],
        "clause_text": clause["text"],
        "heading": clause.get("heading", ""),
        "position": clause.get("position", 0),
        "confidence": clause.get("confidence", "low"),
        "playbook_match": playbook_position is not None,
        "analysis": analysis,
    }


def _build_contract_summary(analyses: list[dict], playbook: dict) -> dict:
    """Build a summary of the full contract review."""
    counts = {"preferred": 0, "fallback": 0, "walk_away": 0, "not_covered": 0}
    risk_levels = {"low": 0, "medium": 0, "high": 0}
    critical_issues = []

    for item in analyses:
        analysis = item.get("analysis", {})
        if isinstance(analysis, dict) and not analysis.get("parse_error"):
            alignment = analysis.get("alignment", "not_covered")
            risk = analysis.get("risk_level", "medium")
        else:
            alignment = "not_covered"
            risk = "medium"

        if alignment not in counts:
            alignment = "not_covered"
        if risk not in risk_levels:
            risk = "medium"

        counts[alignment] = counts.get(alignment, 0) + 1
        risk_levels[risk] = risk_levels.get(risk, 0) + 1

        if alignment == "walk_away":
            critical_issues.append({
                "clause_type": item["clause_type"],
                "heading": item.get("heading", ""),
                "risk_level": "high",
                "reason": (
                    analysis.get("analysis", "Crosses walk-away threshold")
                    if isinstance(analysis, dict) else "Crosses walk-away threshold"
                ),
            })

    # Overall risk: high if any walk-away, medium if fallbacks > preferred, else low
    if counts["walk_away"] > 0:
        overall_risk = "high"
    elif counts["fallback"] > counts["preferred"]:
        overall_risk = "medium"
    else:
        overall_risk = "low"

    return {
        "total_clauses": len(analyses),
        "alignment_counts": counts,
        "risk_levels": risk_levels,
        "overall_risk": overall_risk,
        "critical_issues": critical_issues,
    }


def review_contract(
    contract_text: str,
    playbook_path: str,
    db: dict,
) -> dict:
    """
    Full contract review pipeline.

    1. Load the playbook
    2. Extract and classify clauses from the contract
    3. Match each clause to a playbook position
    4. Review each clause against its position
    5. Build a summary report

    Returns a dict with playbook info, clause analyses, and summary.
    """
    playbook = load_playbook(playbook_path)
    logger.info(f"Loaded playbook: {playbook['name']} ({len(playbook['clauses'])} positions)")

    clauses = extract_clauses(contract_text, db["provider"])
    logger.info(f"Extracted {len(clauses)} clauses from contract")

    analyses = []
    for clause in clauses:
        position = find_playbook_position(clause["clause_type"], playbook)
        analysis = review_clause_against_playbook(clause, position, db)
        analyses.append(analysis)

    summary = _build_contract_summary(analyses, playbook)

    return {
        "playbook": playbook["name"],
        "playbook_id": playbook["playbook_id"],
        "total_clauses": len(clauses),
        "summary": summary,
        "clause_analyses": analyses,
        "review_status": "pending_review",
        "disclaimer": (
            "DRAFT — This analysis is generated by AI and requires review by a qualified attorney. "
            "It does not constitute legal advice."
        ),
    }
