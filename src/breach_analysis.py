"""
Breach Response Analysis Pipeline.

Takes breach parameters as input, retrieves applicable state statutes,
and generates a jurisdiction-by-jurisdiction notification matrix.
"""

import json
import logging

from src.retrieval import search_similar_clauses
from src.generation import generate_analysis
from src.output_parser import parse_json_response_or_raw

logger = logging.getLogger(__name__)

# Breach parameter schema for input validation
BREACH_PARAMS_REQUIRED = {"data_types_compromised", "affected_states"}


def validate_breach_params(params: dict) -> list[str]:
    """Validate breach parameters. Returns list of errors."""
    errors = []
    for field in BREACH_PARAMS_REQUIRED:
        if field not in params:
            errors.append(f"Missing required field: {field}")
    if "data_types_compromised" in params and not params["data_types_compromised"]:
        errors.append("data_types_compromised cannot be empty")
    if "affected_states" in params and not params["affected_states"]:
        errors.append("affected_states cannot be empty")
    return errors


def retrieve_applicable_statutes(
    breach_params: dict,
    db: dict,
    top_k_per_state: int = 5,
) -> dict[str, list[dict]]:
    """
    Retrieve applicable statute provisions for each affected state.

    Uses metadata filtering to get state-specific results, then
    semantic search within each state's provisions to find the most
    relevant sections for the specific breach type.

    Returns: {state_abbr: [list of retrieval results]}
    """
    affected_states = breach_params["affected_states"]
    data_types = breach_params["data_types_compromised"]

    # Build a query that captures the breach specifics
    query = (
        f"Data breach notification requirements for breach involving "
        f"{', '.join(data_types)}. "
        f"Encryption status: {breach_params.get('encryption_status', 'unknown')}. "
        f"Number of affected individuals: {breach_params.get('number_of_affected_individuals', 'unknown')}."
    )

    results_by_state = {}
    for state in affected_states:
        state_upper = state.upper()
        state_results = search_similar_clauses(
            query,
            db,
            top_k=top_k_per_state,
            filters={"jurisdiction": state_upper, "source": "statutes"},
        )
        results_by_state[state_upper] = state_results
        logger.info(f"Retrieved {len(state_results)} provisions for {state_upper}")

    return results_by_state


# Prompt for per-state breach analysis
BREACH_ANALYSIS_PROMPT = """You are a privacy attorney analyzing data breach notification requirements.

Given the breach parameters and the applicable state statute provisions, determine:
1. Whether notification is required under this state's law
2. The notification deadline
3. Who must be notified (individuals, AG, other agencies)
4. Content requirements for the notification
5. Whether any safe harbors apply (e.g., encryption)
6. Special considerations for this breach type

Breach Parameters:
{breach_params}

State Statute Provisions:
{statute_context}

Respond in this JSON format:
{{
    "jurisdiction": "{state}",
    "notification_required": true/false,
    "rationale": "Why notification is or is not required",
    "deadline": "Specific deadline or statutory language",
    "deadline_from_discovery": "Calculated date if specific day count exists, otherwise null",
    "notify_individuals": true/false,
    "notify_ag": true/false,
    "ag_notification_details": "Threshold and details for AG notification",
    "notify_other": ["List of other agencies to notify"],
    "content_requirements": ["List of required notification content"],
    "safe_harbor_applies": true/false,
    "safe_harbor_details": "Whether and why encryption or other safe harbor applies",
    "special_considerations": ["State-specific issues for this breach"],
    "sources_used": [{{"id": "source-id", "relevance": "why"}}],
    "confidence": "high | medium | low"
}}"""


def analyze_breach_for_state(
    breach_params: dict,
    state: str,
    statute_results: list[dict],
    provider,
) -> dict:
    """Generate breach notification analysis for a single state."""
    from src.retrieval import format_retrieval_results

    statute_context = format_retrieval_results(statute_results)

    messages = [
        {"role": "system", "content": "You are an expert privacy attorney. Return only valid JSON."},
        {"role": "user", "content": BREACH_ANALYSIS_PROMPT.format(
            breach_params=json.dumps(breach_params, indent=2),
            statute_context=statute_context,
            state=state,
        )},
    ]

    raw_response = generate_analysis(messages, provider, temperature=0.1)
    parsed = parse_json_response_or_raw(raw_response)
    return parsed


def generate_breach_report(
    breach_params: dict,
    db: dict,
) -> dict:
    """
    Full breach response pipeline.

    1. Validate breach parameters
    2. Retrieve applicable statutes per state
    3. Generate per-state analysis
    4. Build cross-jurisdiction summary matrix

    Returns a complete breach report with per-state details and summary.
    """
    errors = validate_breach_params(breach_params)
    if errors:
        return {"error": "Invalid breach parameters", "details": errors}

    logger.info(f"Breach analysis: {len(breach_params['affected_states'])} states, "
                f"data types: {breach_params['data_types_compromised']}")

    # Retrieve statutes
    statutes_by_state = retrieve_applicable_statutes(breach_params, db)

    # Analyze per state
    state_analyses = []
    for state, statutes in statutes_by_state.items():
        if not statutes:
            logger.warning(f"No statute data found for {state}")
            state_analyses.append({
                "jurisdiction": state,
                "error": f"No breach notification statute data available for {state}",
            })
            continue

        logger.info(f"Analyzing breach requirements for {state}...")
        analysis = analyze_breach_for_state(
            breach_params, state, statutes, db["provider"]
        )
        state_analyses.append(analysis)

    # Build summary
    summary = _build_summary(breach_params, state_analyses)

    return {
        "breach_params": breach_params,
        "summary": summary,
        "state_analyses": state_analyses,
        "review_status": "pending_review",
        "disclaimer": (
            "DRAFT BREACH RESPONSE ANALYSIS — Requires Attorney Review. "
            "This analysis was generated by an AI system and must be verified "
            "against current statute text by a licensed attorney before reliance. "
            "Statutes may have been amended since data was last updated."
        ),
    }


def _build_summary(breach_params: dict, state_analyses: list[dict]) -> dict:
    """Build cross-jurisdiction summary from individual state analyses."""
    notifications_required = 0
    ag_notifications = []
    earliest_deadline = None
    earliest_deadline_state = None
    safe_harbor_applies = False
    safe_harbor_reason = ""

    for analysis in state_analyses:
        if not isinstance(analysis, dict):
            continue
        if analysis.get("notification_required"):
            notifications_required += 1
        if analysis.get("notify_ag"):
            jurisdiction = analysis.get("jurisdiction", "Unknown")
            details = analysis.get("ag_notification_details", "")
            ag_notifications.append(f"{jurisdiction}: {details}")
        deadline = analysis.get("deadline", "")
        if deadline and ("day" in deadline.lower() or "hour" in deadline.lower()):
            if earliest_deadline is None or len(deadline) < len(earliest_deadline):
                earliest_deadline = deadline
                earliest_deadline_state = analysis.get("jurisdiction")
        if analysis.get("safe_harbor_applies"):
            safe_harbor_applies = True
            if not safe_harbor_reason:
                safe_harbor_reason = analysis.get("safe_harbor_details", "")

    return {
        "total_jurisdictions": len(state_analyses),
        "notifications_required": notifications_required,
        "ag_notifications_required": ag_notifications,
        "earliest_deadline": earliest_deadline or "See individual state analyses",
        "earliest_deadline_state": earliest_deadline_state,
        "safe_harbor_applies": safe_harbor_applies,
        "safe_harbor_reason": safe_harbor_reason,
        "data_types_compromised": breach_params.get("data_types_compromised", []),
        "encryption_status": breach_params.get("encryption_status", "unknown"),
    }
