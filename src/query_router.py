"""
Query router for the unified knowledge base.

Classifies user queries to determine:
- Which data sources to search (clauses, statutes, playbooks, etc.)
- What metadata filters to apply
- Whether to use structured retrieval, semantic search, or both
- Optional query rewriting for better retrieval
"""

import logging

from src.generation import generate_analysis
from src.output_parser import parse_json_response_or_raw

logger = logging.getLogger(__name__)


ROUTER_PROMPT = """You are a query classifier for a legal knowledge base.

The knowledge base contains these document types:
- clauses: Contract clauses (NDAs, employment, service agreements) with risk analysis
- statutes: State data breach notification laws (CA, NY, TX, FL, IL, WA, MA, CO, VA, CT)
- playbooks: Firm playbook positions for contract negotiation (SaaS vendor, NDA)

Given a user's question, classify it and determine the best search strategy.

User question: "{query}"

Respond with ONLY this JSON:
{{
    "query_type": "contract_review | breach_response | general_legal | cross_cutting",
    "filters": {{
        "source": "clauses_json | cuad | statutes | common_paper | null",
        "doc_type": "clause | statute | playbook | null",
        "jurisdiction": "state abbreviation or null",
        "clause_type": "specific clause type or null"
    }},
    "search_strategy": "semantic | filtered | hybrid",
    "rewritten_query": "An improved version of the query for better retrieval, or null if the original is fine",
    "explanation": "Brief explanation of why this routing was chosen"
}}

Guidelines:
- If the question mentions a specific state or jurisdiction, set the jurisdiction filter
- If the question is about breach notification, route to statutes
- If the question is about contract terms or negotiation positions, route to clauses/playbooks
- If the question spans multiple areas, use "cross_cutting" with no source filter
- "hybrid" search means apply filters AND semantic search
- "semantic" means pure vector similarity (no filters)
- "filtered" means metadata-only lookup (for very specific queries)
- Set filters to null (not the string "null") when a filter shouldn't be applied"""


def route_query(query: str, provider) -> dict:
    """
    Classify a user query and determine search strategy.

    Returns a routing dict with query_type, filters, search_strategy,
    and optional rewritten_query.
    """
    messages = [
        {"role": "system", "content": "You are a query classifier. Return only valid JSON."},
        {"role": "user", "content": ROUTER_PROMPT.format(query=query)},
    ]

    raw = generate_analysis(messages, provider, temperature=0.0, max_tokens=300)
    parsed = parse_json_response_or_raw(raw)

    if isinstance(parsed, dict) and "query_type" in parsed:
        # Clean up null strings to actual None
        filters = parsed.get("filters", {})
        if filters:
            cleaned = {k: (v if v and v != "null" else None) for k, v in filters.items()}
            # Remove None values entirely
            parsed["filters"] = {k: v for k, v in cleaned.items() if v is not None}
        return parsed

    # Fallback: semantic search with no filters
    logger.warning("Router failed to parse, falling back to semantic search")
    return {
        "query_type": "general_legal",
        "filters": {},
        "search_strategy": "semantic",
        "rewritten_query": None,
        "explanation": "Router parse failure — using broad semantic search",
    }
