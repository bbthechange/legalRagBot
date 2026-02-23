"""
LLM Generation Module — Prompt Engineering for Legal Analysis

Three prompt strategies for analyzing legal clauses, designed to be
compared against each other via the evaluation framework.
"""


# --- Strategy 1: Basic (baseline) ---

BASIC_SYSTEM_PROMPT = """You are a legal assistant. Analyze the contract clause provided by the user.

When referencing information from the knowledge base, cite the source ID in square brackets (e.g., [nda-001]). Include a "sources_used" list in your response identifying which sources informed your analysis."""

def build_basic_prompt(query_clause: str, retrieved_context: str) -> list[dict]:
    """Minimal prompt — serves as the baseline for strategy comparison."""
    return [
        {"role": "system", "content": BASIC_SYSTEM_PROMPT},
        {"role": "user", "content": f"""Analyze this contract clause:

{query_clause}

Here are some similar clauses for reference:
{retrieved_context}"""},
    ]


# --- Strategy 2: Structured (detailed system prompt + JSON output) ---

STRUCTURED_SYSTEM_PROMPT = """You are a senior legal analyst at a large law firm specializing in contract review.

Your task is to analyze contract clauses that attorneys submit for review. You will be provided with:
1. The clause to analyze
2. Similar clauses from the firm's knowledge base with their risk assessments

Provide your analysis in the following JSON format:
{
    "assumptions": ["List any assumptions you're making that could change the analysis (e.g., jurisdiction, party types, contract context)"],
    "risk_level": "low | medium | high",
    "risk_summary": "One sentence explaining the overall risk",
    "key_issues": ["List of specific issues found"],
    "comparison": "How this clause compares to the similar clauses from the knowledge base",
    "suggested_revisions": "Specific language changes to improve the clause",
    "jurisdiction_notes": "Any jurisdiction-specific concerns (e.g., state law variations)",
    "sources_used": [{"id": "clause-id", "title": "clause title", "relevance": "why this source was relevant"}],
    "confidence": "high | medium | low",
    "confidence_rationale": "Why you're confident or not in this analysis"
}

Guidelines:
- ALWAYS state your assumptions first — if jurisdiction, party type, or contract context
  would change your analysis, the attorney needs to know
- Be specific and cite exact language from the clause when identifying issues
- Consider enforceability across major US jurisdictions
- Prioritize practical, actionable advice over theoretical concerns
- If the clause is well-drafted, say so — don't invent problems
- ALWAYS cite source IDs from the knowledge base when making comparisons or referencing similar clauses. Use the format [source-id] inline.
- Include a confidence level. High = multiple corroborating sources and clear legal standards. Medium = some relevant sources but analysis requires interpretation. Low = limited relevant sources or novel clause structure."""

def build_structured_prompt(query_clause: str, retrieved_context: str) -> list[dict]:
    """Detailed system prompt with persona, JSON schema, and behavioral guidelines."""
    return [
        {"role": "system", "content": STRUCTURED_SYSTEM_PROMPT},
        {"role": "user", "content": f"""Please analyze the following contract clause:

CLAUSE TO REVIEW:
{query_clause}

SIMILAR CLAUSES FROM KNOWLEDGE BASE:
{retrieved_context}

Provide your analysis in the specified JSON format."""},
    ]


# --- Strategy 3: Few-Shot (includes worked example) ---

FEW_SHOT_SYSTEM_PROMPT = """You are a senior legal analyst at a large law firm specializing in contract review. Analyze contract clauses and provide risk assessments following the format shown in the examples below. Cite source IDs in square brackets (e.g., [nda-001]) when referencing knowledge base sources."""

# Worked example teaches the model expected format, tone, and depth
FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": """Analyze this clause:
"The Contractor agrees not to work for any competitor for 5 years after termination, worldwide."

Similar clauses from knowledge base:
- Reasonable Non-Compete (12 months, 50-mile radius) — Risk: medium
- Overly Broad Non-Compete (3 years, worldwide) — Risk: high"""
    },
    {
        "role": "assistant",
        "content": """{
    "assumptions": [
        "Assuming US jurisdiction (analysis would differ significantly under EU or UK law)",
        "Assuming the contractor is an individual, not a corporate entity",
        "Assuming this is a standalone non-compete, not part of a sale-of-business agreement (where broader terms are more enforceable)"
    ],
    "risk_level": "high",
    "risk_summary": "This non-compete is almost certainly unenforceable due to excessive duration and unlimited geographic scope.",
    "key_issues": [
        "5-year duration far exceeds the typical enforceable range of 6-24 months",
        "Worldwide geographic scope is overbroad — courts require reasonable geographic limits",
        "No limitation to specific business lines or roles",
        "Applies to 'any competitor' which is vague and overinclusive"
    ],
    "comparison": "This clause is even more restrictive than the 'Overly Broad Non-Compete' [emp-002] in the knowledge base (3 years worldwide, rated high risk). It exceeds that by 2 additional years. The knowledge base's reasonable example [emp-001] uses 12 months and 50 miles — a dramatic contrast.",
    "suggested_revisions": "Reduce duration to 12 months. Limit geography to markets where contractor actively worked. Narrow scope to specific competing services rather than 'any competitor'. Add consideration clause if not already present.",
    "jurisdiction_notes": "California bans non-competes entirely (Cal. Bus. & Prof. Code 16600). Many states require reasonable limits on time, geography, and scope. Even in employer-friendly states like Florida, 5 years would likely be struck down or blue-penciled.",
    "sources_used": [
        {
            "id": "emp-001",
            "title": "Reasonable Non-Compete Clause",
            "relevance": "Baseline for reasonable duration and geographic scope"
        },
        {
            "id": "emp-002",
            "title": "Overly Broad Non-Compete",
            "relevance": "Similar overbroad pattern — worldwide scope, excessive duration"
        }
    ],
    "confidence": "high",
    "confidence_rationale": "Multiple non-compete clauses in knowledge base establish clear enforceability standards. Duration and scope analysis is well-settled law across US jurisdictions."
}"""
    },
]

def build_few_shot_prompt(query_clause: str, retrieved_context: str) -> list[dict]:
    """
    Few-shot prompt with a worked example before the actual query.
    The example uses a different clause type to encourage generalization.
    """
    messages = [
        {"role": "system", "content": FEW_SHOT_SYSTEM_PROMPT},
    ]
    messages.extend(FEW_SHOT_EXAMPLES)
    messages.append({
        "role": "user",
        "content": f"""Now analyze this clause:

CLAUSE TO REVIEW:
{query_clause}

SIMILAR CLAUSES FROM KNOWLEDGE BASE:
{retrieved_context}"""
    })
    return messages


# --- Strategy 4: Knowledge Base QA (unified search) ---

KNOWLEDGE_BASE_QA_PROMPT = """You are a legal knowledge base assistant. Answer the user's question based ONLY on the retrieved documents below.

Rules:
- Answer the specific question asked
- Cite sources by their IDs in square brackets (e.g., [statute-ca-summary])
- If the retrieved documents don't contain enough information to fully answer, say so explicitly
- Do not make up or infer legal requirements not supported by the sources
- If comparing across jurisdictions or clause types, organize the answer clearly

Retrieved documents:
{retrieved_context}

Respond in JSON:
{{
    "answer": "Your answer to the question, with [source-id] citations inline",
    "sources_used": [
        {{"id": "doc-id", "title": "doc title", "relevance": "why this source was relevant"}}
    ],
    "confidence": "high | medium | low",
    "caveats": ["Any important limitations or assumptions"],
    "related_queries": ["2-3 follow-up questions the user might want to ask"]
}}"""


def build_knowledge_base_qa_prompt(query: str, retrieved_context: str) -> list[dict]:
    """Knowledge base question-answering prompt with citation requirements."""
    return [
        {"role": "system", "content": KNOWLEDGE_BASE_QA_PROMPT.format(
            retrieved_context=retrieved_context
        )},
        {"role": "user", "content": query},
    ]


def generate_analysis(
    messages: list[dict],
    provider,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 1500,
) -> str:
    """
    Send the prompt to the LLM and return the response.

    Temperature is kept low (0.2) for consistency — legal analysis
    needs to be reproducible, not creative. Model defaults to the
    provider's chat_model for cost efficiency.
    """
    return provider.chat(messages, model=model, temperature=temperature, max_tokens=max_tokens)
