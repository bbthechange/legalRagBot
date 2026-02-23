"""
RAG Pipeline — Retrieval-Augmented Generation Flow

Connects retrieval and generation into a single pipeline:
Retrieve similar clauses → Augment the prompt with context → Generate analysis.
"""

import logging

from src.embeddings import load_clause_database
from src.retrieval import search_similar_clauses, format_retrieval_results
from src.generation import (
    build_basic_prompt,
    build_structured_prompt,
    build_few_shot_prompt,
    generate_analysis,
)

logger = logging.getLogger(__name__)

# Map strategy names to their prompt builders
STRATEGIES = {
    "basic": build_basic_prompt,
    "structured": build_structured_prompt,
    "few_shot": build_few_shot_prompt,
}


def analyze_clause(
    clause_text: str,
    db: dict,
    strategy: str = "few_shot",
    top_k: int = 3,
    model: str | None = None,
    temperature: float = 0.2,
) -> dict:
    """
    Full RAG pipeline: retrieve similar clauses, then generate analysis.

    Returns the analysis alongside retrieval metadata (which clauses
    were used, similarity scores, strategy, model) for explainability
    and audit trail purposes.
    """
    logger.info("Analyzing clause with strategy=%s, top_k=%d", strategy, top_k)
    retrieved = search_similar_clauses(clause_text, db, top_k=top_k)
    context = format_retrieval_results(retrieved)

    prompt_builder = STRATEGIES[strategy]
    messages = prompt_builder(clause_text, context)

    analysis = generate_analysis(
        messages, db["provider"], model=model, temperature=temperature
    )

    logger.info("Analysis complete (strategy=%s, model=%s)", strategy, model or db["provider"].chat_model)
    return {
        "analysis": analysis,
        "retrieved_clauses": retrieved,
        "strategy": strategy,
        "model": model or db["provider"].chat_model,
    }
