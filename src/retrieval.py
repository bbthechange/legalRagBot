"""
Retrieval Module — Finding Similar Legal Clauses

Given a query clause, find the most semantically similar clauses
from the knowledge base using FAISS vector search.
"""

import numpy as np
import faiss
from src.embeddings import get_embeddings


def search_similar_clauses(
    query: str,
    db: dict,
    top_k: int = 3,
) -> list[dict]:
    """
    Find the most similar clauses to a query string.

    Args:
        query: The clause text to analyze
        db: The database dict from load_clause_database()
        top_k: Number of similar clauses to return

    Returns:
        List of dicts with 'clause' (original data) and 'score' (cosine similarity).
        Could be extended with score threshold filtering or metadata filters.
    """
    query_embedding = get_embeddings([query], db["client"])
    faiss.normalize_L2(query_embedding)
    scores, indices = db["index"].search(query_embedding, top_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:  # FAISS returns -1 for empty slots
            continue
        results.append({
            "clause": db["clauses"][idx],
            "score": float(score),
        })

    return results


def format_retrieval_results(results: list[dict]) -> str:
    """
    Format retrieved clauses into a readable string.
    Used both for display and as context injected into LLM prompts.
    Includes metadata (type, risk level, notes) to give the LLM
    richer context for generating analysis.
    """
    output = []
    for i, result in enumerate(results, 1):
        clause = result["clause"]
        score = result["score"]
        output.append(
            f"--- Similar Clause {i} (similarity: {score:.3f}) ---\n"
            f"Type: {clause['type']} | Category: {clause['category']}\n"
            f"Title: {clause['title']}\n"
            f"Risk Level: {clause['risk_level']}\n"
            f"Text: {clause['text']}\n"
            f"Attorney Notes: {clause['notes']}\n"
        )
    return "\n".join(output)
