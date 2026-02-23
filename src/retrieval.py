"""
Retrieval Module — Finding Similar Legal Documents

Given a query, find the most semantically similar documents
from the knowledge base using the configured vector store.
Reconstructs all results from vector store metadata (source-agnostic).
"""

import logging

import numpy as np
from src.embeddings import get_embeddings

logger = logging.getLogger(__name__)


def search_similar_clauses(
    query: str,
    db: dict,
    top_k: int = 3,
    filters: dict[str, str] | None = None,
) -> list[dict]:
    """
    Find the most similar clauses to a query string.

    Args:
        query: The clause text to analyze
        db: The database dict from load_clause_database() or load_documents()
        top_k: Number of similar clauses to return
        filters: Optional metadata filters (e.g. {"type": "NDA"})

    Returns:
        List of dicts with 'clause' (reconstructed from metadata) and 'score'.
    """
    logger.info("Search: query='%s...', top_k=%d, filters=%s", query[:80], top_k, filters)
    query_embedding = get_embeddings([query], db["provider"])

    # L2 normalize using numpy (no FAISS dependency in retrieval layer)
    norm = np.linalg.norm(query_embedding, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-12)
    query_embedding = query_embedding / norm

    store_results = db["store"].search(query_embedding, top_k, filters)

    results = []
    for hit in store_results:
        meta = hit["metadata"]
        clause = {
            "id": hit["id"],
            "title": meta.get("title", ""),
            "type": meta.get("clause_type", meta.get("type", "")),
            "category": meta.get("category", ""),
            "text": meta.get("text", ""),
            "risk_level": meta.get("risk_level", ""),
            "notes": meta.get("notes", ""),
            "source": meta.get("source", ""),
            "doc_type": meta.get("doc_type", "clause"),
            "jurisdiction": meta.get("jurisdiction", ""),
            "citation": meta.get("citation", ""),
            "position": meta.get("position", ""),
        }
        results.append({
            "clause": clause,
            "score": hit["score"],
        })

    if results:
        logger.info("Retrieved %d results, top score: %.3f", len(results), results[0]["score"])
    else:
        logger.debug("Retrieved 0 results")
    return results


def format_retrieval_results(results: list[dict]) -> str:
    """
    Format retrieved documents into a readable string.
    Uses doc_type to select the appropriate format branch.
    """
    output = []
    for result in results:
        doc = result["clause"]
        score = result["score"]
        doc_type = doc.get("doc_type", "clause")

        header = f"--- Source [{doc['id']}] (similarity: {score:.3f}) ---\n"

        if doc_type == "statute":
            body = (
                f"Jurisdiction: {doc.get('jurisdiction', '')}\n"
                f"Citation: {doc.get('citation', '')}\n"
                f"Title: {doc['title']}\n"
                f"Text: {doc['text']}\n"
            )
        elif doc_type == "playbook":
            body = (
                f"Playbook: {doc.get('source', '')}\n"
                f"Title: {doc['title']}\n"
                f"Position: {doc.get('position', '')}\n"
            )
        else:
            # Default: clause format
            body = (
                f"Type: {doc['type']} | Category: {doc['category']}\n"
                f"Title: {doc['title']}\n"
                f"Risk Level: {doc['risk_level']}\n"
                f"Text: {doc['text']}\n"
                f"Attorney Notes: {doc['notes']}\n"
            )

        output.append(header + body)

    return "\n".join(output)
