"""
Embeddings & Vector Store Loading

Handles converting legal clause text into vector embeddings (via configured
LLM provider) and upserting them into the configured vector store backend.
"""

import json
import os

import numpy as np
from dotenv import load_dotenv

from src.provider import create_provider
from src.vector_store import create_vector_store

# Load API key from .env file
load_dotenv()


def get_embeddings(texts: list[str], provider) -> np.ndarray:
    """
    Convert a list of text strings into vector embeddings via the configured provider.
    Returns a float32 numpy array with shape (len(texts), embedding_dim).
    """
    return provider.embed(texts)


def infer_practice_area(clause_type: str) -> str:
    """Map clause type to a practice area for metadata enrichment."""
    mapping = {
        "NDA": "intellectual_property",
        "Employment": "employment_labor",
        "Service Agreement": "commercial_contracts",
    }
    return mapping.get(clause_type, "general")


def load_clause_database(data_path: str = "data/clauses.json") -> dict:
    """
    Load clauses from JSON, create embeddings, and build vector store.

    Returns a dict with:
    - 'store': the VectorStore instance for similarity search
    - 'clauses': the original clause data
    - 'provider': the LLM provider (reused for query-time embedding and chat)
    """
    provider = create_provider()
    provider_name = os.environ.get("VECTOR_STORE_PROVIDER", "faiss")
    store = create_vector_store(provider_name)

    with open(data_path) as f:
        clauses = json.load(f)

    print(f"Loaded {len(clauses)} legal clauses")

    # Combine title + text for richer semantic representation
    texts_to_embed = [
        f"{clause['title']}: {clause['text']}"
        for clause in clauses
    ]

    print(f"Creating embeddings via {provider.provider_name}...")
    embeddings = get_embeddings(texts_to_embed, provider)
    print(f"Created {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")

    # Build metadata for each clause
    ids = [clause["id"] for clause in clauses]
    metadata = []
    for clause in clauses:
        metadata.append({
            # Current clause fields
            "title": clause["title"],
            "type": clause["type"],
            "category": clause["category"],
            "risk_level": clause["risk_level"],
            # Unified document schema fields for future use
            "source": "clauses_json",
            "doc_type": "clause",
            "clause_type": clause["type"],
            "text": clause["text"],
            "notes": clause["notes"],
            "practice_area": infer_practice_area(clause["type"]),
        })

    count = store.upsert(ids, embeddings, metadata)
    print(f"Vector store ({provider_name}) loaded with {count} vectors")

    return {
        "store": store,
        "clauses": clauses,
        "provider": provider,
    }
