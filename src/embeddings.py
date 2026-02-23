"""
Embeddings & FAISS Vector Store

Handles converting legal clause text into vector embeddings (OpenAI API)
and storing/searching those vectors with FAISS.
"""

import json
import numpy as np
import faiss
from dotenv import load_dotenv
from src.provider import create_provider

# Load API key from .env file
load_dotenv()

def get_embeddings(texts: list[str], provider) -> np.ndarray:
    """
    Convert a list of text strings into vector embeddings via the configured provider.
    Returns a float32 numpy array with shape (len(texts), embedding_dim).
    """
    return provider.embed(texts)


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Create a FAISS index for cosine similarity search.

    Uses IndexFlatIP (Inner Product) after L2 normalization, which is
    equivalent to cosine similarity. IndexFlatIP performs exact search,
    suitable for datasets up to ~100K vectors. For larger corpora,
    consider IndexIVFFlat or IndexHNSW for approximate search.
    """
    faiss.normalize_L2(embeddings)  # normalize so inner product = cosine similarity
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    return index


def load_clause_database(data_path: str = "data/clauses.json") -> dict:
    """
    Load clauses from JSON, create embeddings, and build FAISS index.

    Returns a dict with:
    - 'index': the FAISS index for similarity search
    - 'clauses': the original clause data
    - 'provider': the LLM provider (reused for query-time embedding and chat)

    In production, the index would be persisted and updated incrementally.
    """
    provider = create_provider()

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

    # Build the FAISS index
    index = build_faiss_index(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors")

    return {
        "index": index,
        "clauses": clauses,
        "provider": provider,
    }
