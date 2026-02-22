"""
Embeddings & FAISS Vector Store

Handles converting legal clause text into vector embeddings (OpenAI API)
and storing/searching those vectors with FAISS.
"""

import json
import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI

# Load API key from .env file
load_dotenv()

def get_embeddings(texts: list[str], client: OpenAI) -> np.ndarray:
    """
    Convert a list of text strings into vector embeddings via OpenAI's API.
    Returns a float32 numpy array of shape (len(texts), 1536).
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    embeddings = [item.embedding for item in response.data]
    return np.array(embeddings, dtype="float32")


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
    - 'client': the OpenAI client (reused for query-time embedding)

    In production, the index would be persisted and updated incrementally.
    """
    client = OpenAI()

    with open(data_path) as f:
        clauses = json.load(f)

    print(f"Loaded {len(clauses)} legal clauses")

    # Combine title + text for richer semantic representation
    texts_to_embed = [
        f"{clause['title']}: {clause['text']}"
        for clause in clauses
    ]

    print("Creating embeddings via OpenAI API...")
    embeddings = get_embeddings(texts_to_embed, client)
    print(f"Created {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")

    # Build the FAISS index
    index = build_faiss_index(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors")

    return {
        "index": index,
        "clauses": clauses,
        "client": client,
    }
