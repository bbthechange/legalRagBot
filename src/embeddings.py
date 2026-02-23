"""
Embeddings & Vector Store Loading

Handles converting legal clause text into vector embeddings (via configured
LLM provider) and upserting them into the configured vector store backend.
Supports both legacy clauses.json format and unified document schema.
"""

import hashlib
import json
import logging
import os

import numpy as np
from dotenv import load_dotenv

from src.provider import create_provider
from src.schemas import validate_document
from src.vector_store import create_vector_store, FaissVectorStore

logger = logging.getLogger(__name__)

# Load API key from .env file
load_dotenv()


def get_embeddings(texts: list[str], provider, batch_size: int = 100) -> np.ndarray:
    """
    Convert text strings into vector embeddings, batching for API limits.

    OpenAI allows up to 2048 texts per request, but large batches risk
    token limits. Default batch_size=100 is safe for most text lengths.
    """
    if len(texts) <= batch_size:
        return provider.embed(texts)

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        logger.info(f"Embedding batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1} ({len(batch)} texts)")
        embeddings = provider.embed(batch)
        all_embeddings.append(embeddings)

    return np.concatenate(all_embeddings, axis=0)


def infer_practice_area(clause_type: str) -> str:
    """Map clause type to a practice area for metadata enrichment."""
    mapping = {
        "NDA": "intellectual_property",
        "Employment": "employment_labor",
        "Service Agreement": "commercial_contracts",
    }
    return mapping.get(clause_type, "general")


def _load_clauses_json(data_path: str) -> list[dict]:
    """Load clauses.json and convert each clause to unified schema format."""
    with open(data_path) as f:
        raw_clauses = json.load(f)

    documents = []
    for clause in raw_clauses:
        documents.append({
            "doc_id": clause["id"],
            "source": "clauses_json",
            "doc_type": "clause",
            "title": clause["title"],
            "text": clause["text"],
            "metadata": {
                "clause_type": clause["type"],
                "category": clause["category"],
                "risk_level": clause["risk_level"],
                "notes": clause["notes"],
                "practice_area": infer_practice_area(clause["type"]),
            },
        })

    return documents


def load_documents(
    documents: list[dict] | None = None,
    data_path: str | None = None,
    index_path: str | None = None,
) -> dict:
    """
    Load documents into the vector store, accepting either unified-schema
    docs directly or a clauses.json path.

    Args:
        documents: List of dicts in unified document schema.
        data_path: Path to a clauses.json file (converted automatically).
        index_path: Base path for FAISS persistence (load/save).

    Returns:
        Dict with 'store', 'documents', 'clauses' (alias), 'provider'.
    """
    if documents is None and data_path is None:
        raise ValueError("Must provide either documents or data_path")

    provider = create_provider()
    provider_name = os.environ.get("VECTOR_STORE_PROVIDER", "faiss")
    store = create_vector_store(provider_name)

    if documents is None:
        documents = _load_clauses_json(data_path)

    logger.info("Loading %d documents", len(documents))
    print(f"Loaded {len(documents)} documents")

    # Content fingerprint to detect stale persisted indexes
    content_hash = hashlib.sha256(
        "|".join(d["doc_id"] + ":" + d["text"][:64] for d in documents).encode()
    ).hexdigest()[:16]

    # Try loading from persisted index if available
    if index_path and isinstance(store, FaissVectorStore):
        if store.load(index_path) and store.total_vectors == len(documents):
            stored_hash = store.get_content_hash()
            if stored_hash == content_hash:
                logger.info("Loaded persisted FAISS index from %s", index_path)
                print(f"Loaded persisted index ({store.total_vectors} vectors)")
                return {
                    "store": store,
                    "documents": documents,
                    "clauses": documents,
                    "provider": provider,
                }
            logger.info("Persisted index content hash mismatch, rebuilding")

    # Validate documents (warn but don't fail)
    for doc in documents:
        errors = validate_document(doc)
        if errors:
            logger.warning("Document %s validation: %s", doc.get("doc_id", "?"), errors)

    # Build embedding text
    texts_to_embed = [
        f"{doc['title']}: {doc['text']}"
        for doc in documents
    ]

    print(f"Creating embeddings via {provider.provider_name}...")
    embeddings = get_embeddings(texts_to_embed, provider)
    print(f"Created {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")

    # Build metadata — flatten for vector store upsert
    ids = [doc["doc_id"] for doc in documents]
    metadata = []
    for doc in documents:
        flat = {
            "title": doc["title"],
            "text": doc["text"],
            "source": doc["source"],
            "doc_type": doc["doc_type"],
        }
        # Flatten nested metadata, skip None values
        for k, v in doc.get("metadata", {}).items():
            if v is not None:
                flat[k] = v
        # Keep "type" key for backward compat with clause_type
        if "clause_type" in flat:
            flat["type"] = flat["clause_type"]
        metadata.append(flat)

    count = store.upsert(ids, embeddings, metadata)
    logger.info("Vector store loaded with %d vectors", count)
    print(f"Vector store ({provider_name}) loaded with {count} vectors")

    # Persist if path given
    if index_path and isinstance(store, FaissVectorStore):
        store.save(index_path, content_hash=content_hash)
        logger.info("Saved FAISS index to %s", index_path)

    return {
        "store": store,
        "documents": documents,
        "clauses": documents,
        "provider": provider,
    }


def load_clause_database(data_path: str = "data/clauses.json") -> dict:
    """
    Load clauses from JSON, create embeddings, and build vector store.

    Backward-compatible wrapper around load_documents().

    Returns a dict with:
    - 'store': the VectorStore instance for similarity search
    - 'clauses': the document list (unified schema)
    - 'documents': same as 'clauses'
    - 'provider': the LLM provider (reused for query-time embedding and chat)
    """
    return load_documents(data_path=data_path, index_path="data/index/main")
