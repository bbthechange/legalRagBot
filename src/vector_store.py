"""
Vector Store Abstraction Layer

Provides a unified interface for vector similarity search with
pluggable backends (FAISS for local dev, Pinecone for production).
"""

import json
import os
import time
from abc import ABC, abstractmethod

import numpy as np


class VectorStore(ABC):
    """Abstract base class for vector stores."""

    @abstractmethod
    def upsert(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        metadata: list[dict],
    ) -> int:
        """
        Insert or update vectors with associated metadata.

        Returns the number of vectors upserted.
        """
        ...

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
        filters: dict | None = None,
    ) -> list[dict]:
        """
        Find the most similar vectors to a query.

        Returns list of dicts with keys: id, score, metadata.
        """
        ...

    @abstractmethod
    def delete(self, ids: list[str]) -> int:
        """
        Delete vectors by ID.

        Returns the number of vectors deleted.
        """
        ...


class FaissVectorStore(VectorStore):
    """FAISS-backed vector store for local development."""

    def __init__(self, index_dir: str | None = None):
        self._index = None
        self._ids: list[str] = []
        self._metadata: dict[str, dict] = {}
        self._deleted_ids: set[str] = set()
        self._index_dir = index_dir

    def upsert(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        metadata: list[dict],
    ) -> int:
        import faiss

        embeddings = embeddings.copy().astype(np.float32)
        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dimension)
        self._index.add(embeddings)

        self._ids = list(ids)
        self._metadata = {id_: meta for id_, meta in zip(ids, metadata)}

        return len(ids)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
        filters: dict | None = None,
    ) -> list[dict]:
        if self._index is None:
            return []

        # If filtering, over-fetch then filter
        fetch_k = top_k * 5 if filters else top_k

        scores, indices = self._index.search(query_embedding, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if idx >= len(self._ids):
                continue

            vec_id = self._ids[idx]
            if vec_id in self._deleted_ids:
                continue
            meta = self._metadata.get(vec_id, {})

            # Apply post-retrieval filtering
            if filters:
                if not all(meta.get(k) == v for k, v in filters.items()):
                    continue

            results.append({
                "id": vec_id,
                "score": float(score),
                "metadata": meta,
            })

            if len(results) >= top_k:
                break

        return results

    def delete(self, ids: list[str]) -> int:
        deleted = 0
        for vec_id in ids:
            if vec_id in self._metadata and vec_id not in self._deleted_ids:
                self._deleted_ids.add(vec_id)
                deleted += 1
        return deleted

    @property
    def total_vectors(self) -> int:
        return self._index.ntotal if self._index else 0

    def save(self, path: str | None = None, content_hash: str | None = None) -> str:
        """Save the FAISS index and metadata to disk.

        Args:
            path: Base path (without extension). Defaults to self._index_dir
                  or "data/index/main".
            content_hash: Optional content fingerprint for staleness detection.

        Returns:
            The base path used.
        """
        import faiss

        if self._index is None:
            raise ValueError("Cannot save: no index has been built (call upsert first)")

        if path is None:
            path = self._index_dir or "data/index/main"

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        faiss.write_index(self._index, f"{path}.index")

        meta = {
            "ids": self._ids,
            "metadata": self._metadata,
            "deleted_ids": list(self._deleted_ids),
            "content_hash": content_hash,
        }
        with open(f"{path}.meta.json", "w") as f:
            json.dump(meta, f)

        return path

    def load(self, path: str | None = None) -> bool:
        """Load FAISS index and metadata from disk.

        Args:
            path: Base path (without extension). Defaults to self._index_dir
                  or "data/index/main".

        Returns:
            True on success, False if files don't exist or are corrupted.
        """
        import faiss

        if path is None:
            path = self._index_dir or "data/index/main"

        index_file = f"{path}.index"
        meta_file = f"{path}.meta.json"

        if not os.path.exists(index_file) or not os.path.exists(meta_file):
            return False

        try:
            self._index = faiss.read_index(index_file)

            with open(meta_file) as f:
                meta = json.load(f)

            if not all(k in meta for k in ("ids", "metadata", "deleted_ids")):
                logging.getLogger(__name__).warning("Corrupted meta.json at %s", meta_file)
                return False

            self._ids = meta["ids"]
            self._metadata = meta["metadata"]
            self._deleted_ids = set(meta["deleted_ids"])
            self._content_hash = meta.get("content_hash")
        except (json.JSONDecodeError, KeyError, Exception) as e:
            logging.getLogger(__name__).warning("Failed to load index from %s: %s", path, e)
            return False

        return True

    def get_content_hash(self) -> str | None:
        """Return the content hash from the last load(), or None."""
        return getattr(self, "_content_hash", None)


class PineconeVectorStore(VectorStore):
    """Pinecone-backed vector store for production deployments."""

    def __init__(self, api_key: str, index_name: str, dimension: int = 1536):
        from pinecone import Pinecone, ServerlessSpec

        self._client = Pinecone(api_key=api_key)
        self._index_name = index_name

        # Create index if it doesn't exist
        existing = [idx.name for idx in self._client.list_indexes()]
        if index_name not in existing:
            self._client.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            # Wait for index to be ready with timeout
            deadline = time.time() + 120
            while not self._client.describe_index(index_name).status.ready:
                if time.time() > deadline:
                    raise TimeoutError(
                        f"Pinecone index '{index_name}' not ready after 120s"
                    )
                time.sleep(1)

        self._index = self._client.Index(index_name)

    def upsert(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        metadata: list[dict],
    ) -> int:
        batch_size = 100
        total = 0

        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size]
            batch_metadata = metadata[i : i + batch_size]

            vectors = []
            for vec_id, emb, meta in zip(
                batch_ids, batch_embeddings, batch_metadata
            ):
                # Strip None values from metadata (Pinecone rejects them)
                clean_meta = {k: v for k, v in meta.items() if v is not None}
                vectors.append({
                    "id": vec_id,
                    "values": emb.tolist(),
                    "metadata": clean_meta,
                })

            self._index.upsert(vectors=vectors)
            total += len(vectors)

        return total

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
        filters: dict | None = None,
    ) -> list[dict]:
        # Convert filters to Pinecone format
        pinecone_filter = None
        if filters:
            pinecone_filter = {
                k: {"$eq": v} for k, v in filters.items()
            }

        query_vec = query_embedding[0].tolist()

        # Retry once on transient errors
        for attempt in range(2):
            try:
                response = self._index.query(
                    vector=query_vec,
                    top_k=top_k,
                    filter=pinecone_filter,
                    include_metadata=True,
                )
                break
            except Exception as e:
                if attempt == 0:
                    print(f"Warning: Pinecone query failed ({e}), retrying...")
                    time.sleep(2)
                else:
                    raise

        results = []
        for match in response.matches:
            results.append({
                "id": match.id,
                "score": float(match.score),
                "metadata": dict(match.metadata) if match.metadata else {},
            })

        return results

    def delete(self, ids: list[str]) -> int:
        self._index.delete(ids=ids)
        return len(ids)


def create_vector_store(provider: str) -> VectorStore:
    """Factory function to create the configured vector store backend."""
    if provider == "faiss":
        return FaissVectorStore()
    elif provider == "pinecone":
        api_key = os.environ.get("PINECONE_API_KEY")
        index_name = os.environ.get("PINECONE_INDEX_NAME", "legal-clauses")
        if not api_key:
            raise ValueError(
                "PINECONE_API_KEY environment variable is required "
                "when VECTOR_STORE_PROVIDER=pinecone"
            )
        return PineconeVectorStore(api_key=api_key, index_name=index_name)
    else:
        raise ValueError(
            f"Unknown vector store provider: {provider!r}. "
            "Supported: 'faiss', 'pinecone'"
        )
