"""
Upsert legal clauses to Pinecone.

Standalone script that loads clauses from data/clauses.json, embeds them
via the configured LLM provider, and upserts to Pinecone. Idempotent —
clause IDs are used as Pinecone vector IDs, so re-running updates in place.

Usage:
    python -m scripts.upsert_to_pinecone
"""

import json
import os

from dotenv import load_dotenv

from src.embeddings import get_embeddings, infer_practice_area
from src.provider import create_provider
from src.vector_store import PineconeVectorStore

load_dotenv()


def main():
    api_key = os.environ.get("PINECONE_API_KEY")
    index_name = os.environ.get("PINECONE_INDEX_NAME", "legal-clauses")

    if not api_key:
        print("Error: PINECONE_API_KEY environment variable is required.")
        raise SystemExit(1)

    data_path = "data/clauses.json"
    with open(data_path) as f:
        clauses = json.load(f)

    print(f"Loaded {len(clauses)} clauses from {data_path}")

    provider = create_provider()
    texts_to_embed = [
        f"{clause['title']}: {clause['text']}"
        for clause in clauses
    ]

    print(f"Creating embeddings via {provider.provider_name}...")
    embeddings = get_embeddings(texts_to_embed, provider)
    print(f"Created {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")

    ids = [clause["id"] for clause in clauses]
    metadata = []
    for clause in clauses:
        metadata.append({
            "title": clause["title"],
            "type": clause["type"],
            "category": clause["category"],
            "risk_level": clause["risk_level"],
            "text": clause["text"],
            "notes": clause["notes"],
            "source": "clauses_json",
            "doc_type": "clause",
            "clause_type": clause["type"],
            "practice_area": infer_practice_area(clause["type"]),
        })

    print(f"Connecting to Pinecone index '{index_name}'...")
    store = PineconeVectorStore(api_key=api_key, index_name=index_name)

    count = store.upsert(ids, embeddings, metadata)
    print(f"Upserted {count} vectors to Pinecone index '{index_name}'")


if __name__ == "__main__":
    main()
