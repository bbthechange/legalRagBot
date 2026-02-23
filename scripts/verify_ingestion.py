"""
Verify that ingested data is searchable.

Usage:
    python -m scripts.verify_ingestion
"""

from dotenv import load_dotenv
load_dotenv()

from src.embeddings import load_documents
from src.retrieval import search_similar_clauses

# Load from persisted index
db = load_documents(data_path="data/clauses.json")

# Try to load persisted index if available
store_provider = __import__('os').environ.get("VECTOR_STORE_PROVIDER", "faiss")
if store_provider == "faiss" and hasattr(db["store"], "load"):
    if db["store"].load("data/index/main"):
        print(f"Loaded persisted index: {db['store'].total_vectors} vectors")

# Test queries
queries = [
    ("limitation of liability clause", None),
    ("non-compete agreement restrictions", None),
    ("data breach notification requirements", None),
    ("indemnification obligations", {"source": "cuad"}),
]

for query, filters in queries:
    print(f"\nQuery: '{query}' (filters: {filters})")
    results = search_similar_clauses(query, db, top_k=3, filters=filters)
    for i, r in enumerate(results, 1):
        clause = r["clause"]
        print(f"  {i}. [{clause['id'][:40]}] {clause['title'][:60]} "
              f"(score: {r['score']:.3f}, source: {clause.get('source', 'N/A')})")
