"""
Orchestrator for running all data source ingestors.

Collects documents from all configured sources, embeds them,
and builds the vector store index.

Usage:
    python -m src.ingest.ingest_all
    python -m src.ingest.ingest_all --sources clauses_json cuad
    python -m src.ingest.ingest_all --max-cuad 100  # Limit CUAD for testing
"""

import argparse
import logging
import os
import time

from dotenv import load_dotenv

from src.embeddings import load_documents, get_embeddings
from src.logging_config import setup_logging

load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)

# Registry of available ingestors
INGESTORS = {}


def register_ingestors(sources: list[str] | None = None, max_cuad: int | None = None):
    """Build the ingestor registry based on requested sources."""
    from src.ingest.clauses_json import ClausesJsonIngestor
    from src.ingest.cuad import CuadIngestor
    from src.ingest.playbooks import PlaybookIngestor
    from src.ingest.statutes import StatuteIngestor

    available = {
        "clauses_json": lambda: ClausesJsonIngestor(),
        "cuad": lambda: CuadIngestor(max_docs=max_cuad),
        "common_paper": lambda: PlaybookIngestor(),
        "statutes": lambda: StatuteIngestor(),
    }

    if sources:
        return {k: v for k, v in available.items() if k in sources}
    return available


def main():
    parser = argparse.ArgumentParser(description="Ingest data sources into the vector store")
    parser.add_argument("--sources", nargs="+", help="Specific sources to ingest (default: all)")
    parser.add_argument("--max-cuad", type=int, default=None,
                        help="Max CUAD documents to ingest (for testing)")
    parser.add_argument("--save-index", default="data/index/main",
                        help="Path to save the FAISS index")
    args = parser.parse_args()

    ingestors = register_ingestors(sources=args.sources, max_cuad=args.max_cuad)

    # Collect documents from all sources
    all_docs = []
    for name, factory in ingestors.items():
        logger.info(f"Running ingestor: {name}")
        start = time.time()
        ingestor = factory()
        docs = ingestor.ingest()
        elapsed = time.time() - start
        logger.info(f"[{name}] Ingested {len(docs)} documents in {elapsed:.1f}s")
        all_docs.extend(docs)

    logger.info(f"Total documents across all sources: {len(all_docs)}")

    if not all_docs:
        logger.error("No documents to index!")
        return

    # Load into vector store
    logger.info("Building vector store...")
    start = time.time()
    db = load_documents(documents=all_docs)
    elapsed = time.time() - start
    logger.info(f"Vector store built in {elapsed:.1f}s with {len(all_docs)} documents")

    # Persist the index
    store_provider = os.environ.get("VECTOR_STORE_PROVIDER", "faiss")
    if store_provider == "faiss" and hasattr(db["store"], "save"):
        db["store"].save(args.save_index)
        logger.info(f"Index saved to {args.save_index}")

    # Print summary
    print(f"\n{'=' * 50}")
    print(f"  INGESTION COMPLETE")
    print(f"{'=' * 50}")
    source_counts = {}
    for doc in all_docs:
        src = doc["source"]
        source_counts[src] = source_counts.get(src, 0) + 1
    for src, count in sorted(source_counts.items()):
        print(f"  {src}: {count} documents")
    print(f"  TOTAL: {len(all_docs)} documents")
    print(f"  Upserted: {len(all_docs)} vectors")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
