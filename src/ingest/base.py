"""
Base class for data source ingestors.

Each ingestor is responsible for:
1. Loading data from its source (file, API, HuggingFace, etc.)
2. Converting it into the unified document schema
3. Returning a list of validated documents
"""

import logging
from abc import ABC, abstractmethod

from src.schemas import validate_document, normalize_document

logger = logging.getLogger(__name__)


class BaseIngestor(ABC):
    """Base class for all data source ingestors."""

    source_name: str = ""  # Must be set by subclass

    @abstractmethod
    def load_raw(self) -> list[dict]:
        """
        Load raw data from the source.
        Returns a list of dicts in whatever format the source provides.
        """
        ...

    @abstractmethod
    def transform(self, raw_data: list[dict]) -> list[dict]:
        """
        Transform raw data into unified document schema.
        Returns a list of dicts conforming to the schema in src/schemas.py.
        """
        ...

    def ingest(self) -> list[dict]:
        """
        Full ingest pipeline: load → transform → validate → normalize.

        Returns validated, normalized documents ready for the vector store.
        """
        logger.info(f"[{self.source_name}] Loading raw data...")
        raw = self.load_raw()
        logger.info(f"[{self.source_name}] Loaded {len(raw)} raw records")

        logger.info(f"[{self.source_name}] Transforming to unified schema...")
        docs = self.transform(raw)
        logger.info(f"[{self.source_name}] Transformed into {len(docs)} documents")

        # Validate and filter
        valid_docs = []
        error_count = 0
        for doc in docs:
            errors = validate_document(doc)
            if errors:
                error_count += 1
                if error_count <= 5:  # Log first 5 errors only
                    logger.warning(f"[{self.source_name}] Invalid doc {doc.get('doc_id', '?')}: {errors}")
            else:
                valid_docs.append(normalize_document(doc))

        if error_count > 0:
            logger.warning(f"[{self.source_name}] {error_count} invalid documents skipped")
        logger.info(f"[{self.source_name}] {len(valid_docs)} valid documents ready for indexing")
        return valid_docs
