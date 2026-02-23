"""Shared test fixtures for the legalRag test suite."""

import json
from pathlib import Path

import numpy as np
import pytest


class MockProvider:
    """Deterministic mock LLM provider for testing."""

    provider_name = "MockProvider"
    embedding_model = "mock-embed-v1"
    chat_model = "mock-chat-v1"

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return deterministic embeddings seeded by text hash."""
        dim = 128
        embeddings = []
        for text in texts:
            seed = hash(text) % (2**32)
            rng = np.random.RandomState(seed)
            vec = rng.randn(dim).astype(np.float32)
            vec = vec / np.linalg.norm(vec)
            embeddings.append(vec)
        return np.array(embeddings, dtype=np.float32)

    def chat(self, messages: list[dict], model=None,
             temperature=0.2, max_tokens=1500) -> str:
        """Return canned structured JSON response."""
        return json.dumps({
            "assumptions": ["Mock assumption"],
            "risk_level": "medium",
            "risk_summary": "Mock risk summary",
            "key_issues": ["Mock issue 1"],
            "comparison": "Similar to [test-001]. Mock comparison.",
            "suggested_revisions": "Mock revision",
            "jurisdiction_notes": "Mock jurisdiction note",
            "sources_used": [
                {"id": "test-001", "title": "Test NDA Clause", "relevance": "Mock relevance"}
            ],
            "confidence": "medium",
            "confidence_rationale": "Mock confidence rationale",
        })


@pytest.fixture
def mock_provider():
    return MockProvider()


@pytest.fixture
def sample_clauses():
    data_path = Path(__file__).parent.parent / "data" / "clauses.json"
    with open(data_path) as f:
        return json.load(f)


@pytest.fixture
def sample_clauses_subset():
    return [
        {
            "id": "test-001",
            "type": "NDA",
            "category": "confidentiality",
            "title": "Test NDA Clause",
            "text": "Both parties agree to keep information confidential.",
            "risk_level": "low",
            "notes": "Standard test clause.",
        },
        {
            "id": "test-002",
            "type": "Employment",
            "category": "non_compete",
            "title": "Test Non-Compete",
            "text": "Employee shall not compete for one year.",
            "risk_level": "medium",
            "notes": "Reasonable restriction.",
        },
        {
            "id": "test-003",
            "type": "Service Agreement",
            "category": "liability",
            "title": "Test Liability Cap",
            "text": "Liability limited to fees paid in prior 12 months.",
            "risk_level": "low",
            "notes": "Standard cap.",
        },
    ]


@pytest.fixture
def sample_unified_documents():
    """Three documents in unified schema: clause, statute, playbook."""
    return [
        {
            "doc_id": "uni-001",
            "source": "clauses_json",
            "doc_type": "clause",
            "title": "Test NDA Clause",
            "text": "Both parties agree to keep information confidential.",
            "metadata": {
                "clause_type": "NDA",
                "category": "confidentiality",
                "risk_level": "low",
                "notes": "Standard test clause.",
                "practice_area": "intellectual_property",
            },
        },
        {
            "doc_id": "uni-002",
            "source": "statutes",
            "doc_type": "statute",
            "title": "Data Protection Act Section 5",
            "text": "Personal data shall be processed lawfully and fairly.",
            "metadata": {
                "jurisdiction": "UK",
                "citation": "DPA 2018 s.5",
            },
        },
        {
            "doc_id": "uni-003",
            "source": "common_paper",
            "doc_type": "playbook",
            "title": "Indemnification Playbook",
            "text": "Company position on mutual indemnification clauses.",
            "metadata": {
                "position": "Prefer mutual indemnification with caps.",
            },
        },
    ]


@pytest.fixture
def loaded_multi_source_db(mock_provider, sample_unified_documents):
    """Build a FAISS store with the 3 unified-schema documents."""
    from src.vector_store import FaissVectorStore
    from src.embeddings import infer_practice_area

    store = FaissVectorStore()
    docs = sample_unified_documents

    texts = [f"{d['title']}: {d['text']}" for d in docs]
    embeddings = mock_provider.embed(texts)

    ids = [d["doc_id"] for d in docs]
    metadata = []
    for doc in docs:
        flat = {
            "title": doc["title"],
            "text": doc["text"],
            "source": doc["source"],
            "doc_type": doc["doc_type"],
        }
        for k, v in doc.get("metadata", {}).items():
            if v is not None:
                flat[k] = v
        if "clause_type" in flat:
            flat["type"] = flat["clause_type"]
        metadata.append(flat)

    store.upsert(ids, embeddings, metadata)

    return {
        "store": store,
        "documents": docs,
        "clauses": docs,
        "provider": mock_provider,
    }


@pytest.fixture
def faiss_store():
    from src.vector_store import FaissVectorStore
    return FaissVectorStore()


@pytest.fixture
def loaded_faiss_db(mock_provider, sample_clauses):
    """Build a complete FAISS-backed database using mock embeddings."""
    from src.vector_store import FaissVectorStore
    from src.embeddings import infer_practice_area

    store = FaissVectorStore()

    texts = [f"{c['title']}: {c['text']}" for c in sample_clauses]
    embeddings = mock_provider.embed(texts)

    ids = [c["id"] for c in sample_clauses]
    metadata = []
    for clause in sample_clauses:
        metadata.append({
            "title": clause["title"],
            "type": clause["type"],
            "category": clause["category"],
            "risk_level": clause["risk_level"],
            "source": "clauses_json",
            "doc_type": "clause",
            "clause_type": clause["type"],
            "text": clause["text"],
            "notes": clause["notes"],
            "practice_area": infer_practice_area(clause["type"]),
        })

    store.upsert(ids, embeddings, metadata)

    return {
        "store": store,
        "clauses": sample_clauses,
        "documents": sample_clauses,
        "provider": mock_provider,
    }
