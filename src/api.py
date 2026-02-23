"""
FastAPI REST API for the legal RAG pipeline.

Provides /analyze, /search, and /health endpoints with API key authentication.

Usage:
    uvicorn src.api:app --reload
    # or
    python -m src.api
"""

import logging
import os
import time

from fastapi import FastAPI, Depends, HTTPException, Security, Request
from fastapi.security import APIKeyHeader
from dotenv import load_dotenv

from src.api_models import (
    AnalyzeRequest, AnalyzeResponse,
    SearchRequest, SearchResponse, SearchResult,
    HealthResponse, ErrorResponse, SourceInfo,
)
from src.embeddings import load_clause_database
from src.rag_pipeline import analyze_clause, STRATEGIES
from src.retrieval import search_similar_clauses
from src.logging_config import setup_logging

load_dotenv()
setup_logging()
logger = logging.getLogger(__name__)

# --- App setup ---

app = FastAPI(
    title="Legal RAG API",
    description="RAG-powered legal contract analysis and knowledge base search",
    version="0.1.0",
)

# --- Auth ---

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str = Security(API_KEY_HEADER)) -> str:
    """
    Validate the API key from the X-API-Key header.

    The expected key is set via the API_KEY environment variable.
    If API_KEY is not set, auth is disabled (development mode).
    """
    expected = os.environ.get("API_KEY")
    if expected is None:
        # No API_KEY configured — allow all requests (dev mode)
        return "dev"
    if not api_key or api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


# --- Database lifecycle ---

_db: dict | None = None


def get_db() -> dict:
    """Get the loaded database, initializing on first call."""
    global _db
    if _db is None:
        logger.info("Initializing knowledge base...")
        _db = load_clause_database()
    return _db


# --- Request logging middleware ---

@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every request with timing."""
    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start
    logger.info(
        f"{request.method} {request.url.path} "
        f"status={response.status_code} "
        f"time={elapsed:.3f}s"
    )
    return response


# --- Routes ---

@app.get("/health", response_model=HealthResponse)
def health(db: dict = Depends(get_db)):
    """System health check."""
    store = db["store"]
    return HealthResponse(
        status="healthy",
        provider=db["provider"].provider_name,
        vector_store=os.environ.get("VECTOR_STORE_PROVIDER", "faiss").upper(),
        document_count=store.total_vectors,
        available_strategies=list(STRATEGIES.keys()),
    )


@app.post(
    "/analyze",
    response_model=AnalyzeResponse,
    dependencies=[Depends(verify_api_key)],
)
def analyze(req: AnalyzeRequest, db: dict = Depends(get_db)):
    """
    Analyze a contract clause using RAG.

    Retrieves similar clauses from the knowledge base and generates
    a risk analysis with citations. Output is always a draft
    requiring attorney review.
    """
    if req.strategy not in STRATEGIES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown strategy '{req.strategy}'. "
                   f"Available: {list(STRATEGIES.keys())}"
        )

    logger.info(f"Analyze request: strategy={req.strategy}, top_k={req.top_k}")

    result = analyze_clause(
        req.clause_text,
        db,
        strategy=req.strategy,
        top_k=req.top_k,
    )

    return AnalyzeResponse(
        analysis=result["analysis"],
        sources=[SourceInfo(**s) for s in result["sources"]],
        strategy=result["strategy"],
        model=result["model"],
        top_k=result["top_k"],
        review_status=result["review_status"],
        disclaimer=result["disclaimer"],
    )


@app.post(
    "/search",
    response_model=SearchResponse,
    dependencies=[Depends(verify_api_key)],
)
def search(req: SearchRequest, db: dict = Depends(get_db)):
    """
    Semantic search across the knowledge base.

    Returns the most similar documents to the query,
    optionally filtered by metadata.
    """
    logger.info(f"Search request: query='{req.query[:80]}', top_k={req.top_k}, filters={req.filters}")

    results = search_similar_clauses(
        req.query, db, top_k=req.top_k, filters=req.filters
    )

    search_results = []
    for r in results:
        clause = r["clause"]
        search_results.append(SearchResult(
            id=clause.get("id", ""),
            title=clause.get("title", ""),
            text=clause.get("text", ""),
            score=r["score"],
            source=clause.get("source", ""),
            doc_type=clause.get("doc_type", ""),
            risk_level=clause.get("risk_level", ""),
        ))

    return SearchResponse(
        results=search_results,
        query=req.query,
        total_results=len(search_results),
    )


# --- Entrypoint for python -m ---

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
