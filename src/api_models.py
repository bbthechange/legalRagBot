"""
API request/response models for the legal RAG pipeline.

All models use Pydantic v2 for validation and serialization.
"""

from pydantic import BaseModel, Field


# --- Requests ---

class AnalyzeRequest(BaseModel):
    """Request body for clause analysis."""
    clause_text: str = Field(
        ...,
        min_length=10,
        description="The contract clause text to analyze",
        json_schema_extra={"example": "Employee agrees not to compete for 2 years worldwide."}
    )
    strategy: str = Field(
        default="few_shot",
        description="Prompt strategy to use",
        json_schema_extra={"example": "few_shot"}
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of similar clauses to retrieve"
    )


class SearchRequest(BaseModel):
    """Request body for semantic search."""
    query: str = Field(
        ...,
        min_length=3,
        description="Natural language search query"
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of results to return"
    )
    filters: dict[str, str] | None = Field(
        default=None,
        description='Optional metadata filters (e.g., {"source": "clauses_json", "risk_level": "high"})'
    )


# --- Responses ---

class SourceInfo(BaseModel):
    """A retrieved source document."""
    id: str
    title: str
    score: float
    risk_level: str = ""


class AnalyzeResponse(BaseModel):
    """Response from clause analysis."""
    analysis: dict | str
    sources: list[SourceInfo]
    strategy: str
    model: str
    top_k: int
    review_status: str = "pending_review"
    disclaimer: str


class SearchResult(BaseModel):
    """A single search result."""
    id: str
    title: str
    text: str
    score: float
    source: str = ""
    doc_type: str = ""
    risk_level: str = ""


class SearchResponse(BaseModel):
    """Response from semantic search."""
    results: list[SearchResult]
    query: str
    total_results: int


class HealthResponse(BaseModel):
    """System health status."""
    status: str  # "healthy" or "degraded"
    provider: str
    vector_store: str
    document_count: int
    available_strategies: list[str]


class ErrorResponse(BaseModel):
    """Error response body."""
    error: str
    detail: str = ""


# --- Contract Review ---

class ContractReviewRequest(BaseModel):
    """Request body for playbook-driven contract review."""
    contract_text: str = Field(
        ...,
        min_length=50,
        max_length=500_000,
        description="The full contract text to review",
    )
    playbook: str = Field(
        default="saas-vendor-review",
        pattern=r"^[a-z0-9][a-z0-9_-]*$",
        description="Playbook ID to review against (e.g., 'saas-vendor-review', 'nda-review')",
    )


class ContractReviewResponse(BaseModel):
    """Response from contract review."""
    playbook: str
    total_clauses: int
    summary: dict
    clause_analyses: list[dict]
    review_status: str = "pending_review"
    disclaimer: str
