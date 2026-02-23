# Legal RAG Pipeline

A Retrieval-Augmented Generation system for legal document analysis. Combines vector search across multiple legal data sources with LLM-powered analysis to help attorneys review contracts, assess breach notification requirements, and search a unified legal knowledge base.

This is a portfolio/demo project demonstrating RAG architecture patterns applied to legal use cases. It is not production legal software.

## Architecture

```
┌─────────────────────────────────────────────┐
│              API Layer (FastAPI)             │
│  /analyze  /search  /ask  /breach  /review  │
└──────────────────┬──────────────────────────┘
                   │
         ┌─────────┴─────────┐
         │   Query Router    │
         └─────────┬─────────┘
                   │
    ┌──────────────┼──────────────┐
    │              │              │
┌───┴───┐   ┌─────┴────┐  ┌─────┴─────┐
│Clause │   │ Breach   │  │ Playbook  │
│Analysis│  │ Response │  │ Review    │
└───┬───┘   └─────┬────┘  └─────┬─────┘
    │              │              │
    └──────────────┼──────────────┘
                   │
         ┌─────────┴─────────┐
         │  Retrieval Layer  │
         │  (Vector Search   │
         │   + Filtering)    │
         └─────────┬─────────┘
                   │
         ┌─────────┴─────────┐
         │   Vector Store    │
         │  (FAISS/Pinecone) │
         └─────────┬─────────┘
                   │
    ┌──────────────┼──────────────┐
    │              │              │
┌───┴────┐  ┌─────┴─────┐ ┌─────┴──────┐
│Clauses │  │ Statutes  │ │ Playbooks  │
│ (CUAD) │  │ (10 states│ │(Comm Paper)│
└────────┘  └───────────┘ └────────────┘
```

## Use Cases

1. **Clause Risk Analysis** — Submit a contract clause, get risk assessment with citations to similar clauses from the knowledge base
2. **Breach Notification Response** — Given breach parameters (data types, affected states), get jurisdiction-specific notification requirements
3. **Playbook-Driven Contract Review** — Review a full contract against firm playbook positions (preferred terms, fallbacks, walk-away triggers)
4. **Unified Knowledge Base Search** — Ask natural language questions across all data sources with intelligent query routing

## Quick Start

```bash
# Clone and set up
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Add your OpenAI API key
echo "OPENAI_API_KEY=sk-your-key" > .env

# Ingest all data sources
python -m src.ingest.ingest_all

# Interactive clause analysis
python main.py

# Knowledge base search
python main.py --kb

# Breach notification analysis
python main.py --breach

# Playbook contract review
python main.py --review

# Run evaluation suite
python main.py --evaluate

# Compare prompt strategies
python main.py --compare

# Start the API server
uvicorn src.api:app --reload

# Run the scripted demo
python -m scripts.demo
```

## API Reference

All endpoints (except `/health`) require an `X-API-Key` header. Set the `API_KEY` environment variable; if unset, auth is disabled (dev mode).

### `GET /health`
System health check. Returns provider info, document count, and available strategies.

### `POST /analyze`
Analyze a contract clause using RAG.
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"clause_text": "Employee agrees not to compete for 2 years worldwide.", "strategy": "few_shot", "top_k": 3}'
```

### `POST /search`
Semantic search across the knowledge base.
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"query": "non-compete clause", "top_k": 5}'
```

### `POST /ask`
Ask a natural language question across the full knowledge base. Uses the query router to intelligently select data sources.
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"query": "What is a reasonable non-compete duration?", "top_k": 5, "use_router": true}'
```

### `POST /breach-analysis`
Analyze breach notification requirements across jurisdictions.
```bash
curl -X POST http://localhost:8000/breach-analysis \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"data_types_compromised": ["ssn", "email"], "affected_states": ["CA", "NY"], "encryption_status": "unencrypted"}'
```

### `POST /contract-review`
Review a contract against a firm playbook.
```bash
curl -X POST http://localhost:8000/contract-review \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"contract_text": "...", "playbook": "saas-vendor-review"}'
```

## Data Sources

| Source | Description | Documents |
|--------|-------------|-----------|
| `clauses.json` | Hand-curated contract clauses with risk assessments | ~20 clauses across NDA, employment, service agreement types |
| CUAD | Contract Understanding Atticus Dataset — real contract clauses | Subset of labeled contract clauses |
| State Statutes | Data breach notification laws | 10 states (CA, NY, TX, FL, IL, WA, MA, CO, VA, CT) |
| Common Paper Playbooks | Firm negotiation positions | SaaS vendor review, NDA review playbooks |

## Evaluation

The evaluation framework measures both retrieval and generation quality:

**Retrieval Metrics:**
- Recall@k — Fraction of relevant clauses found in top-k results
- Mean Reciprocal Rank (MRR) — Ranking quality of relevant results

**Generation Metrics (LLM-as-Judge):**
- Risk Accuracy — Correct risk level identification
- Issue Coverage — Identification of required key issues
- Actionability — Specificity and practicality of suggested revisions
- Grounding — References to retrieved knowledge base clauses

Three prompt strategies are compared: basic (baseline), structured (JSON schema + guidelines), and few-shot (worked examples). Few-shot consistently outperforms.

## Tech Stack

- **Python** — Core language
- **FAISS** — Vector similarity search (Meta's library), with optional Pinecone support
- **OpenAI API** — Embeddings (text-embedding-3-small) and generation (GPT-4o-mini)
- **FastAPI** — REST API layer with Pydantic validation
- **RAG Architecture** — Retrieval-augmented generation with query routing
- **pytest** — Test suite

## Key Modules

| Module | Purpose |
|--------|---------|
| `src/embeddings.py` | Vector embeddings + FAISS indexing |
| `src/retrieval.py` | Semantic similarity search with metadata filtering |
| `src/generation.py` | LLM prompt strategies (basic, structured, few-shot, KB QA) |
| `src/rag_pipeline.py` | End-to-end RAG orchestration |
| `src/query_router.py` | LLM-based query classification and routing |
| `src/kb_search.py` | Unified knowledge base search pipeline |
| `src/breach_analysis.py` | Breach notification response generation |
| `src/playbook_review.py` | Playbook-driven contract review |
| `src/evaluation.py` | Retrieval metrics + LLM-as-Judge |
| `src/api.py` | FastAPI REST endpoints |
| `src/ingest/` | Data ingestion pipelines for all sources |
