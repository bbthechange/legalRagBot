# Data Ingestion

## Running Ingestion

```bash
# Activate venv first (required — Homebrew Python won't let you pip install globally)
source venv/bin/activate

# All sources
python -m src.ingest.ingest_all

# Specific sources only
python -m src.ingest.ingest_all --sources clauses_json statutes common_paper

# Limit CUAD documents (for testing / cost control)
python -m src.ingest.ingest_all --max-cuad 100
```

## Data Sources

| Source | Ingestor | Doc Count | Data Location | External Deps |
|--------|----------|-----------|---------------|---------------|
| clauses_json | `src/ingest/clauses_json.py` | ~15 | `data/clauses.json` | none |
| statutes | `src/ingest/statutes.py` | ~40 | `data/statutes/*.json` (10 states) | none |
| common_paper | `src/ingest/playbooks.py` | ~18 | `data/playbooks/*.json` (2 playbooks) | none |
| cuad | `src/ingest/cuad.py` | ~4,000-5,000 | HuggingFace download (~500MB) | `datasets` library |

## CUAD Known Issue

The `datasets` library v4+ dropped support for dataset loading scripts. The CUAD dataset (`theatticusproject/cuad-qa`) uses a legacy `cuad-qa.py` script that triggers:

```
RuntimeError: Dataset scripts are no longer supported, but found cuad-qa.py
```

**Workaround options:**
1. Pin `datasets<3` in requirements.txt
2. Skip CUAD: `--sources clauses_json statutes common_paper`
3. Rewrite the CUAD ingestor to load data without the script (e.g., direct parquet download)

## Vector Store Backends

Controlled by `VECTOR_STORE_PROVIDER` env var in `.env`.

### FAISS (local dev)
- Persists to `data/index/main.index` + `data/index/main.meta.json`
- Content hash detects stale indexes and triggers rebuild
- `--save-index` flag controls persist path

### Pinecone (production)
- Env vars: `PINECONE_API_KEY`, `PINECONE_INDEX_NAME` (default: `legal-clauses`)
- Auto-creates index on first run (serverless, AWS us-east-1, cosine, 1536 dims)
- Upserts are idempotent by doc_id — re-running overwrites, doesn't duplicate
- No persist/save step needed (cloud-hosted)
- Eventually consistent: `total_vectors` count may lag slightly after upsert

### Pinecone Free Tier Limits
- 1 index, ~2GB storage (~300k+ vectors at 1536 dims)
- Current dataset (~73 local docs) uses negligible capacity
- Even with full CUAD (~5,000 docs) well within limits
- No proactive rate limiting in upsert code, but batch size of 100 is conservative enough for free tier throughput

## Ingestion Pipeline Flow

```
register_ingestors()          # Select which sources to run
    ↓
ingestor.ingest()             # For each source: load_raw() → transform() → validate
    ↓
load_documents(all_docs)      # embeddings.py orchestrator
    ↓
create_provider()             # OpenAI / Azure / Bedrock (from LLM_PROVIDER env)
    ↓
get_embeddings(texts)         # Batched at 100 texts per API call, retry with backoff
    ↓
store.upsert(ids, embeddings, metadata)  # Batched at 100 vectors for Pinecone
    ↓
store.save() [FAISS only]     # Persist to disk
```

## Embedding Details

- Model: `text-embedding-3-small` (OpenAI default) — 1536 dimensions
- Text format: `"{title}: {text}"` per document
- Batch size: 100 texts per API call
- Retry: 3 attempts with exponential backoff (1s, 2s, 4s)
- Metadata is flattened from nested doc schema before upsert

## Environment Setup

Requires a Python virtual environment (macOS Homebrew Python is externally-managed):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Key `.env` vars for ingestion:
```
LLM_PROVIDER=openai              # openai | azure | bedrock
OPENAI_API_KEY=sk-...
VECTOR_STORE_PROVIDER=pinecone   # faiss | pinecone
PINECONE_API_KEY=pcsk_...
PINECONE_INDEX_NAME=legal-clauses
```

## Package Notes

- `pinecone-client` was renamed to `pinecone` (v5+). Use `pinecone>=5.0.0` in requirements.txt.
- `datasets` library needed only for CUAD ingestor. Other sources use stdlib only.
