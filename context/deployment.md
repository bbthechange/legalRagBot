# Deployment Architecture

Single-service deployment: FastAPI serves both the API and the Angular static build from one container.

## How It Works

```
Browser → Render (legalrag.onrender.com)
           ├── /api/*  → strip prefix → FastAPI routes (/health, /analyze, etc.)
           └── /*      → static files (Angular SPA) with index.html fallback
```

- `src/api.py` has a middleware that strips the `/api` prefix from incoming requests so the frontend's `/api/health` reaches the `/health` route
- A catch-all route serves Angular static files, but **only activates when `/app/static` directory exists** (i.e., inside the Docker container). In local dev, this directory doesn't exist, so the catch-all is never registered
- Local dev is completely unaffected — Angular CLI proxy (`proxy.conf.json`) handles `/api` → `localhost:8000` as before

## Docker Build (multi-stage)

1. **Stage 1** (Node 20): `npm ci` + `ng build` → produces static files at `frontend/dist/frontend/browser/`
2. **Stage 2** (Python 3.12): installs `requirements-prod.txt`, copies backend + data + static files, runs uvicorn

Python 3.12 is used in Docker (not 3.14 which is used locally) because dependency wheels are more reliably available.

## Production Requirements

`requirements-prod.txt` is a subset of `requirements.txt`, excluding:
- `faiss-cpu` — not needed when using Pinecone (faiss is only imported inside method bodies, not at module level)
- `boto3` — only needed for AWS Bedrock provider
- `datasets` — only used for offline CUAD ingestion
- `pytest`, `pytest-mock` — test dependencies

## Render Configuration

`render.yaml` defines the web service. Environment variables that must be set in the Render dashboard (not in the file):
- `OPENAI_API_KEY`
- `PINECONE_API_KEY`

Static config in `render.yaml`:
- `VECTOR_STORE_PROVIDER=pinecone`
- `LLM_PROVIDER=openai`
- `PINECONE_INDEX_NAME=legal-clauses`
- Health check path: `/api/health`

## Startup Behavior

On first request, `load_clause_database()` runs lazily:
1. Connects to Pinecone (existing cloud index)
2. Loads `data/clauses.json` (~20 documents)
3. Embeds all documents via OpenAI and upserts to Pinecone

This takes ~5-10 seconds on cold start. Subsequent requests are fast.

## Suspend/Resume

The Render service can be suspended from the dashboard to stop billing. Resuming restores the same URL. The service is on the Starter plan (~$7/mo, pro-rated).

## Key Constraints

- **No auth in production currently** — `API_KEY` env var is not set, so all endpoints are open. This is acceptable since the URL is only shared with specific people.
- **CORS is irrelevant** in the single-service setup (same origin), but the middleware remains for local dev compatibility.
- Secrets (API keys) are configured via Render's dashboard, never committed to git. The `.env` file is gitignored.
