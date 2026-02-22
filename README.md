# Legal Contract Clause Analyzer

A RAG (Retrieval-Augmented Generation) system that helps lawyers review contract clauses by finding similar clauses from a knowledge base and generating risk analysis with actionable recommendations.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
│  User Input  │────▶│  Embeddings  │────▶│  FAISS Vector   │────▶│  Retrieval   │
│  (clause)    │     │  (OpenAI)    │     │  Search         │     │  (top-k)     │
└─────────────┘     └──────────────┘     └─────────────────┘     └──────┬───────┘
                                                                        │
                    ┌──────────────┐     ┌─────────────────┐           │
                    │  Analysis    │◀────│  LLM Generation │◀──────────┘
                    │  (JSON)      │     │  (GPT-4o-mini)  │   + retrieved context
                    └──────────────┘     └─────────────────┘
```

## Key Components

| Module | Purpose | Key Concepts |
|--------|---------|-------------|
| `src/embeddings.py` | Vector embeddings + FAISS indexing | OpenAI embeddings API, cosine similarity, L2 normalization |
| `src/retrieval.py` | Semantic similarity search | top-k retrieval, score thresholds, metadata formatting |
| `src/generation.py` | LLM analysis with 3 prompt strategies | System prompts, few-shot learning, structured output, temperature tuning |
| `src/rag_pipeline.py` | End-to-end RAG orchestration | Retrieve → Augment → Generate pattern |
| `src/evaluation.py` | Retrieval metrics + LLM-as-Judge | Recall@k, MRR, multi-dimensional scoring, strategy comparison |

## Prompt Strategies

Three strategies are implemented and empirically compared:

- **Basic** — Minimal prompt (baseline for comparison)
- **Structured** — Detailed system prompt with JSON schema and behavioral guidelines
- **Few-Shot** — Includes worked examples of ideal analysis output

Evaluation results show few-shot consistently outperforms the other strategies, particularly on issue coverage.

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

## Usage

```bash
# Set up
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Add your OpenAI API key
echo "OPENAI_API_KEY=sk-your-key" > .env

# Interactive clause analysis
python main.py

# Run evaluation suite
python main.py --evaluate

# Compare prompt strategies
python main.py --compare
```

## Tech Stack

- **Python** — Core language
- **FAISS** — Vector similarity search (Meta's library)
- **OpenAI API** — Embeddings (text-embedding-3-small) and generation (GPT-4o-mini)
- **RAG Architecture** — Retrieval-augmented generation pattern
