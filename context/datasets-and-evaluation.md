# Datasets & Evaluation Integration

Context for building agents that integrate real legal datasets and LegalBench-RAG evaluation into the existing RAG prototype. See `context/use-cases.md` for the three use cases these datasets serve.

## Current State

The prototype uses a single hand-authored file: `data/clauses.json` with 15 legal clauses. Each clause has `id`, `type`, `category`, `title`, `text`, `risk_level`, and `notes`. The evaluation in `src/evaluation.py` has 4 hardcoded `TEST_CASES` with expected retrieval IDs and LLM-as-Judge scoring.

This works for the demo but doesn't show the ability to work with real-world legal data at scale.

---

## Datasets to Integrate

### 1. CUAD (Contract Understanding Atticus Dataset)

**What:** 510 real commercial contracts (from SEC EDGAR filings) with 13,000+ expert annotations across 41 clause types. Annotated by law students and lawyers.

**Serves:** Use Case 1 (Playbook Contract Review) — provides real contracts to review and ground-truth clause type labels for evaluating clause extraction accuracy.

**Where to get it:**
- HuggingFace: `datasets` library, dataset ID `theatticusproject/cuad`
- Direct: https://www.atticusprojectai.org/cuad
- GitHub: https://github.com/TheAtticusProject/cuad

**Format:**
- The HuggingFace version is in SQuAD-style QA format: each row has a `context` (contract text), `question` (e.g., "Highlight the parts that discuss Limitation of Liability"), and `answers` with start positions and text spans.
- The 41 clause categories include: Parties, Agreement Date, Effective Date, Expiration Date, Renewal Term, Notice Period, Governing Law, Most Favored Nation, Non-Compete, Exclusivity, No-Solicit, Termination for Convenience, Change of Control, Anti-Assignment, Revenue/Profit Sharing, IP Ownership Assignment, License Grant, Non-Transferable License, Cap on Liability, Indemnification, Insurance, Minimum Commitment, Volume Restriction, Warranty Duration, Uncapped Liability, Audit Rights, and more.

**Ingestion approach:**
```python
# pip install datasets
from datasets import load_dataset

cuad = load_dataset("theatticusproject/cuad", split="test")

# Each example has:
# - "context": full contract text (or large section)
# - "question": clause type query (e.g., "Highlight the parts (if any) related to...")
# - "answers": {"text": [...], "answer_start": [...]}
# - "title": contract filename
# - "id": unique identifier

# To extract clause instances grouped by type:
# 1. Group by question (= clause type)
# 2. For each, extract the answer spans as individual clause chunks
# 3. Store with clause_type, source_contract, and text
```

**What to extract for our pipeline:**
- Individual clause text spans (the `answers.text` values) become documents in the vector store
- The clause type (derived from the `question` field) becomes the `doc_type` / `clause_type` metadata
- The contract title becomes a `source` metadata field
- Clauses with empty answers (no annotation) are also useful — they indicate that clause type is absent from the contract, which is relevant for playbook gap analysis

**Volume:** Expect ~4,000-5,000 non-empty clause annotations across the 510 contracts after filtering. Not all contracts have all 41 clause types.

**New dependency:** `datasets` (HuggingFace datasets library). Add to `requirements.txt`:
```
datasets>=2.14.0
```

---

### 2. Common Paper Standard Contracts

**What:** 8 standardized, open-source commercial contracts: NDA, Cloud Service Agreement, Mutual NDA, Design Partner Agreement, DPA, Partner Program, Professional Services, SLA.

**Serves:** Use Case 1 (Playbook Contract Review) — these become the "playbook" representing well-negotiated, balanced positions. Each clause in a Common Paper agreement represents what the firm considers its preferred starting position.

**Where to get it:**
- https://commonpaper.com/standards/
- Available as PDF, DOCX, HTML, and Markdown
- Licensed CC BY 4.0

**Ingestion approach:**
- Download the Markdown versions (easiest to parse)
- Split each agreement into clause-level chunks by heading
- Each chunk becomes a playbook entry with `clause_type`, `preferred_position` (the Common Paper language), and metadata

**What to extract for our pipeline:**
- For the Cloud Service Agreement: extract clauses for limitation of liability, indemnification, data protection, termination, IP, warranties, governing law, etc.
- For the NDA: confidentiality scope, duration, exclusions, remedies
- Map each extracted clause to a `clause_type` that aligns with CUAD's 41 categories where possible

**Format for playbook entries** (see `context/use-cases.md` for full schema):
- `preferred_position`: The Common Paper language verbatim
- `fallback_position`: A relaxed version (generate via LLM or author manually for key clauses)
- `walk_away`: The minimum acceptable position (same — generate or author)

**No new dependencies needed.** Markdown parsing can use Python stdlib or simple string splitting on headings.

---

### 3. State Breach Notification Statutes

**What:** Each US state (plus DC and territories) has enacted data breach notification laws. These are public law, freely available from state legislature websites.

**Serves:** Use Case 2 (Privacy/Breach Response Accelerator) — the core knowledge base for breach response queries.

**Where to get it:**
- National Conference of State Legislatures (NCSL): https://www.ncsl.org/technology-and-communication/security-breach-notification-laws
- Individual state legislature websites for full statute text
- DLA Piper Global Data Protection Laws: https://www.dlapiperdataprotection.com/ (comparative reference)

**Priority states (build these first):**
1. California (Cal. Civ. Code 1798.29, 1798.82) — broadest, most referenced
2. New York (Gen. Bus. Law 899-aa) — major commercial hub
3. Texas (Bus. & Com. Code 521.053) — large state, specific timeline
4. Florida (501.171) — 30-day hard deadline
5. Illinois (815 ILCS 530/) — BIPA state, aggressive enforcement
6. Washington (RCW 19.255.010) — DWT's home state
7. Massachusetts (201 CMR 17.00) — prescriptive security requirements
8. Colorado (CRS 6-1-716) — 30-day deadline, recent updates
9. Virginia (Va. Code 18.2-186.6) — CDPA state
10. Connecticut (CGS 36a-701b) — recent comprehensive update

**Ingestion approach:**
- Manually structure each statute into the JSON schema defined in `context/use-cases.md` (fields: jurisdiction, citation, PI definition, breach definition, encryption safe harbor, notification timeline, recipients, content requirements, penalties, special provisions)
- Also store the full statute text as a separate document for vector search (the structured JSON is for deterministic lookups, the full text is for semantic queries)
- Each state gets two entries in the vector store: one structured (for filtered retrieval), one full-text (for semantic search)

**This is manual work.** Unlike CUAD (downloadable dataset), statute curation requires reading each statute and extracting the structured fields. Budget 1-2 hours per state for careful extraction. The NCSL compilation page provides a useful starting summary, but the actual statute text should come from official sources.

**No new dependencies needed.** The statutes are structured manually into JSON.

---

### 4. Open Terms Archive

**What:** A French NGO that tracks changes to Terms of Service and Privacy Policies from major platforms in near real-time. Policies are scraped multiple times daily, all versions stored with diffs in git repositories.

**Serves:** Use Case 2 (cross-referencing company privacy commitments against breach statute requirements) and Use Case 3 (searchable privacy policy knowledge base).

**Where to get it:**
- Main site: https://opentermsarchive.org/
- PGA (Platform Governance Archive) versions repo: https://github.com/OpenTermsArchive/pga-versions
- Documentation: https://docs.opentermsarchive.org/
- Datasets page: https://opentermsarchive.org/en/datasets/

**Format:** Git repos where each tracked service has a directory with Markdown files for each document type (Terms of Service, Privacy Policy, Community Guidelines, etc.). Git history contains all version changes.

**Ingestion approach:**
```bash
# Clone the versions repo (contains latest versions of all tracked documents)
git clone https://github.com/OpenTermsArchive/pga-versions.git data/external/open-terms-archive
```
```python
# Walk the directory structure to find privacy policies
# Each service has: ServiceName/Privacy Policy.md, ServiceName/Terms of Service.md, etc.
# Parse each Markdown file into chunks by section heading
# Store with metadata: service_name, document_type, section, last_updated (from git log)
```

**What to extract for our pipeline:**
- Privacy Policy documents from major tech services (Google, Meta, Amazon, Apple, Microsoft, etc.)
- Terms of Service documents
- Chunk by section heading (each H2 or H3 becomes a document in the vector store)
- Tag with service name and section topic (data collection, data sharing, data retention, user rights, etc.)

**Priority:** Focus on 10-15 major services to keep scope manageable. The full archive covers hundreds of services.

**No new dependencies needed.** Git clone + Python file walking + Markdown splitting.

---

### 5. OPP-115 (Online Privacy Policies)

**What:** 115 website privacy policies with 23,000+ fine-grained annotations across 10 data practice categories: First Party Collection, Third Party Sharing, User Choice/Control, User Access/Edit/Deletion, Data Retention, Data Security, Policy Change, Do Not Track, International/Specific Audiences, Other.

**Serves:** Use Case 2 (training/testing extraction of data practice categories) and Use Case 3 (annotated privacy content for knowledge base).

**Where to get it:**
- https://usableprivacy.org/data
- HuggingFace: `alzoubi36/opp_115`

**Format:** The HuggingFace version has columns for policy text segments and category labels. The original download has HTML files of the policies plus CSV annotation files.

**Ingestion approach:**
```python
from datasets import load_dataset

opp = load_dataset("alzoubi36/opp_115")

# Each row has:
# - text segment from a privacy policy
# - annotation category (one of 10 data practice categories)
# - fine-grained attributes within each category
```

**What to extract for our pipeline:**
- Pre-segmented, labeled privacy policy text — each segment becomes a document with its data practice category as metadata
- The category labels enable structured filtering: "show me all Third Party Sharing clauses across policies"
- Useful for testing whether the breach response pipeline correctly identifies what data practices a company has committed to

**Uses the same `datasets` dependency as CUAD.** No additional dependencies.

---

## Evaluation: LegalBench-RAG

### What It Is

LegalBench-RAG is a purpose-built benchmark for evaluating RAG systems on legal documents. It contains **6,858 query-answer pairs** over a corpus of **79M+ characters**, entirely human-annotated by legal experts. It was built from four source datasets: PrivacyQA, CUAD, MAUD (M&A agreements), and ContractNLI.

This directly replaces the 4 hand-rolled test cases in `src/evaluation.py` with an industry-standard benchmark.

### Where to Get It

- GitHub: https://github.com/zeroentropy-ai/legalbenchrag
- Paper: https://arxiv.org/abs/2408.10343
- Mini version (776 queries): included in the repo for rapid iteration

### Format

The repo contains:
- `corpus/` — the document corpus (text files organized by source dataset)
- `questions/` — JSON files with query-answer pairs
- Each query has: `question`, `answer`, `source_file`, `answer_location` (character offsets into the source document)

```json
{
  "question": "What is the limitation of liability under this agreement?",
  "answer": "The total aggregate liability shall not exceed...",
  "source_file": "cuad/contract_123.txt",
  "answer_location": {"start": 4521, "end": 4789}
}
```

### How It Fits Into Our Pipeline

The existing evaluation framework in `src/evaluation.py` already has the right structure:
- `evaluate_retrieval()` measures Recall@k and MRR against expected results
- `evaluate_generation()` uses LLM-as-Judge scoring
- `compare_strategies()` benchmarks across prompt strategies

**What changes:**

1. **Replace `TEST_CASES`** with LegalBench-RAG queries loaded from disk instead of hardcoded
2. **Add retrieval ground truth**: LegalBench-RAG provides `source_file` and `answer_location`, so we know exactly which document and which span the retrieval should find. This gives us real Recall@k measurement instead of checking against hand-picked IDs.
3. **Add answer accuracy metric**: Compare the generated answer against the gold answer from the benchmark. This can be done with:
   - Exact match (strict)
   - Token-level F1 (standard for QA benchmarks)
   - LLM-as-Judge (already implemented, extend to compare against gold answer)
4. **Segment evaluation by source**: Since LegalBench-RAG spans CUAD, MAUD, PrivacyQA, and ContractNLI, we can report metrics per source — showing how well our pipeline handles contracts vs. privacy policies vs. M&A agreements.

### Evaluation Architecture

```
src/evaluation.py (refactored)
  |
  |-- load_benchmark(name="legalbench-rag-mini")  # Load queries from disk
  |-- evaluate_retrieval(db, benchmark)             # Recall@k, MRR against gold docs
  |-- evaluate_generation(db, benchmark, strategy)  # LLM-as-Judge + gold answer comparison
  |-- evaluate_answer_accuracy(predicted, gold)      # Token F1 + exact match
  |-- compare_strategies(db, benchmark)              # Existing comparison, now against benchmark
  |-- report_by_source(results)                      # Break down metrics by source dataset
```

### Practical Considerations

- **Start with the mini version** (776 queries). The full 6,858-query benchmark is expensive to run against the OpenAI API (embedding + generation for each query). The mini version is representative and fast.
- **Corpus ingestion**: The LegalBench-RAG corpus needs to be ingested into our vector store alongside our other data. The corpus documents map to the unified document schema with `source: "legalbench"` and `doc_type` based on the subdirectory (cuad, maud, privacyqa, contractnli).
- **Separate eval index vs. main index**: Consider building a dedicated eval index from just the LegalBench-RAG corpus to get clean benchmark numbers. Then separately test retrieval against the full mixed index (all our data sources) to measure real-world performance.
- **Cost estimate**: At ~$0.02/1K tokens for gpt-4o-mini, the mini benchmark (~776 queries with retrieval + generation + judge) should cost roughly $2-5 per full eval run.

### What This Replaces vs. Extends

| Current | After Integration |
|---------|-------------------|
| 4 hardcoded test cases | 776+ benchmark queries (mini) or 6,858 (full) |
| Hand-picked expected retrieval IDs | Gold-standard source documents with character offsets |
| LLM-as-Judge only | LLM-as-Judge + token F1 + exact match against gold answers |
| Single eval run | Per-source breakdown (contracts, privacy, M&A) |
| Custom test clauses | Real legal questions written by legal experts |

The existing LLM-as-Judge pattern and strategy comparison framework stay. They just get fed real benchmark data instead of synthetic test cases.

### New Dependency

```
# Add to requirements.txt
datasets>=2.14.0    # HuggingFace datasets (also used for CUAD, OPP-115)
```

The LegalBench-RAG repo itself is cloned into `data/external/`:
```bash
git clone https://github.com/zeroentropy-ai/legalbenchrag.git data/external/legalbench-rag
```

---

## Data Directory Structure (Target State)

```
data/
  clauses.json                          # Original 15 clauses (preserved)
  playbooks/
    saas-vendor-review.json             # From Common Paper Cloud Service Agreement
    nda-review.json                     # From Common Paper NDA
    employment-review.json              # From CUAD employment contract annotations
  statutes/
    california.json                     # Structured breach notification statute
    new_york.json
    texas.json
    ... (10+ states)
  external/                             # Git-cloned external datasets (gitignored)
    legalbench-rag/                     # Cloned benchmark repo
    open-terms-archive/                 # Cloned OTA versions repo
  index/                                # Persisted FAISS indexes (gitignored)
    main.index                          # Full vector index
    main.metadata.json                  # Document metadata for post-filter search
    eval.index                          # Eval-only index (LegalBench-RAG corpus)
    eval.metadata.json
```

Add to `.gitignore`:
```
data/external/
data/index/
```

---

## Ingestion Scripts

Build a script per data source in `src/ingest/`:

```
src/ingest/
  __init__.py
  cuad.py               # Download + parse CUAD from HuggingFace → unified schema
  common_paper.py       # Parse Common Paper Markdown → playbook JSON
  statutes.py           # Validate + load structured statute JSON
  opp115.py             # Download + parse OPP-115 from HuggingFace → unified schema
  open_terms_archive.py # Walk cloned repo → chunked privacy policies
  legalbench.py         # Parse LegalBench-RAG corpus + queries → eval format
  ingest_all.py         # Orchestrator: run all ingestors, build unified index
```

Each ingestor outputs documents in the unified schema:
```python
{
    "doc_id": str,          # Globally unique
    "source": str,          # "cuad" | "common_paper" | "statutes" | "opp115" | "open_terms_archive"
    "doc_type": str,        # "clause" | "statute" | "playbook" | "privacy_policy" | "terms_of_service"
    "title": str,
    "text": str,            # The content to embed
    "metadata": {
        "clause_type": str | None,      # For CUAD clauses
        "jurisdiction": str | None,      # For statutes
        "service_name": str | None,      # For OTA policies
        "practice_area": str | None,     # "contracts" | "privacy" | "employment"
        "risk_level": str | None,        # If available from source
        "citation": str | None,          # For statutes
        "source_contract": str | None,   # For CUAD (which contract it came from)
        "data_practice_category": str | None,  # For OPP-115
    }
}
```

---

## Integration with Existing Code

### What stays the same
- `src/provider.py` — unchanged, all datasets use the same provider for embedding/chat
- `src/generation.py` — existing 3 strategies preserved, new strategies added alongside
- `main.py` — original interactive mode still works

### What gets refactored
- `src/embeddings.py` — `load_clause_database()` replaced by `VectorStore` class (see `context/use-cases.md`). `get_embeddings()` and `build_faiss_index()` become internal to `VectorStore`.
- `src/retrieval.py` — `search_similar_clauses()` becomes `VectorStore.search()`. `format_retrieval_results()` updated to handle different doc types.
- `src/evaluation.py` — `TEST_CASES` replaced by benchmark loader. Core `evaluate_retrieval()` and `evaluate_generation()` logic preserved but parameterized to accept external test data. `compare_strategies()` extended with per-source reporting.

### New files
- `src/vector_store.py` — multi-collection FAISS wrapper with persistence and metadata filtering
- `src/ingest/*.py` — per-source ingestion scripts
- `src/query_router.py` — classifies user queries to determine search filters
- New prompt strategies in `src/generation.py` — `playbook_review`, `breach_response`, `knowledge_base_qa`

---

## Implementation Priority for Datasets

**Phase 1 (do first — unblocks everything else):**
1. Build `VectorStore` class
2. Ingest CUAD (provides bulk contract data, unblocks Use Case 1 + eval)
3. Clone and ingest LegalBench-RAG mini (unblocks evaluation against real benchmark)

**Phase 2 (do alongside Use Case 2):**
4. Curate 10 state breach notification statutes (manual work, unblocks Use Case 2)
5. Clone and ingest Open Terms Archive (supplements Use Case 2)
6. Ingest OPP-115 (supplements Use Case 2 + 3)

**Phase 3 (do alongside Use Case 1):**
7. Download and parse Common Paper contracts into playbook format (unblocks Use Case 1 playbook comparison)

**Phase 4 (evaluation polish):**
8. Run full LegalBench-RAG evaluation, report per-source metrics
9. Compare evaluation results against current hand-rolled test cases
10. Document benchmark results in README
