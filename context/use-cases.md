# Use Case Specifications

Three use cases to build on top of the existing RAG prototype. Each extends the current architecture (FAISS + provider abstraction + prompt strategies + evaluation framework) into something closer to what an internal law firm innovation team would ship.

## Current Architecture (What Already Exists)

```
src/
  provider.py        # Multi-cloud LLM abstraction (OpenAI, Azure, Bedrock)
  embeddings.py      # FAISS index builder, embedding via provider
  retrieval.py       # Cosine similarity search over FAISS, result formatting
  generation.py      # 3 prompt strategies (basic, structured, few_shot)
  rag_pipeline.py    # Orchestrator: retrieve → augment → generate
  evaluation.py      # Recall@k, MRR, LLM-as-Judge, strategy comparison
main.py              # Interactive CLI with sample clauses
data/clauses.json    # 15 hand-authored legal clauses with metadata
```

Key patterns already established:
- Provider interface: `provider.embed(texts) -> np.ndarray`, `provider.chat(messages) -> str`
- Database dict: `{"index": faiss.Index, "clauses": list[dict], "provider": Provider}`
- Prompt strategies registered in `STRATEGIES` dict, each returns `list[dict]` messages
- Evaluation uses ground-truth test cases + LLM-as-Judge scoring
- Temperature 0.2 for legal consistency

---

## Use Case 1: Playbook-Driven Contract Review

### What It Is

Attorneys define a "playbook" — a set of preferred positions on key contract terms (e.g., "indemnification cap should be 12 months of fees", "governing law should be our client's home state"). When a third-party contract is uploaded, the system extracts each clause, retrieves the relevant playbook position, and generates a clause-by-clause comparison with risk flags and suggested redlines.

This is different from the existing clause analyzer: instead of generic risk assessment, it compares against the firm's specific preferred positions.

### Data Sources

All publicly available, no proprietary firm data needed:

1. **CUAD (Contract Understanding Attunement Dataset)**
   - 510 commercial contracts with 13,000+ expert annotations across 41 clause types
   - Source: https://www.atticusprojectai.org/cuad (or HuggingFace: `cuad`)
   - Use for: Real contract clauses to analyze. The annotations give us ground-truth clause boundaries and types.
   - Format: PDF contracts + CSV annotations mapping clause types to text spans

2. **Common Paper (open-source legal agreements)**
   - Standard Cloud Service Agreement, NDA, Mutual NDA
   - Source: https://commonpaper.com/standards/
   - Use for: The "playbook" — these represent well-negotiated, balanced positions that a firm might adopt as their preferred starting point. Each clause in a Common Paper agreement becomes a playbook entry.

3. **ACORD forms (insurance industry standard forms)**
   - Publicly available certificate and policy forms
   - Source: https://www.acord.org/standards-architecture/acord-forms
   - Use for: Industry-specific contract templates showing domain specialization

### What to Build

#### Data Layer

**Playbook schema** (`data/playbooks/`):
```json
{
  "playbook_id": "saas-vendor-review",
  "name": "SaaS Vendor Agreement Review",
  "description": "Standard positions for reviewing third-party SaaS vendor contracts",
  "clauses": [
    {
      "clause_type": "limitation_of_liability",
      "preferred_position": "Mutual cap at 12 months of fees paid. Carve-outs for IP indemnification and data breach. No limitation on either party's liability for willful misconduct.",
      "fallback_position": "Mutual cap at total contract value. Carve-outs for IP indemnification only.",
      "walk_away": "Liability cap below 3 months of fees, or no carve-outs for data breach.",
      "risk_factors": ["cap_amount", "mutuality", "carve_outs", "consequential_damages"],
      "notes": "Always push for data breach carve-out given regulatory exposure."
    }
  ]
}
```

Build 2-3 playbooks from Common Paper standards:
- `saas-vendor-review` (from Common Paper Cloud Service Agreement)
- `nda-review` (from Common Paper NDA)
- `employment-agreement-review` (from CUAD employment contract annotations)

**Contract ingestion pipeline** (`src/ingestion.py`):
- Input: Raw contract text (paste or file upload)
- Processing: Split contract into clause-level chunks. Use an LLM call to identify clause types from the text (clause type classification). Each chunk gets a `clause_type` label.
- Output: List of `{"clause_type": str, "text": str, "position_in_doc": int}`

This is a key missing piece in the current prototype — it only handles pre-chunked clauses. Real contracts need splitting.

#### Retrieval Layer

Extend the current retrieval to support **two-stage lookup**:

1. **Playbook lookup**: Given a classified clause type, retrieve the matching playbook entry (exact match on `clause_type`, not vector search)
2. **Precedent retrieval**: Use existing FAISS search to find similar clauses from the knowledge base for additional context

The retrieval result for each clause should contain:
- The playbook position (preferred / fallback / walk-away)
- Top-k similar clauses from the knowledge base (existing functionality)
- The clause's position in the overall contract (for context)

#### Generation Layer

New prompt strategy: `playbook_review`

The prompt should instruct the LLM to:
1. Compare the clause against the playbook's preferred position
2. Identify gaps between the clause and the preferred position
3. Determine if the clause meets preferred, falls to fallback, or hits walk-away
4. Generate specific redline suggestions (exact language to add/change/remove)
5. Flag any issues not covered by the playbook

Output schema:
```json
{
  "clause_type": "limitation_of_liability",
  "extracted_text": "...",
  "playbook_match": "preferred | fallback | walk_away | not_covered",
  "gaps": [
    {
      "issue": "No data breach carve-out",
      "severity": "high",
      "playbook_says": "Require carve-out for data breach liability",
      "clause_says": "Silent on data breach"
    }
  ],
  "suggested_redline": "Add: 'Notwithstanding the foregoing, the limitations set forth in this Section shall not apply to either party's liability arising from a breach of its data protection obligations under Section X.'",
  "risk_level": "high | medium | low",
  "negotiation_notes": "This is a common vendor pushback point. Be prepared to accept a sub-cap for data breach (e.g., 2x annual fees) as fallback."
}
```

#### Full Contract Report

After processing all clauses, generate a contract-level summary:
```json
{
  "contract_summary": {
    "total_clauses_reviewed": 15,
    "preferred_match": 8,
    "fallback_match": 4,
    "walk_away_triggered": 2,
    "not_in_playbook": 1,
    "overall_risk": "high",
    "critical_issues": ["No data breach carve-out in liability cap", "Perpetual IP assignment"],
    "recommended_action": "Negotiate data breach carve-out and IP assignment scope before signing"
  },
  "clause_analyses": [...]
}
```

#### Evaluation

Extend the existing evaluation framework:
- **Clause extraction accuracy**: Does the chunker correctly identify clause boundaries and types?
- **Playbook matching accuracy**: Does the system correctly classify each clause against the playbook (preferred/fallback/walk-away)?
- **Redline quality** (LLM-as-Judge): Are suggested revisions specific, legally sound, and practically useful?

Use CUAD's annotated contracts as ground truth — they have labeled clause types and boundaries for 41 categories.

### Interview Talking Points

- Playbook-driven review is the #1 contract review workflow at large firms — every major legal AI vendor (LegalOn, Ironclad, Docusign IAM) has built this
- The playbook abstraction separates legal expertise (what the firm's positions are) from engineering (how to find and compare clauses) — practice groups can update playbooks without touching code
- Clause extraction/chunking from raw contracts is the hard engineering problem — pre-chunked data is easy, real documents need splitting strategies (heading-based, semantic, or hybrid)

---

## Use Case 2: Privacy/Data Breach Response Accelerator

### What It Is

When a data breach occurs, a privacy attorney needs to determine notification requirements across all affected jurisdictions within hours. Each US state has different breach notification statutes with different triggers, timelines, definitions of "personal information," and notification content requirements. This tool ingests those statutes via RAG, takes breach parameters as input, and generates a jurisdiction-by-jurisdiction notification matrix.

### Data Sources

1. **State breach notification statutes (public law)**
   - All 50 states + DC + territories have breach notification laws
   - Primary source: National Conference of State Legislatures (NCSL) compilation
   - Alternative: Individual state statute text from state legislature websites
   - Use for: The core knowledge base — each statute becomes a document in the RAG pipeline
   - Collect at minimum the top 10 most commercially relevant states: CA, NY, TX, FL, IL, WA, MA, CO, VA, CT

2. **Open Terms Archive**
   - Tracks changes to terms of service and privacy policies over time
   - Source: https://opentermsarchive.org/
   - Use for: Real-world privacy policy language to test extraction against. Shows how companies actually describe their data practices, which feeds into breach impact analysis.

3. **OPP-115 (Online Privacy Policies dataset)**
   - 115 website privacy policies annotated with 10 data practice categories
   - Source: https://usableprivacy.org/data
   - Use for: Training/testing the system's ability to extract data practice categories from policy text. Categories include: data collection, data sharing, data retention, user choice, etc.

### What to Build

#### Data Layer

**Statute schema** (`data/statutes/`):
```json
{
  "jurisdiction": "California",
  "statute_citation": "Cal. Civ. Code 1798.29, 1798.82",
  "effective_date": "2024-01-01",
  "personal_information_definition": [
    "First name or initial + last name combined with: SSN, driver's license, financial account number, medical info, health insurance info, unique biometric data",
    "Username or email + password or security question (added 2024)"
  ],
  "breach_definition": "Unauthorized acquisition of computerized data that compromises the security, confidentiality, or integrity of personal information",
  "encryption_safe_harbor": true,
  "encryption_safe_harbor_details": "No notification required if data was encrypted and encryption key was not compromised",
  "notification_timeline": "Expedient time and without unreasonable delay",
  "notification_timeline_days": null,
  "notification_recipients": {
    "individuals": true,
    "attorney_general": true,
    "ag_threshold": 500,
    "other_agencies": ["If health data: CDPH"]
  },
  "notification_content_requirements": [
    "Name and contact info of notifying entity",
    "Types of PI compromised",
    "Date/estimated date of breach",
    "Whether notification was delayed due to law enforcement",
    "Description of breach in general terms"
  ],
  "private_right_of_action": true,
  "penalties": "Up to $7,500 per violation under CCPA",
  "special_provisions": [
    "CCPA adds additional notification obligations for consumers",
    "Must offer identity theft prevention and mitigation services if SSN exposed"
  ]
}
```

Build structured statute entries for 10+ states. This is the knowledge base that replaces `data/clauses.json` for this use case.

**Breach parameters input schema**:
```json
{
  "data_types_compromised": ["ssn", "financial_account", "email_password"],
  "number_of_affected_individuals": 15000,
  "affected_states": ["CA", "NY", "TX", "WA"],
  "entity_type": "for_profit",
  "industry": "healthcare",
  "encryption_status": "unencrypted",
  "date_of_discovery": "2025-01-15",
  "law_enforcement_delay": false
}
```

#### Retrieval Layer

This use case needs **structured retrieval** more than pure vector search:

1. **Jurisdiction filter**: Only retrieve statutes for affected states (exact match, not semantic)
2. **Relevance ranking**: Within each state's statute, use vector search to find the most relevant provisions for the specific data types compromised
3. **Cross-reference**: For healthcare breaches, also retrieve HIPAA requirements; for financial, also retrieve GLBA

Build a new retrieval function that combines structured filtering + semantic search:
```python
def retrieve_applicable_statutes(
    breach_params: dict,
    db: dict,
    include_federal: bool = True,
) -> list[dict]:
    """
    Returns statutes filtered by jurisdiction, then ranked by relevance
    to the specific breach parameters.
    """
```

#### Generation Layer

New prompt strategy: `breach_response`

For each applicable jurisdiction, generate:
```json
{
  "jurisdiction": "California",
  "notification_required": true,
  "rationale": "SSN and financial account data are covered PI under Cal. Civ. Code 1798.82. Data was unencrypted so safe harbor does not apply.",
  "deadline": "As soon as possible; no specific day count but 'without unreasonable delay'",
  "deadline_from_discovery": "2025-01-15",
  "notify_individuals": true,
  "notify_ag": true,
  "ag_notification_details": "Required because >500 individuals affected",
  "content_requirements": ["Entity contact info", "Types of PI", "Date of breach", "..."],
  "special_considerations": [
    "Must offer identity theft prevention services (SSN exposed)",
    "CCPA may provide additional private right of action"
  ],
  "sample_notification_language": "Dear [Name], we are writing to notify you of a data security incident..."
}
```

Then generate a **cross-jurisdiction summary matrix**:
```json
{
  "breach_id": "BR-2025-001",
  "summary": {
    "total_jurisdictions": 4,
    "notifications_required": 4,
    "earliest_deadline": "Immediately (no specific day count — CA, WA)",
    "strictest_jurisdiction": "California",
    "ag_notifications_required": ["CA (>500)", "NY (>500)", "TX (>10000)", "WA (>500)"]
  },
  "timeline": [
    {"action": "Notify CA AG", "deadline": "ASAP", "status": "pending"},
    {"action": "Notify NY AG", "deadline": "ASAP", "status": "pending"},
    {"action": "Notify affected individuals (all states)", "deadline": "Within 30 days (strictest)", "status": "pending"}
  ],
  "jurisdiction_details": [...]
}
```

#### Evaluation

- **Statute coverage**: Does the system correctly identify which states require notification for a given breach scenario?
- **Deadline accuracy**: Are notification timelines correct per statute?
- **Completeness** (LLM-as-Judge): Does the output cover all required notification content?

Create 3-4 test scenarios with known correct answers:
- Simple breach (one state, one data type)
- Multi-state breach with different triggers
- Breach with encryption safe harbor
- Healthcare breach requiring both state and HIPAA notification

### Interview Talking Points

- Breach response is the most time-critical legal workflow — hours matter for compliance deadlines
- This demonstrates multi-document RAG (querying across 50+ state statutes simultaneously), structured extraction, and the ability to synthesize conflicting requirements across jurisdictions
- The structured statute schema shows how to bridge unstructured legal text and structured, queryable data — a core challenge in legal AI
- DWT has a major privacy practice — this tool would be immediately useful to their attorneys

---

## Use Case 3: Knowledge Base Search (Unified Platform)

### What It Is

A unified semantic search interface across all the data from use cases 1 and 2, plus the original clause database. An attorney types a natural language question and gets answers grounded in the firm's knowledge base with source citations.

This is the foundational RAG infrastructure — the platform that every other tool sits on top of.

### Data Sources

Everything from Use Cases 1 and 2, unified:
- Original 15 clauses from `data/clauses.json`
- Playbook definitions from Use Case 1
- Contract clauses extracted from CUAD
- State breach notification statutes from Use Case 2
- Privacy policies from OPP-115

### What to Build

#### Data Layer

**Unified document store** with a common schema:

```json
{
  "doc_id": "statute-ca-breach-001",
  "source": "breach_statutes",
  "doc_type": "statute",
  "jurisdiction": "California",
  "title": "California Data Breach Notification Law",
  "text": "...",
  "metadata": {
    "citation": "Cal. Civ. Code 1798.29",
    "effective_date": "2024-01-01",
    "last_updated": "2024-01-01",
    "practice_area": "privacy",
    "tags": ["breach_notification", "consumer_privacy", "california"]
  }
}
```

All documents — regardless of source — conform to this schema. The `source` and `doc_type` fields enable filtered search.

**Multi-collection FAISS index** (`src/vector_store.py`):
- Replace the single flat index with a collection-aware store
- Support filtering by `source`, `doc_type`, `practice_area`, `jurisdiction`
- Support index persistence (save/load to disk) so the index doesn't rebuild every session
- Support incremental updates (add new documents without rebuilding)

```python
class VectorStore:
    """
    Multi-collection vector store wrapping FAISS.

    Supports:
    - Multiple document collections (clauses, statutes, playbooks)
    - Metadata filtering before vector search
    - Index persistence to disk
    - Incremental document addition
    """

    def __init__(self, provider, index_dir: str = "data/index"):
        ...

    def add_documents(self, documents: list[dict]) -> int:
        """Add documents to the store. Returns count added."""
        ...

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,  # e.g. {"source": "breach_statutes", "jurisdiction": "CA"}
    ) -> list[dict]:
        """Semantic search with optional metadata filtering."""
        ...

    def save(self):
        """Persist index and metadata to disk."""
        ...

    def load(self):
        """Load persisted index from disk."""
        ...
```

Implementation note: FAISS doesn't natively support metadata filtering. Two approaches:
1. **Pre-filter then search**: Filter document list by metadata, build a temporary sub-index, search that. Simple but rebuilds sub-index each query.
2. **Search then post-filter**: Search the full index with a larger top-k, then filter results by metadata. Wastes some compute but avoids rebuilding. Better for production.

Go with approach 2 (post-filter) for simplicity, with an inflated `top_k` (e.g., request 5x the desired results, then filter down).

#### Retrieval Layer

**Query router** (`src/query_router.py`):

Not all questions should hit the same collection. A query router classifies the user's question and decides where to search:

```python
def route_query(query: str, provider) -> dict:
    """
    Classify a user query and determine search strategy.

    Returns:
        {
            "query_type": "contract_review | breach_response | general_legal | cross_cutting",
            "filters": {"source": "...", "doc_type": "...", ...},
            "search_strategy": "vector | structured | hybrid",
            "rewritten_query": "..." (optional, for better retrieval)
        }
    """
```

The router is a single LLM call with a classification prompt. It makes the system feel intelligent — the user doesn't need to know which collection to search.

#### Generation Layer

New prompt strategy: `knowledge_base_qa`

This is a **question-answering** prompt, not a clause analysis prompt. Key differences from existing strategies:
- Answers the user's specific question (not "analyze this clause")
- Always cites sources with document IDs and titles
- Says "I don't have information on this" when the retrieved context doesn't answer the question (reduces hallucination)
- Handles follow-up questions (maintains conversation context)

Output schema:
```json
{
  "answer": "Under California law, breach notification must be made 'in the most expedient time possible and without unreasonable delay.' There is no specific day count, unlike states such as Florida (30 days) or Colorado (30 days).",
  "sources": [
    {
      "doc_id": "statute-ca-breach-001",
      "title": "California Data Breach Notification Law",
      "relevance_score": 0.92,
      "relevant_excerpt": "..."
    }
  ],
  "confidence": "high",
  "caveats": ["This analysis is based on statute text as of 2024. Check for recent amendments."],
  "related_queries": [
    "What are the AG notification thresholds across states?",
    "Does encryption provide a safe harbor in California?"
  ]
}
```

#### CLI / Interface

Extend `main.py` with a new mode or build a separate entry point:

```
python main.py --kb

> What is the breach notification deadline in California?
[Routes to breach_statutes collection]
[Retrieves CA statute]
[Generates answer with citation]

> How does our playbook handle limitation of liability for SaaS vendors?
[Routes to playbooks collection]
[Retrieves SaaS vendor playbook entry]
[Generates answer]

> Compare California and New York breach notification requirements
[Routes to breach_statutes, filters CA + NY]
[Retrieves both statutes]
[Generates comparative analysis]
```

#### Evaluation

- **Retrieval relevance**: Does the system find the right documents for a given question?
- **Answer accuracy** (LLM-as-Judge): Is the answer factually correct based on the retrieved context?
- **Citation quality**: Does the answer cite the right sources?
- **Routing accuracy**: Does the query router send questions to the right collection?

Build test cases that span all collections:
- Contract-specific questions → should route to clauses/playbooks
- Breach-specific questions → should route to statutes
- Cross-cutting questions → should search multiple collections

### Interview Talking Points

- This is the foundational infrastructure play — every other tool is a specialized prompt strategy on top of this search platform
- The query router demonstrates agentic behavior (the system decides where to look, not the user)
- Index persistence and incremental updates are the gap between prototype and production — show you understand the operational requirements
- Metadata filtering is a real engineering decision (pre-filter vs. post-filter) with tradeoffs worth discussing

---

## Implementation Order

Build in this order because each layer depends on the previous:

### Phase 1: Knowledge Base Infrastructure (Use Case 3 — data + vector store)
1. Design the unified document schema
2. Build `VectorStore` class with persistence and metadata filtering
3. Ingest all data sources into the unified store
4. Build document ingestion scripts for each data source (CUAD, statutes, Common Paper, OPP-115)

### Phase 2: Breach Response (Use Case 2)
1. Curate 10+ state breach notification statutes into structured JSON
2. Build the structured + semantic retrieval for statutes
3. Build the `breach_response` prompt strategy
4. Build the notification matrix generator
5. Evaluation test cases

### Phase 3: Playbook Contract Review (Use Case 1)
1. Build playbooks from Common Paper standards
2. Build the contract chunking/clause extraction pipeline
3. Build playbook matching + comparison logic
4. Build the `playbook_review` prompt strategy
5. Build full contract report generation
6. Evaluation test cases

### Phase 4: Unified Search (Use Case 3 — query router + QA)
1. Build the query router
2. Build the `knowledge_base_qa` prompt strategy
3. Wire up the CLI interface
4. Evaluation test cases

### Phase 5: Polish
1. FastAPI web service wrapping the pipeline
2. Error handling, input validation, rate limiting
3. Comprehensive evaluation run across all use cases
4. Update README with architecture diagram and demo instructions

---

## Existing Code to Preserve vs. Refactor

**Preserve (still used directly):**
- `src/provider.py` — Multi-cloud abstraction. Used as-is by all use cases.
- `src/evaluation.py` — LLM-as-Judge pattern. Extended with new test cases but core logic stays.
- `src/generation.py` — Existing 3 strategies stay. New strategies added alongside them.

**Refactor:**
- `src/embeddings.py` — `load_clause_database()` is too specific. Replace with `VectorStore` class that handles any document type. Keep `get_embeddings()` and `build_faiss_index()` as internal methods of `VectorStore`.
- `src/retrieval.py` — `search_similar_clauses()` becomes a method on `VectorStore`. `format_retrieval_results()` needs to handle different document types (clauses, statutes, playbooks) with appropriate formatting.
- `src/rag_pipeline.py` — `analyze_clause()` becomes one of several pipeline functions. Add `review_contract()`, `analyze_breach()`, `search_knowledge_base()`. Consider a base pipeline class or keep as separate functions — separate functions are simpler and more explicit.

**Keep working:**
- `main.py` interactive mode should still work for the original clause analysis use case
- `data/clauses.json` stays as one data source in the unified store
- `--evaluate` and `--compare` flags should still work

---

## New Dependencies (add to requirements.txt)

```
# Web framework (Phase 5)
fastapi>=0.100.0
uvicorn>=0.20.0

# Document processing
python-docx>=0.8.11      # Word document parsing for contract ingestion
pdfplumber>=0.9.0         # PDF parsing for CUAD contracts

# Data handling
pyyaml>=6.0               # YAML config for playbooks (alternative to JSON)
```

Keep the stack minimal. No LangChain — the custom pipeline is simpler, more transparent, and demonstrates deeper understanding of the RAG architecture (good interview signal).
