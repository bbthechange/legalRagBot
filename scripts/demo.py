"""
Scripted demo showing all three use cases.

Usage:
    python -m scripts.demo
"""

from dotenv import load_dotenv
load_dotenv()

from src.embeddings import load_clause_database
from src.rag_pipeline import analyze_clause
from src.kb_search import search_knowledge_base


def main():
    print("Loading knowledge base...")
    db = load_clause_database()

    print("\n" + "=" * 60)
    print("  DEMO: Legal RAG Pipeline")
    print("=" * 60)

    # Demo 1: Clause Analysis
    print("\n\n--- DEMO 1: Clause Risk Analysis ---")
    clause = """The Service Provider shall bear no responsibility for any loss,
    damage, or injury arising from the services provided, including but not
    limited to loss of data, business interruption, or financial losses."""
    result = analyze_clause(clause, db)
    print(f"Analysis: {result['analysis']}")
    print(f"Sources: {[s['id'] for s in result['sources']]}")
    print(f"Status: {result['review_status']}")

    # Demo 2: Knowledge Base Q&A
    print("\n\n--- DEMO 2: Knowledge Base Search ---")
    questions = [
        "What is a reasonable non-compete duration?",
        "How do limitation of liability clauses typically work in SaaS agreements?",
    ]
    for q in questions:
        print(f"\nQ: {q}")
        result = search_knowledge_base(q, db)
        answer = result["answer"]
        if isinstance(answer, dict):
            print(f"A: {answer.get('answer', 'N/A')[:300]}")
        print(f"Routed to: {result['routing'].get('query_type')}")
        print(f"Sources: {[s['id'] for s in result['sources'][:3]]}")

    # Demo 3: Breach Response (skipped if data not available)
    print("\n\n--- DEMO 3: Breach Notification Search ---")
    try:
        result = search_knowledge_base(
            "What is California's breach notification deadline?", db
        )
        answer = result["answer"]
        if isinstance(answer, dict):
            print(f"A: {answer.get('answer', 'N/A')[:300]}")
        print(f"Routed to: {result['routing'].get('query_type')}")
        print(f"Sources: {[s['id'] for s in result['sources'][:3]]}")
    except Exception as e:
        print(f"  (Skipped — breach data may not be ingested: {e})")

    # Demo 4: Playbook Search (skipped if data not available)
    print("\n\n--- DEMO 4: Playbook Search ---")
    try:
        result = search_knowledge_base(
            "What is the firm's position on indemnification caps?", db
        )
        answer = result["answer"]
        if isinstance(answer, dict):
            print(f"A: {answer.get('answer', 'N/A')[:300]}")
        print(f"Routed to: {result['routing'].get('query_type')}")
        print(f"Sources: {[s['id'] for s in result['sources'][:3]]}")
    except Exception as e:
        print(f"  (Skipped — playbook data may not be ingested: {e})")

    print("\n\n--- Demo complete ---")


if __name__ == "__main__":
    main()
