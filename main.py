"""
Legal Contract Clause Analyzer — Interactive CLI

A RAG-powered tool that helps lawyers review contract clauses by:
1. Finding similar clauses from the firm's knowledge base (FAISS vector search)
2. Generating risk analysis with actionable suggestions (LLM + prompt engineering)

Usage:
    python main.py              # Interactive clause analysis
    python main.py --evaluate   # Run evaluation suite
    python main.py --compare    # Compare prompt strategies
"""

import argparse
import os
import sys

from src.logging_config import setup_logging
from src.embeddings import load_clause_database
from src.rag_pipeline import analyze_clause, STRATEGIES
from src.evaluation import (
    evaluate_retrieval,
    compare_strategies,
    print_retrieval_results,
    print_comparison_results,
)

# Built-in sample clauses for quick testing
SAMPLE_CLAUSES = {
    "1": {
        "name": "Aggressive Non-Compete",
        "text": """The Employee agrees that for a period of two years following
        termination, Employee shall not directly or indirectly work for, consult
        with, or provide services to any business that competes with the Company
        in any market where the Company operates or plans to operate.""",
    },
    "2": {
        "name": "Unlimited Liability Waiver",
        "text": """The Service Provider shall bear no responsibility for any loss,
        damage, or injury arising from the services provided, including but not
        limited to loss of data, business interruption, or financial losses.
        The Client expressly waives all claims against Provider.""",
    },
    "3": {
        "name": "Broad Confidentiality with No Exceptions",
        "text": """The Receiving Party agrees that all information provided by the
        Disclosing Party, in any form, is strictly confidential and shall not be
        disclosed, copied, or used for any purpose other than evaluating the
        proposed transaction. This obligation has no expiration date.""",
    },
}


def interactive_mode(db: dict):
    """Run the interactive clause analysis loop."""
    provider_name = db["provider"].provider_name
    store_name = os.environ.get("VECTOR_STORE_PROVIDER", "faiss").upper()
    print("\n" + "=" * 60)
    print("  LEGAL CONTRACT CLAUSE ANALYZER")
    print(f"  Powered by RAG ({store_name} + {provider_name})")
    print("=" * 60)

    while True:
        print("\nOptions:")
        print("  [1-3]  Analyze a sample clause")
        print("  [p]    Paste your own clause")
        print("  [s]    Switch prompt strategy")
        print("  [q]    Quit")

        for num, sample in SAMPLE_CLAUSES.items():
            print(f"    {num}: {sample['name']}")

        choice = input("\nChoice: ").strip().lower()

        if choice == "q":
            print("Goodbye!")
            break

        # Select prompt strategy
        strategy = "few_shot"

        if choice == "s":
            print("\nAvailable strategies:")
            for name in STRATEGIES:
                marker = " (default)" if name == "few_shot" else ""
                print(f"  - {name}{marker}")
            strategy = input("Strategy: ").strip()
            if strategy not in STRATEGIES:
                print(f"Unknown strategy. Using few_shot.")
                strategy = "few_shot"
            continue

        # Get the clause text
        if choice in SAMPLE_CLAUSES:
            clause_text = SAMPLE_CLAUSES[choice]["text"]
            print(f"\nAnalyzing: {SAMPLE_CLAUSES[choice]['name']}")
        elif choice == "p":
            print("Paste your clause (enter a blank line when done):")
            lines = []
            while True:
                line = input()
                if line == "":
                    break
                lines.append(line)
            clause_text = "\n".join(lines)
            if not clause_text.strip():
                print("No clause entered.")
                continue
        else:
            print("Invalid choice.")
            continue

        # Run the RAG pipeline
        print("\nSearching knowledge base and generating analysis...")
        result = analyze_clause(clause_text, db, strategy=strategy)

        # Display draft header
        print(f"\n{'=' * 60}")
        print(f"  DRAFT ANALYSIS — Requires Attorney Review")
        print(f"  Strategy: {result['strategy']} | Model: {result['model']}")
        print(f"{'=' * 60}")

        # Display source trail
        print(f"\n--- Sources Retrieved ({len(result['sources'])}) ---")
        for src in result["sources"]:
            print(f"  [{src['id']}] {src['title']} "
                  f"(relevance: {src['score']:.3f}, risk: {src['risk_level']})")

        # Display analysis
        print(f"\n--- Analysis ---")
        analysis = result["analysis"]
        if isinstance(analysis, dict):
            import json
            print(json.dumps(analysis, indent=2))
        else:
            print(analysis)

        # Display footer
        print(f"\n{'=' * 60}")
        print(f"  Review Status: {result['review_status'].upper()}")
        print(f"  {result['disclaimer']}")
        print(f"{'=' * 60}")


def main():
    setup_logging()
    parser = argparse.ArgumentParser(description="Legal Contract Clause Analyzer")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run the evaluation suite")
    parser.add_argument("--compare", action="store_true",
                        help="Compare all prompt strategies")
    args = parser.parse_args()

    print("Initializing knowledge base...")
    db = load_clause_database()

    if args.evaluate:
        print("\nRunning retrieval evaluation...")
        retrieval_results = evaluate_retrieval(db)
        print_retrieval_results(retrieval_results)

    elif args.compare:
        print("\nComparing prompt strategies (this will take a minute)...")
        comparison = compare_strategies(db)
        print_comparison_results(comparison)

    else:
        interactive_mode(db)


if __name__ == "__main__":
    main()
