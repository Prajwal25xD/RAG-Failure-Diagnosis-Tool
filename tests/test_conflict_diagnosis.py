
"""Test conflict diagnosis using actual PostgreSQL evidence."""

import config

from retrieval.hybrid_search import hybrid_search
from diagnostics.conflict_diagnosis import diagnose_conflicts


def main():
    print("\nRAG CONFLICT DIAGNOSIS")
    print("=" * 50)

    question = input("Enter your question: ").strip()

    if not question:
        print("Question cannot be empty.")
        return

    document_id = input(
        "Document ID (optional): "
    ).strip() or None

    chunks = hybrid_search(
        question=question,
        document_id=document_id,
        top_k=config.FINAL_TOP_K,
    )

    print(f"\nRetrieved {len(chunks)} real chunks.")

    if not chunks:
        print("No matching chunks found.")
        return

    print("\nAnalyzing conflicts using LM Studio...\n")

    report = diagnose_conflicts(
        question=question,
        retrieved_chunks=chunks,
    )

    print("OVERALL STATUS:", report["status"])
    print("Documents analyzed:", report["documents_analyzed"])
    print("Potential conflicts:", report["total_conflicts"])

    for document_report in report["document_reports"]:
        print("\n" + "-" * 50)
        print("Document:", document_report["document_id"])
        print("Status:", document_report["status"])
        print("Summary:", document_report.get("summary", ""))

        for conflict in document_report.get("conflicts", []):
            print("\nConflicting chunk IDs:", conflict["chunk_ids"])
            print("Statement A:", conflict.get("statement_a", ""))
            print("Statement B:", conflict.get("statement_b", ""))
            print("Explanation:", conflict.get("explanation", ""))
            print("Suggested fix:", conflict.get("recommended_fix", ""))


if __name__ == "__main__":
    main()


# Before running
# Make sure your LM Studio local server is running and a model is loaded.
# python -m tests.test_conflict_diagnosis