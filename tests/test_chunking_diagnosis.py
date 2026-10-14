
"""Test chunking diagnosis using actual PostgreSQL chunks."""

import config

from retrieval.hybrid_search import hybrid_search
from diagnostics.chunking_diagnosis import diagnose_chunking


def main():
    print("\nRAG CHUNKING DIAGNOSIS")
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

    report = diagnose_chunking(
        retrieved_chunks=chunks,
        inspect_neighbors=True,
    )

    print("\nCHUNKING DIAGNOSIS REPORT")
    print("-" * 50)

    print("Status:", report["status"])
    print("Analyzed chunks:", report["analyzed_chunks"])
    print("Issue counts:", report["issue_counts"])

    for chunk_report in report["chunk_reports"]:
        print("\nChunk ID:", chunk_report["chunk_id"])
        print("Document ID:", chunk_report["document_id"])
        print("Chunk index:", chunk_report["chunk_index"])
        print("Characters:", chunk_report["character_count"])
        print("Whitespace ratio:", chunk_report["whitespace_ratio"])
        print("Issues:", chunk_report["issues"])
        print(
            "Neighbor IDs:",
            chunk_report.get("neighbor_chunk_ids", []),
        )


if __name__ == "__main__":
    main()


# python -m tests.test_chunking_diagnosis