
"""
End-to-end reasoning diagnosis test.

Uses:
- Real PostgreSQL chunks
- Hybrid retrieval
- LM Studio answer generation
- Evidence-grounded reasoning diagnosis
"""

import config

from retrieval.hybrid_search import hybrid_search

from diagnostics.reasoning_diagnosis import (
    generate_rag_answer,
    diagnose_reasoning,
)


def main():
    print("\nRAG REASONING DIAGNOSIS")
    print("=" * 55)

    question = input(
        "Enter your question: "
    ).strip()

    if not question:
        print("Question cannot be empty.")
        return

    document_id = input(
        "Document ID (optional): "
    ).strip() or None

    print("\nRetrieving evidence from PostgreSQL...")

    chunks = hybrid_search(
        question=question,
        document_id=document_id,
        top_k=config.FINAL_TOP_K,
    )

    print(f"Retrieved chunks: {len(chunks)}")

    if not chunks:
        print(
            "No evidence retrieved. "
            "Investigate retrieval before reasoning."
        )
        return

    print("\nGENERATING ANSWER")
    print("-" * 55)

    answer = generate_rag_answer(
        question=question,
        retrieved_chunks=chunks,
    )

    print("\nANSWER:")
    print(answer)

    print("\nRUNNING REASONING DIAGNOSIS")
    print("-" * 55)

    report = diagnose_reasoning(
        question=question,
        generated_answer=answer,
        retrieved_chunks=chunks,
    )

    print("\nSTATUS:", report["status"])
    print(
        "Chunks analyzed:",
        report.get("chunks_analyzed", 0),
    )
    print("Summary:", report.get("summary", ""))
    print(
        "Recommended fix:",
        report.get("recommended_fix", ""),
    )

    findings = report.get("findings", [])

    print("\nFindings:", len(findings))

    for index, finding in enumerate(
        findings, start=1
    ):
        print(f"\nFinding {index}")
        print("Type:", finding["type"])
        print("Claim:", finding["claim"])
        print("Chunk IDs:", finding["chunk_ids"])
        print("Explanation:", finding["explanation"])
        print(
            "Recommended fix:",
            finding["recommended_fix"],
        )

    print("\nSOURCE CHUNKS")
    print("-" * 55)

    for chunk in chunks:
        print(
            f"ID={chunk['id']} | "
            f"Document={chunk.get('document_id')} | "
            f"Vector Rank={chunk.get('vector_rank')} | "
            f"Keyword Rank={chunk.get('keyword_rank')}"
        )


if __name__ == "__main__":
    main()


# python -m tests.test_reasoning_diagnosis