
"""Test retrieval diagnosis using real PostgreSQL chunks."""

import config

from retrieval.hybrid_search import hybrid_search
from diagnostics.retrieval_diagnosis import diagnose_retrieval


def main():
    print("\nRAG RETRIEVAL DIAGNOSIS TEST")
    print("=" * 50)

    question = input("Enter your question: ").strip()

    if not question:
        print("Question cannot be empty.")
        return

    document_id = input(
        "Document ID (optional, press Enter to search all): "
    ).strip() or None

    candidate_limit = (
        config.VECTOR_CANDIDATE_LIMIT
        + config.KEYWORD_CANDIDATE_LIMIT
    )

    chunks = hybrid_search(
        question=question,
        document_id=document_id,
        top_k=candidate_limit,
    )

    report = diagnose_retrieval(
        retrieved_chunks=chunks,
        top_k=config.FINAL_TOP_K,
    )

    print("\nRETRIEVAL DIAGNOSIS")
    print("-" * 50)

    for key, value in report.items():
        print(f"{key}: {value}")

    print("\nTOP CHUNKS PASSED TO LLM")
    print("-" * 50)

    for chunk in chunks[:config.FINAL_TOP_K]:
        print(f"Chunk ID: {chunk['id']}")
        print(f"Document ID: {chunk['document_id']}")
        print(f"Vector Rank: {chunk.get('vector_rank')}")
        print(f"Keyword Rank: {chunk.get('keyword_rank')}")
        print(f"RRF Score: {chunk['rrf_score']:.6f}")
        print("-" * 50)


if __name__ == "__main__":
    main()


# TO RUN: python -m tests.test_retrieval_diagnosis