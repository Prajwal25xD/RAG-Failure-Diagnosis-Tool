
import config

from retrieval.hybrid_search import hybrid_search
from diagnostics.retrieval_diagnosis import diagnose_retrieval


def main():
    print("\nRAG RETRIEVAL DIAGNOSIS")
    print("=" * 50)

    question = input("Enter your question: ").strip()

    if not question:
        print("Question cannot be empty.")
        return

    document_id = input(
        "Enter document ID (optional): "
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

    print("\nCHUNKS PASSED TO LLM")
    print("-" * 50)

    for chunk in chunks[:config.FINAL_TOP_K]:
        print(
            f"ID={chunk['id']} | "
            f"Document={chunk['document_id']} | "
            f"Vector Rank={chunk.get('vector_rank')} | "
            f"Keyword Rank={chunk.get('keyword_rank')} | "
            f"RRF={chunk['rrf_score']:.6f}"
        )


if __name__ == "__main__":
    main()
