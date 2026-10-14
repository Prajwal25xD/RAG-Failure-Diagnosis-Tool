
"""
Hybrid Search Module
====================

Combines:
- Hugging Face vector similarity
- PostgreSQL full-text search
- Reciprocal Rank Fusion (RRF)

Modes:
- Fast: limited candidate retrieval
- Exhaustive: full eligible vector ranking and
              all keyword matches

No hardcoded questions, document IDs, or chunk IDs.
"""

import config

from retrieval.vector_search import vector_search
from retrieval.keyword_search import keyword_search


def hybrid_search(
    question: str,
    document_id=None,
    top_k=None,
    exhaustive: bool = False,
) -> list[dict]:
    """
    Retrieve and rank chunks using vector + keyword search.

    Fast Mode:
        Uses configured candidate limits.

    Exhaustive Mode:
        Searches all eligible embedded chunks and all
        keyword-matching chunks.

    top_k:
        Maximum number of final ranked chunks returned.
        None means return all fused candidates in
        Exhaustive Mode.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    if top_k is not None:
        top_k = int(top_k)

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

    if exhaustive:
        vector_results = vector_search(
            question=question,
            document_id=document_id,
            top_k=None,
            exhaustive=True,
        )

        keyword_results = keyword_search(
            question=question,
            document_id=document_id,
            top_k=None,
            exhaustive=True,
        )

    else:
        vector_results = vector_search(
            question=question,
            document_id=document_id,
            top_k=config.VECTOR_CANDIDATE_LIMIT,
            exhaustive=False,
        )

        keyword_results = keyword_search(
            question=question,
            document_id=document_id,
            top_k=config.KEYWORD_CANDIDATE_LIMIT,
            exhaustive=False,
        )

    # -----------------------------------------------------
    # RECIPROCAL RANK FUSION
    # -----------------------------------------------------

    combined = {}

    for chunk in vector_results:

        chunk_id = str(chunk["id"])

        rank = chunk["vector_rank"]

        rrf_score = 1.0 / (
            config.RRF_K + rank
        )

        combined[chunk_id] = {
            **chunk,
            "rrf_score": rrf_score,
            "keyword_rank": None,
            "keyword_score": None,
        }

    for chunk in keyword_results:

        chunk_id = str(chunk["id"])

        rank = chunk["keyword_rank"]

        rrf_score = 1.0 / (
            config.RRF_K + rank
        )

        if chunk_id in combined:

            combined[chunk_id]["rrf_score"] += (
                rrf_score
            )

            combined[chunk_id]["keyword_rank"] = rank

            combined[chunk_id]["keyword_score"] = (
                chunk.get("keyword_score")
            )

        else:
            combined[chunk_id] = {
                **chunk,
                "rrf_score": rrf_score,
                "vector_rank": None,
                "distance": None,
            }

    ranked_results = sorted(
        combined.values(),
        key=lambda chunk: (
            -chunk["rrf_score"],
            str(chunk["id"]),
        ),
    )

    # In Fast Mode, preserve the configured default.
    if not exhaustive and top_k is None:
        top_k = config.FINAL_TOP_K

    if top_k is not None:
        ranked_results = ranked_results[:top_k]

    return ranked_results


if __name__ == "__main__":

    question = input(
        "Enter your question: "
    ).strip()

    mode = input(
        "Mode [fast/exhaustive] (default fast): "
    ).strip().lower() or "fast"

    if mode not in {"fast", "exhaustive"}:
        raise ValueError(
            "Mode must be fast or exhaustive."
        )

    document_id = input(
        "Document ID (optional): "
    ).strip() or None

    chunks = hybrid_search(
        question=question,
        document_id=document_id,
        top_k=10,
        exhaustive=(mode == "exhaustive"),
    )

    print("\nMode:", mode)
    print("Returned chunks:", len(chunks))

    for chunk in chunks:
        print(
            f"ID={chunk['id']} | "
            f"Document={chunk['document_id']} | "
            f"RRF={chunk['rrf_score']:.6f} | "
            f"Vector Rank={chunk.get('vector_rank')} | "
            f"Keyword Rank={chunk.get('keyword_rank')}"
        )

# To Run: 
# python -m retrieval.hybrid_search