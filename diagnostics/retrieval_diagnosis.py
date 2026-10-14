
"""Dynamic retrieval diagnosis for RAG systems."""

from typing import Any


def diagnose_retrieval(
    retrieved_chunks: list[dict[str, Any]],
    supporting_chunk_ids: list | None = None,
    top_k: int | None = None,
) -> dict:
    """
    Diagnose retrieval using real ranked candidates.

    If verified supporting IDs are unavailable, report retrieval
    statistics without claiming a confirmed retrieval failure.
    """

    if top_k is None:
        import config
        top_k = config.FINAL_TOP_K

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    ranked_ids = []

    for chunk in retrieved_chunks:
        chunk_id = chunk.get("id", chunk.get("chunk_id"))

        if chunk_id is None:
            raise ValueError("Retrieved chunk has no ID.")

        ranked_ids.append(str(chunk_id))

    if len(ranked_ids) != len(set(ranked_ids)):
        raise ValueError("Duplicate chunk IDs in ranked results.")

    context_ids = set(ranked_ids[:top_k])
    candidate_ids = set(ranked_ids)

    result = {
        "total_candidates": len(retrieved_chunks),
        "context_chunk_count": min(top_k, len(retrieved_chunks)),
        "top_k": top_k,
        "vector_matches": sum(
            chunk.get("vector_rank") is not None
            for chunk in retrieved_chunks
        ),
        "keyword_matches": sum(
            chunk.get("keyword_rank") is not None
            for chunk in retrieved_chunks
        ),
        "failure_type": "unknown",
        "reason": "No verified supporting evidence was provided.",
        "recall_at_k": None,
        "recommended_fix": (
            "Assess evidence relevance or supply verified "
            "supporting chunk IDs."
        ),
    }

    if not supporting_chunk_ids:
        return result

    expected_ids = {str(cid) for cid in supporting_chunk_ids}

    found_in_context = expected_ids & context_ids
    found_in_candidates = expected_ids & candidate_ids
    missing_ids = expected_ids - candidate_ids
    ranked_out_ids = found_in_candidates - context_ids

    result["expected_chunk_ids"] = sorted(expected_ids)
    result["found_in_context"] = sorted(found_in_context)
    result["missing_chunk_ids"] = sorted(missing_ids)
    result["ranked_out_chunk_ids"] = sorted(ranked_out_ids)
    result["supporting_chunk_ranks"] = {
        cid: ranked_ids.index(cid) + 1
        for cid in expected_ids
        if cid in candidate_ids
    }
    result["recall_at_k"] = (
        len(found_in_context) / len(expected_ids)
    )

    if expected_ids <= context_ids:
        result["failure_type"] = "retrieval_success"
        result["reason"] = "All verified evidence is in the LLM context."
        result["recommended_fix"] = "No retrieval fix required."

    elif missing_ids and ranked_out_ids:
        result["failure_type"] = "mixed_retrieval_failure"
        result["reason"] = (
            "Some expected evidence is missing from the candidates; "
            "other evidence was ranked outside the LLM context."
        )

    elif missing_ids:
        result["failure_type"] = "retrieval_miss"
        result["reason"] = (
            "Some verified evidence was not found among the candidates."
        )

    else:
        result["failure_type"] = "ranking_failure"
        result["reason"] = (
            "Verified evidence was found but ranked outside top-k."
        )

    if result["failure_type"] != "retrieval_success":
        result["recommended_fix"] = (
            "Inspect indexing coverage, candidate limits, "
            "hybrid ranking and reranking."
        )

    return result

# To Run:
# python -m tests.test_retrieval_diagnosis