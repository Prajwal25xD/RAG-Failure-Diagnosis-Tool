def diagnose_retrieval(
    retrieved_chunks,
    supporting_chunk_id,
    top_k
):
    """
    retrieved_chunks: list of chunks returned to the LLM
    supporting_chunk_id: ID of the known correct passage
    top_k: number of chunks passed to the LLM

    """

    retrieved_ids = [
        chunk["chunk_id"]
        for chunk in retrieved_chunks
    ]

    if supporting_chunk_id in retrieved_ids:
        rank = retrieved_ids.index(supporting_chunk_id) + 1

        return {
            "failure_type": "not_a_retrieval_failure",
            "supporting_chunk_id": rank,
            "recommended_fix":( "Inspect answer generation and evidence usage."
            )
        }

    return {
        "failure_type": "retrieval_miss",
        "supporting_chunk_rank": None,
        "recommended_fix": (
            "Inspect the full-corpus ranking. "
            "Test hybrid search, reranking, "
            "or a larger top-k."
        )
    }

# EXAMPLE TEST

chunks = [
    {"chunk_id": "doc_12"},
    {"chunk_id": "doc_08"},
    {"chunk_id": "doc_31"}
]

result = diagnose_retrieval(
    retrieved_chunks=chunks,
    supporting_chunk_id="doc_14",
    top_k=3
)

print(result)
    