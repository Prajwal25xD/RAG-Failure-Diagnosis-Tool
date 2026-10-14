
"""
Keyword Search Module
=====================

PostgreSQL full-text search with:

- Fast Mode: limited top-k keyword matches
- Exhaustive Mode: all keyword-matching chunks
- Dynamic schema and column names
- Optional document filtering
- Ranked results
- Batch fetching for exhaustive mode
"""

from sqlalchemy import text

import config
from retrieval.vector_search import (
    get_engine,
    selected_columns,
)


def keyword_search(
    question: str,
    document_id=None,
    top_k=None,
    exhaustive: bool = False,
) -> list[dict]:
    """
    Search document chunks using PostgreSQL full-text search.

    Fast Mode:
        Return top-k matching chunks.

    Exhaustive Mode:
        Return every matching chunk, ranked by keyword relevance.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not exhaustive:
        if top_k is None:
            top_k = config.KEYWORD_CANDIDATE_LIMIT

        top_k = int(top_k)

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

    table = config.qualified_table()

    chunk_text_column = config.quote_identifier(
        config.CHUNK_TEXT_COLUMN,
        "CHUNK_TEXT_COLUMN",
    )

    document_column = config.quote_identifier(
        config.DOCUMENT_ID_COLUMN,
        "DOCUMENT_ID_COLUMN",
    )

    chunk_id_column = config.quote_identifier(
        config.CHUNK_ID_COLUMN,
        "CHUNK_ID_COLUMN",
    )

    search_vector = (
        f"to_tsvector("
        f"'simple', "
        f"COALESCE(c.{chunk_text_column}, '')"
        f")"
    )

    search_query = (
        "plainto_tsquery('simple', :query)"
    )

    score_expression = (
        f"ts_rank_cd("
        f"{search_vector}, "
        f"{search_query}"
        f")"
    )

    limit_clause = (
        ""
        if exhaustive
        else "LIMIT :limit"
    )

    sql = text(
        f"""
        SELECT
            {selected_columns()},
            {score_expression} AS keyword_score

        FROM {table} AS c

        WHERE {search_vector} @@ {search_query}

          AND (
              CAST(:document_id AS text) IS NULL
              OR CAST(c.{document_column} AS text) = :document_id
          )

        ORDER BY
            keyword_score DESC,
            c.{chunk_id_column}

        {limit_clause}
        """
    )

    params = {
        "query": question.strip(),
        "document_id": (
            str(document_id)
            if document_id is not None
            else None
        ),
    }

    if not exhaustive:
        params["limit"] = top_k

    results = []

    with get_engine().connect() as connection:

        if exhaustive:
            batch_size = int(
                getattr(
                    config,
                    "EXHAUSTIVE_BATCH_SIZE",
                    500,
                )
            )

            if batch_size <= 0:
                raise ValueError(
                    "EXHAUSTIVE_BATCH_SIZE must be positive."
                )

            execution = connection.execution_options(
                stream_results=True,
                yield_per=batch_size,
            )

            cursor = execution.execute(
                sql,
                params,
            ).mappings()

            try:
                rank = 0

                while True:
                    rows = cursor.fetchmany(
                        batch_size
                    )

                    if not rows:
                        break

                    for row in rows:
                        rank += 1

                        chunk = dict(row)
                        chunk["keyword_rank"] = rank

                        results.append(chunk)

            finally:
                cursor.close()

        else:
            rows = connection.execute(
                sql,
                params,
            ).mappings().all()

            results = [
                dict(row)
                for row in rows
            ]

            for rank, chunk in enumerate(
                results,
                start=1,
            ):
                chunk["keyword_rank"] = rank

    return results


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

    chunks = keyword_search(
        question=question,
        document_id=document_id,
        exhaustive=(mode == "exhaustive"),
    )

    print("\nMode:", mode)
    print("Keyword matches:", len(chunks))

    for chunk in chunks[:10]:
        print(
            f"ID={chunk['id']} | "
            f"Document={chunk['document_id']} | "
            f"Keyword Rank={chunk['keyword_rank']} | "
            f"Score={chunk['keyword_score']}"
        )
