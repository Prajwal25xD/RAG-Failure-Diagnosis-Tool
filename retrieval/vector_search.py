
"""
Vector Retrieval Module
=======================

Supports:
- Fast Mode: top-k vector retrieval
- Exhaustive Mode: exact ranking of all eligible embeddings
- Hugging Face sentence-transformers
- PostgreSQL pgvector
- Dynamic database schema and column configuration
- Optional document ID filtering
- Batch-based result fetching

No hardcoded questions, tender IDs, or chunk IDs.
"""

from functools import lru_cache

from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text

import config


# ---------------------------------------------------------
# DATABASE CONNECTION
# ---------------------------------------------------------

@lru_cache(maxsize=1)
def get_engine():
    if not config.DATABASE_URL:
        raise ValueError("DATABASE_URL is missing from .env")

    return create_engine(
        config.DATABASE_URL,
        pool_pre_ping=True,
    )


# ---------------------------------------------------------
# HUGGING FACE EMBEDDING MODEL
# ---------------------------------------------------------

@lru_cache(maxsize=1)
def get_embedder():
    model = SentenceTransformer(
        config.EMBEDDING_MODEL_NAME,
        trust_remote_code=True,
    )

    actual_dim = model.get_sentence_embedding_dimension()

    if actual_dim != config.EMBEDDING_DIMENSIONS:
        raise ValueError(
            "Embedding dimension mismatch: "
            f"expected {config.EMBEDDING_DIMENSIONS}, "
            f"got {actual_dim}"
        )

    return model


def embed_query(question: str) -> list[float]:
    """Generate a normalized embedding for the user question."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    embedding = get_embedder().encode(
        config.EMBEDDING_QUERY_PREFIX + question.strip(),
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    return [
        float(value)
        for value in embedding.tolist()
    ]


# ---------------------------------------------------------
# DYNAMIC COLUMN SELECTION
# ---------------------------------------------------------

def selected_columns(alias="c"):
    """Select chunk fields using configured database columns."""

    prefix = f"{alias}."

    fields = [
        (
            f'{prefix}'
            f'{config.quote_identifier(config.CHUNK_ID_COLUMN, "CHUNK_ID_COLUMN")}'
            " AS id"
        ),
        (
            f'{prefix}'
            f'{config.quote_identifier(config.DOCUMENT_ID_COLUMN, "DOCUMENT_ID_COLUMN")}'
            " AS document_id"
        ),
        (
            f'{prefix}'
            f'{config.quote_identifier(config.CHUNK_TEXT_COLUMN, "CHUNK_TEXT_COLUMN")}'
            " AS chunk_text"
        ),
    ]

    optional_fields = [
        (
            config.DOCUMENT_TYPE_COLUMN,
            "document_type",
            "DOCUMENT_TYPE_COLUMN",
        ),
        (
            config.DOCUMENT_URL_COLUMN,
            "document_url",
            "DOCUMENT_URL_COLUMN",
        ),
        (
            config.CHUNK_INDEX_COLUMN,
            "chunk_index",
            "CHUNK_INDEX_COLUMN",
        ),
    ]

    for column, output_name, setting in optional_fields:

        quoted = config.quote_identifier(
            column,
            setting,
            allow_empty=True,
        )

        if quoted:
            fields.append(
                f"{prefix}{quoted} AS {output_name}"
            )
        else:
            fields.append(
                f"NULL AS {output_name}"
            )

    return ", ".join(fields)


# ---------------------------------------------------------
# VECTOR SEARCH
# ---------------------------------------------------------

def vector_search(
    question: str,
    document_id=None,
    top_k=None,
    exhaustive: bool = False,
) -> list[dict]:
    """
    Retrieve semantically similar chunks.

    Fast Mode:
        Returns the top-k vector matches.

    Exhaustive Mode:
        Computes exact distances for every eligible row,
        ranks all matches, and returns the full ranked list.

    Warning:
        Returning all rows can consume substantial memory.
        For very large databases, use a streaming consumer.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not exhaustive:
        if top_k is None:
            top_k = config.VECTOR_CANDIDATE_LIMIT

        top_k = int(top_k)

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

    embedding = embed_query(question)

    embedding_string = (
        "[" + ",".join(str(value) for value in embedding) + "]"
    )

    table = config.qualified_table()

    embedding_column = config.quote_identifier(
        config.EMBEDDING_COLUMN,
        "EMBEDDING_COLUMN",
    )

    document_column = config.quote_identifier(
        config.DOCUMENT_ID_COLUMN,
        "DOCUMENT_ID_COLUMN",
    )

    chunk_text_column = config.quote_identifier(
        config.CHUNK_TEXT_COLUMN,
        "CHUNK_TEXT_COLUMN",
    )

    # Fast Mode uses LIMIT; Exhaustive Mode does not.
    limit_clause = "" if exhaustive else "LIMIT :limit"

    sql = text(
        f"""
        SELECT
            {selected_columns()},
            c.{embedding_column}
                <=> CAST(:embedding AS vector) AS distance

        FROM {table} AS c

        WHERE c.{embedding_column} IS NOT NULL
          AND c.{chunk_text_column} IS NOT NULL
          AND (
              CAST(:document_id AS text) IS NULL
              OR CAST(c.{document_column} AS text) = :document_id
          )

        ORDER BY
            c.{embedding_column}
                <=> CAST(:embedding AS vector),
            c.{config.quote_identifier(config.CHUNK_ID_COLUMN, "CHUNK_ID_COLUMN")}

        {limit_clause}
        """
    )

    params = {
        "embedding": embedding_string,
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
            # Prevent pgvector approximate index scans for
            # this exact, full-database ranking operation.
            #
            # PostgreSQL can still choose other access paths.
            # A full ordered result may require substantial
            # database-side sorting resources.

            connection.execute(
                text("SET LOCAL enable_indexscan = off")
            )

            connection.execute(
                text("SET LOCAL enable_bitmapscan = off")
            )

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

            rank = 0

            while True:
                rows = cursor.fetchmany(batch_size)

                if not rows:
                    break

                for row in rows:
                    rank += 1

                    chunk = dict(row)
                    chunk["vector_rank"] = rank

                    results.append(chunk)

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
                chunk["vector_rank"] = rank

    return results


# ---------------------------------------------------------
# DIRECT MODULE TEST
# ---------------------------------------------------------

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

    print("\nRunning vector retrieval...")

    chunks = vector_search(
        question=question,
        document_id=document_id,
        exhaustive=(mode == "exhaustive"),
    )

    print("\nMode:", mode)
    print("Retrieved chunks:", len(chunks))

    print("\nTop results:")

    for chunk in chunks[:10]:
        print(
            f"ID={chunk['id']} | "
            f"Document={chunk['document_id']} | "
            f"Rank={chunk['vector_rank']} | "
            f"Distance={chunk['distance']}"
        )
