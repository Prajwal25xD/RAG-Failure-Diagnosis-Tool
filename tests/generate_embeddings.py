
import os
import ollama

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from pgvector.sqlalchemy import Vector

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found in .env")

engine = create_engine(DATABASE_URL)

MODEL = "nomic-embed-text"
BATCH_SIZE = 20


def generate_embedding(chunk_text):
    response = ollama.embed(
        model=MODEL,
        input="search_document: " + chunk_text
    )
    embedding = response["embeddings"][0]

    if len(embedding) != 768:
        raise ValueError(
            f"Expected 768 dimensions, got {len(embedding)}"
        )

    return embedding


def main():
    processed = 0

    while True:
        # Read a small batch of chunks that need embeddings.
        with engine.connect() as conn:
            chunks = conn.execute(text("""
                SELECT id, chunk_text
                FROM public.tender_chunks
                WHERE embedding IS NULL
                ORDER BY id
                LIMIT :batch_size
            """), {"batch_size": BATCH_SIZE}).mappings().all()

        if not chunks:
            break

        for chunk in chunks:
            vector = generate_embedding(chunk["chunk_text"])

            # Save each vector. Commit each successful update.
            with engine.begin() as conn:
                conn.execute(text("""
                    UPDATE public.tender_chunks
                    SET embedding = CAST(:embedding AS vector)
                    WHERE id = :chunk_id
                      AND embedding IS NULL
                """), {
                    "chunk_id": chunk["id"],
                    "embedding": str(vector),
                })

            processed += 1
            print(f"Embedded chunk {chunk['id']} | Total this run: {processed}")

    # Refresh per-document embedding counts in the main table.
    with engine.begin() as conn:
        conn.execute(text("""
            UPDATE public.gujrat_tender AS t
            SET bid_document_embedding_count = counts.bid_count,
                rfp_embedding_count = counts.rfp_count,
                pre_bid_and_corrigendum_embedding_count = counts.pre_bid_count,
                total_embedding_count =
                    counts.bid_count + counts.rfp_count + counts.pre_bid_count
            FROM (
                SELECT
                    tender_id,
                    COUNT(*) FILTER (
                        WHERE document_type = 'bid_document'
                          AND embedding IS NOT NULL
                    ) AS bid_count,
                    COUNT(*) FILTER (
                        WHERE document_type = 'rfp'
                          AND embedding IS NOT NULL
                    ) AS rfp_count,
                    COUNT(*) FILTER (
                        WHERE document_type = 'pre_bid_and_corrigendum'
                          AND embedding IS NOT NULL
                    ) AS pre_bid_count
                FROM public.tender_chunks
                GROUP BY tender_id
            ) AS counts
            WHERE t.ten_id = counts.tender_id
        """))

    print(f"\nCompleted! Generated {processed} embeddings.")
    print("Embedding counts in gujrat_tender have been refreshed.")


if __name__ == "__main__":
    main()