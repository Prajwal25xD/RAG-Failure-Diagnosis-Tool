
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found in .env")

engine = create_engine(DATABASE_URL)

MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
BATCH_SIZE = 16

print("Loading Hugging Face Nomic model...")

model = SentenceTransformer(
    MODEL_NAME,
    trust_remote_code=True
)

total_generated = 0

while True:
    # Read chunks that do not have Hugging Face embeddings yet
    with engine.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT id, chunk_text
                FROM public.tender_chunks
                WHERE embedding_hf IS NULL
                ORDER BY id
                LIMIT :batch_size
            """),
            {"batch_size": BATCH_SIZE}
        ).mappings().all()

    if not rows:
        break

    ids = [row["id"] for row in rows]
    texts = [
        "search_document: " + row["chunk_text"]
        for row in rows
    ]

    # Generate embeddings
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    # Save the embeddings in PostgreSQL
    with engine.begin() as conn:
        for chunk_id, embedding in zip(ids, embeddings):
            if len(embedding) != 768:
                raise ValueError(
                    f"Unexpected dimensions for chunk {chunk_id}"
                )

            embedding_string = "[" + ",".join(
                str(float(value)) for value in embedding
            ) + "]"

            conn.execute(
                text("""
                    UPDATE public.tender_chunks
                    SET embedding_hf = CAST(:embedding AS vector)
                    WHERE id = :chunk_id
                      AND embedding_hf IS NULL
                """),
                {
                    "embedding": embedding_string,
                    "chunk_id": chunk_id
                }
            )

    total_generated += len(rows)
    print(f"Processed {total_generated} chunks")

print("Finished generating Hugging Face embeddings!")

with engine.connect() as conn:
    counts = conn.execute(
        text("""
            SELECT
                COUNT(*) AS total_chunks,
                COUNT(embedding_hf) AS embedded_chunks,
                COUNT(*) FILTER (
                    WHERE embedding_hf IS NULL
                ) AS pending_chunks
            FROM public.tender_chunks
        """)
    ).mappings().one()

print(f"Total chunks: {counts['total_chunks']}")
print(f"Embedded chunks: {counts['embedded_chunks']}")
print(f"Pending chunks: {counts['pending_chunks']}")

engine.dispose()