
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer

# 1. Load database configuration
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL is missing from .env")

engine = create_engine(DATABASE_URL)

# 2. Load the same Hugging Face model used for embeddings
print("Loading Hugging Face model...")

model = SentenceTransformer(
    "nomic-ai/nomic-embed-text-v1.5",
    trust_remote_code=True
)

print("Model loaded successfully!")


def semantic_search(question, top_k=5):
    # 3. Convert the question into an embedding
    query_embedding = model.encode(
        ["search_query: " + question],
        normalize_embeddings=True
    )[0]

    # Convert the embedding to PostgreSQL vector format
    embedding_string = (
        "[" + ",".join(str(float(x)) for x in query_embedding) + "]"
    )

    # 4. Find the most similar tender chunks
    sql = text("""
        SELECT
            c.tender_id,
            c.document_type,
            c.chunk_index,
            c.chunk_text,
            c.embedding_hf <=> CAST(:embedding AS vector)
                AS distance,
            t.title AS tender_title
        FROM public.tender_chunks AS c
        LEFT JOIN public.gujrat_tender AS t
            ON t.ten_id = c.tender_id
        WHERE c.embedding_hf IS NOT NULL
        ORDER BY c.embedding_hf <=> CAST(:embedding AS vector)
        LIMIT :top_k
    """)

    with engine.connect() as connection:
        results = connection.execute(
            sql,
            {
                "embedding": embedding_string,
                "top_k": top_k
            }
        ).fetchall()

    # 5. Display results
    print(f"\nQuestion: {question}")
    print(f"Retrieved {len(results)} chunks\n")

    for rank, row in enumerate(results, start=1):
        print("=" * 70)
        print(f"Result: {rank}")
        print(f"Tender ID: {row.tender_id}")
        print(f"Tender title: {row.tender_title}")
        print(f"Document type: {row.document_type}")
        print(f"Chunk index: {row.chunk_index}")
        print(f"Cosine distance: {row.distance:.4f}")
        print("\nChunk text:")
        print(row.chunk_text[:1200])
        print()

    if not results:
        print("No embedded chunks found.")

    return results


if __name__ == "__main__":
    question = input("Ask a question about the tenders: ").strip()

    if question:
        semantic_search(question, top_k=5)

    engine.dispose()