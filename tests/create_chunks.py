
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load database connection from .env
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found in .env")

engine = create_engine(DATABASE_URL)

# Start with 2,000 characters per chunk and 200 characters overlap.
# These are characters, not tokens.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# Source text column, document type, source URL column, count column
DOCUMENTS = [
    (
        "bid_document_text",
        "bid_document",
        "bid_document",
        "bid_document_chunks",
    ),
    (
        "rfp_text",
        "rfp",
        "rfp",
        "rfp_chunks",
    ),
    (
        "pre_bid_and_corrigendum_text",
        "pre_bid_and_corrigendum",
        "pre_bid_and_corrigendum",
        "pre_bid_and_corrigendum_chunks",
    ),
]


def create_chunks():
    with engine.begin() as conn:

        # Read the existing extracted text from PostgreSQL.
        rows = conn.execute(text("""
            SELECT ten_id,
                   bid_document, rfp, pre_bid_and_corrigendum,
                   bid_document_text,
                   rfp_text,
                   pre_bid_and_corrigendum_text
            FROM public.gujrat_tender
            ORDER BY ten_id
        """)).mappings().all()

        print(f"Found {len(rows)} tenders")

        for row in rows:
            tender_id = row["ten_id"]
            total_chunks = 0
            document_counts = {}

            for text_column, doc_type, url_column, count_column in DOCUMENTS:
                document_text = row[text_column]
                document_url = row[url_column] or ""

                # Skip empty or missing extracted text.
                if not document_text or not document_text.strip():
                    document_counts[count_column] = 0
                    continue

                chunks = splitter.split_text(document_text)
                document_counts[count_column] = len(chunks)
                total_chunks += len(chunks)

                for index, chunk in enumerate(chunks):
                    conn.execute(text("""
                        INSERT INTO public.tender_chunks
                            (tender_id, document_type, document_url,
                             chunk_index, chunk_text)
                        VALUES
                            (:tender_id, :document_type, :document_url,
                             :chunk_index, :chunk_text)
                        ON CONFLICT
                            (tender_id, document_type,
                             document_url, chunk_index)
                        DO UPDATE SET
                            chunk_text = EXCLUDED.chunk_text
                    """), {
                        "tender_id": tender_id,
                        "document_type": doc_type,
                        "document_url": document_url,
                        "chunk_index": index,
                        "chunk_text": chunk,
                    })

            # Count successfully stored chunk embeddings.
            embedding_counts = conn.execute(text("""
                SELECT
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
                WHERE tender_id = :tender_id
            """), {"tender_id": tender_id}).mappings().one()

            conn.execute(text("""
                UPDATE public.gujrat_tender
                SET bid_document_chunks = :bid_chunks,
                    rfp_chunks = :rfp_chunks,
                    pre_bid_and_corrigendum_chunks = :pre_chunks,
                    total_chunks = :total_chunks,
                    bid_document_embedding_count = :bid_emb,
                    rfp_embedding_count = :rfp_emb,
                    pre_bid_and_corrigendum_embedding_count = :pre_emb,
                    total_embedding_count = :total_emb
                WHERE ten_id = :tender_id
            """), {
                "tender_id": tender_id,
                "bid_chunks": document_counts.get("bid_document_chunks", 0),
                "rfp_chunks": document_counts.get("rfp_chunks", 0),
                "pre_chunks": document_counts.get(
                    "pre_bid_and_corrigendum_chunks", 0
                ),
                "total_chunks": total_chunks,
                "bid_emb": embedding_counts["bid_count"],
                "rfp_emb": embedding_counts["rfp_count"],
                "pre_emb": embedding_counts["pre_bid_count"],
                "total_emb": (
                    embedding_counts["bid_count"]
                    + embedding_counts["rfp_count"]
                    + embedding_counts["pre_bid_count"]
                ),
            })

            print(
                f"Tender {tender_id}: "
                f"{total_chunks} chunks created/updated"
            )

    print("\nChunking completed successfully!")


if __name__ == "__main__":
    create_chunks()