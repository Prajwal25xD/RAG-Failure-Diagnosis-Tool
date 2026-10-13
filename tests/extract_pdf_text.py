import os
import json
import hashlib
import requests
import pymupdf

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


# ==================================================
# 1. DATABASE CONNECTION
# ==================================================

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL is missing from your .env file")

engine = create_engine(DATABASE_URL)

PDF_DIR = "data/raw_pdfs"
TEXT_DIR = "data/extracted"

os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)


# ==================================================
# 2. PREPARE DATABASE TABLES
# ==================================================

with engine.begin() as conn:

    # Matching text columns in the original tender table
    conn.execute(text("""
        ALTER TABLE public.gujrat_tender
        ADD COLUMN IF NOT EXISTS bid_document_text TEXT,
        ADD COLUMN IF NOT EXISTS rfp_text TEXT,
        ADD COLUMN IF NOT EXISTS pre_bid_and_corrigendum_text TEXT,
        ADD COLUMN IF NOT EXISTS documents_text JSONB
    """))

    # Document-level table for RAG processing
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS public.tender_documents (
            id BIGSERIAL PRIMARY KEY,
            tender_id INTEGER NOT NULL,
            tender_title TEXT,
            document_url TEXT NOT NULL,
            document_type TEXT,
            extracted_text TEXT,
            page_count INTEGER,
            extracted_characters INTEGER,
            extraction_status TEXT NOT NULL DEFAULT 'pending',
            error_message TEXT,
            extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (tender_id, document_url)
        )
    """))

    # Support databases created using the earlier script
    conn.execute(text("""
        ALTER TABLE public.tender_documents
        ADD COLUMN IF NOT EXISTS document_type TEXT,
        ADD COLUMN IF NOT EXISTS error_message TEXT
    """))

print("Database tables are ready.")


# ==================================================
# 3. FETCH AND EXTRACT A PDF
# ==================================================

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0"})


def extract_pdf(url, tender_id):
    """Download a PDF, extract text and save local copies."""

    response = session.get(url, timeout=90)
    response.raise_for_status()

    if not response.content.startswith(b"%PDF-"):
        raise ValueError("URL did not return a valid PDF")

    # Stable filename based on the URL
    url_hash = hashlib.sha256(
        url.encode("utf-8")
    ).hexdigest()[:16]

    filename = f"tender_{tender_id}_{url_hash}"

    pdf_path = os.path.join(PDF_DIR, filename + ".pdf")
    txt_path = os.path.join(TEXT_DIR, filename + ".txt")

    with open(pdf_path, "wb") as file:
        file.write(response.content)

    with pymupdf.open(pdf_path) as pdf:
        page_count = len(pdf)
        page_texts = [
            page.get_text("text") for page in pdf
        ]

    extracted_text = "\n\n".join(page_texts)

    with open(txt_path, "w", encoding="utf-8") as file:
        file.write(extracted_text)

    status = (
        "success"
        if extracted_text.strip()
        else "needs_ocr"
    )

    return extracted_text, page_count, status


# ==================================================
# 4. SAVE EACH DOCUMENT TO tender_documents
# ==================================================

def save_document(
    tender_id,
    title,
    url,
    document_type,
    extracted_text,
    page_count,
    status,
    error_message=None
):
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO public.tender_documents (
                    tender_id,
                    tender_title,
                    document_url,
                    document_type,
                    extracted_text,
                    page_count,
                    extracted_characters,
                    extraction_status,
                    error_message,
                    extracted_at
                )
                VALUES (
                    :tender_id,
                    :title,
                    :url,
                    :document_type,
                    :content,
                    :pages,
                    :characters,
                    :status,
                    :error,
                    CURRENT_TIMESTAMP
                )
                ON CONFLICT (tender_id, document_url)
                DO UPDATE SET
                    tender_title = EXCLUDED.tender_title,
                    document_type = EXCLUDED.document_type,
                    extracted_text = EXCLUDED.extracted_text,
                    page_count = EXCLUDED.page_count,
                    extracted_characters =
                        EXCLUDED.extracted_characters,
                    extraction_status =
                        EXCLUDED.extraction_status,
                    error_message = EXCLUDED.error_message,
                    extracted_at = CURRENT_TIMESTAMP
            """),
            {
                "tender_id": tender_id,
                "title": title,
                "url": url,
                "document_type": document_type,
                "content": extracted_text,
                "pages": page_count,
                "characters": len(extracted_text or ""),
                "status": status,
                "error": error_message
            }
        )


# ==================================================
# 5. FETCH ALL TENDERS
# ==================================================

with engine.connect() as conn:
    tenders = conn.execute(text("""
        SELECT
            ten_id,
            title,
            bid_document,
            rfp,
            pre_bid_and_corrigendum,
            documents
        FROM public.gujrat_tender
        ORDER BY ten_id
    """)).mappings().all()

print(f"Total tenders: {len(tenders)}")


# ==================================================
# 6. PROCESS ALL TENDERS
# ==================================================

success_count = 0
skipped_count = 0
failed_count = 0

try:
    for tender in tenders:

        tender_id = tender["ten_id"]
        title = tender["title"]

        print(f"\n{'=' * 60}")
        print(f"Tender ID: {tender_id}")
        print(f"Title: {title}")

        # URL -> document type
        url_types = {}

        # --------------------------------------------------
        # A. URLS FROM THREE COLUMNS
        # --------------------------------------------------

        column_mapping = {
            "bid_document": "bid_document",
            "rfp": "rfp",
            "pre_bid_and_corrigendum":
                "pre_bid_and_corrigendum"
        }

        for column, document_type in column_mapping.items():

            url = tender[column]

            if isinstance(url, str) and url.strip().startswith("http"):
                url_types.setdefault(url.strip(), document_type)

        # --------------------------------------------------
        # B. URLS FROM documents JSON
        # --------------------------------------------------

        documents = tender["documents"] or {}

        if isinstance(documents, str):
            try:
                documents = json.loads(documents)
            except json.JSONDecodeError:
                documents = {}

        json_items = []

        if isinstance(documents, dict):
            items = documents.get("items", [])

            if isinstance(items, list):
                for item in items:

                    if not isinstance(item, dict):
                        continue

                    url = item.get("url")
                    label = item.get(
                        "label", "additional_document"
                    )

                    if not isinstance(url, str):
                        continue

                    url = url.strip()

                    if not url.startswith("http"):
                        continue

                    url_types.setdefault(url, str(label))

                    json_items.append({
                        "label": str(label),
                        "url": url
                    })

        print(f"Unique PDF URLs: {len(url_types)}")

        # URL -> extracted text
        extracted_by_url = {}

        # --------------------------------------------------
        # C. DOWNLOAD, EXTRACT AND SAVE EACH PDF
        # --------------------------------------------------

        for url, document_type in url_types.items():

            # Check for an already successfully extracted document
            with engine.connect() as conn:
                existing = conn.execute(
                    text("""
                        SELECT
                            extracted_text,
                            extraction_status,
                            page_count
                        FROM public.tender_documents
                        WHERE tender_id = :tid
                          AND document_url = :url
                    """),
                    {"tid": tender_id, "url": url}
                ).mappings().first()

            if (
                existing
                and existing["extraction_status"] == "success"
                and existing["extracted_text"]
            ):
                extracted_text = existing["extracted_text"]
                extracted_by_url[url] = extracted_text
                skipped_count += 1
                print(f"Already extracted; reusing: {url}")
                continue

            try:
                extracted_text, page_count, status = extract_pdf(
                    url, tender_id
                )

                save_document(
                    tender_id=tender_id,
                    title=title,
                    url=url,
                    document_type=document_type,
                    extracted_text=extracted_text,
                    page_count=page_count,
                    status=status
                )

                extracted_by_url[url] = extracted_text
                success_count += 1

                print(
                    f"Saved: {page_count} pages, "
                    f"{len(extracted_text)} characters, "
                    f"status={status}"
                )

                if status == "needs_ocr":
                    print("No selectable text found; OCR is needed.")

            except Exception as error:

                failed_count += 1

                print(f"Failed URL: {url}")
                print(f"Reason: {error}")

                # Record the failure without stopping other PDFs.
                save_document(
                    tender_id=tender_id,
                    title=title,
                    url=url,
                    document_type=document_type,
                    extracted_text=None,
                    page_count=None,
                    status="failed",
                    error_message=str(error)[:2000]
                )

        # --------------------------------------------------
        # D. MATCH TEXT TO ORIGINAL URL COLUMNS
        # --------------------------------------------------

        def get_column_text(column):
            url = tender[column]

            if not isinstance(url, str) or not url.strip():
                return None

            return extracted_by_url.get(url.strip())

        bid_text = get_column_text("bid_document")
        rfp_text = get_column_text("rfp")
        pre_bid_text = get_column_text(
            "pre_bid_and_corrigendum"
        )

        # --------------------------------------------------
        # E. ADD EXTRACTED TEXT TO EACH JSON DOCUMENT
        # --------------------------------------------------

        documents_text = []

        for item in json_items:

            url = item["url"]
            extracted_text = extracted_by_url.get(url)

            if url in extracted_by_url:
                status = (
                    "success"
                    if extracted_text and extracted_text.strip()
                    else "needs_ocr"
                )
            else:
                status = "failed"

            documents_text.append({
                "label": item["label"],
                "url": url,
                "extracted_text": extracted_text,
                "extraction_status": status
            })

        # --------------------------------------------------
        # F. UPDATE ORIGINAL TENDER ROW
        # --------------------------------------------------

        with engine.begin() as conn:
            conn.execute(
                text("""
                    UPDATE public.gujrat_tender
                    SET
                        bid_document_text = :bid_text,
                        rfp_text = :rfp_text,
                        pre_bid_and_corrigendum_text = :pre_bid_text,
                        documents_text = CAST(:docs_text AS JSONB)
                    WHERE ten_id = :tid
                """),
                {
                    "bid_text": bid_text,
                    "rfp_text": rfp_text,
                    "pre_bid_text": pre_bid_text,
                    "docs_text": json.dumps(
                        documents_text,
                        ensure_ascii=False
                    ),
                    "tid": tender_id
                }
            )

        print("Updated matching text columns in gujrat_tender.")

finally:
    session.close()
    engine.dispose()


# ==================================================
# 7. SUMMARY
# ==================================================

print(f"\n{'=' * 60}")
print("PROCESSING FINISHED")
print(f"Newly processed documents: {success_count}")
print(f"Previously extracted documents reused: {skipped_count}")
print(f"Failed documents: {failed_count}")
print("Text saved to both PostgreSQL tables.")