
import os
import re

from dotenv import load_dotenv

load_dotenv()

# Database
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
DB_SCHEMA = os.getenv("DB_SCHEMA", "public").strip()
CHUNKS_TABLE = os.getenv("CHUNKS_TABLE", "tender_chunks").strip()

# Database columns
CHUNK_ID_COLUMN = os.getenv("CHUNK_ID_COLUMN", "id").strip()
DOCUMENT_ID_COLUMN = os.getenv("DOCUMENT_ID_COLUMN", "tender_id").strip()
CHUNK_TEXT_COLUMN = os.getenv("CHUNK_TEXT_COLUMN", "chunk_text").strip()
EMBEDDING_COLUMN = os.getenv("EMBEDDING_COLUMN", "embedding_hf").strip()

DOCUMENT_TYPE_COLUMN = os.getenv("DOCUMENT_TYPE_COLUMN", "document_type").strip()
DOCUMENT_URL_COLUMN = os.getenv("DOCUMENT_URL_COLUMN", "document_url").strip()
CHUNK_INDEX_COLUMN = os.getenv("CHUNK_INDEX_COLUMN", "chunk_index").strip()

# Embedding model
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "nomic-ai/nomic-embed-text-v1.5"
)
EMBEDDING_DIMENSIONS = int(
    os.getenv("EMBEDDING_DIMENSIONS", "768")
)
EMBEDDING_QUERY_PREFIX = os.getenv(
    "EMBEDDING_QUERY_PREFIX", "search_query: "
)

# Retrieval
VECTOR_CANDIDATE_LIMIT = int(
    os.getenv("VECTOR_CANDIDATE_LIMIT", "30")
)
KEYWORD_CANDIDATE_LIMIT = int(
    os.getenv("KEYWORD_CANDIDATE_LIMIT", "30")
)
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "8"))
RRF_K = int(os.getenv("RRF_K", "60"))
EXHAUSTIVE_BATCH_SIZE = int(
    os.getenv("EXHAUSTIVE_BATCH_SIZE", "4")
)

# LM Studio
LM_STUDIO_BASE_URL = os.getenv(
    "LM_STUDIO_BASE_URL",
    "http://localhost:1234/v1"
).rstrip("/")

LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "").strip()
LM_STUDIO_TIMEOUT = int(os.getenv("LM_STUDIO_TIMEOUT", "600"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1800"))

# Citation configuration
CITATION_LABEL = os.getenv("CITATION_LABEL", "DB CHUNK ID")

# Document ID labels
DOCUMENT_ID_LABELS = [
    item.strip()
    for item in os.getenv(
        "DOCUMENT_ID_LABELS",
        "document,record,tender,contract,case,project"
    ).split(",")
    if item.strip()
]


def validate_identifier(value, setting_name, allow_empty=False):
    """Validate a configured PostgreSQL identifier."""
    if allow_empty and not value:
        return value

    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(
            f"Invalid SQL identifier for {setting_name}: {value!r}"
        )

    return value


def quote_identifier(value, setting_name, allow_empty=False):
    """Safely quote a validated PostgreSQL identifier."""
    value = validate_identifier(
        value, setting_name, allow_empty=allow_empty
    )
    return f'"{value}"' if value else ""


def qualified_table():
    """Return the configured schema-qualified chunk table."""
    return (
        f'{quote_identifier(DB_SCHEMA, "DB_SCHEMA")}.'
        f'{quote_identifier(CHUNKS_TABLE, "CHUNKS_TABLE")}'
    )

# Chunking diagnosis thresholds
CHUNK_MIN_CHARS = int(os.getenv("CHUNK_MIN_CHARS", "100"))
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "4000"))
CHUNK_WHITESPACE_RATIO = float(
    os.getenv("CHUNK_WHITESPACE_RATIO", "0.30")
)
CHUNK_NEIGHBOR_LIMIT = int(
    os.getenv("CHUNK_NEIGHBOR_LIMIT", "2")
)

# Conflict diagnosis
CONFLICT_MAX_CHUNKS = int(
    os.getenv("CONFLICT_MAX_CHUNKS", "8")
)

CONFLICT_MAX_CHARS_PER_CHUNK = int(
    os.getenv("CONFLICT_MAX_CHARS_PER_CHUNK", "2500")
)

CONFLICT_MAX_OUTPUT_TOKENS = int(
    os.getenv("CONFLICT_MAX_OUTPUT_TOKENS", "1800")
)


# Reasoning diagnosis configuration
REASONING_MAX_CHUNKS = int(
    os.getenv("REASONING_MAX_CHUNKS", "8")
)

REASONING_MAX_CHARS_PER_CHUNK = int(
    os.getenv("REASONING_MAX_CHARS_PER_CHUNK", "2500")
)

REASONING_MAX_OUTPUT_TOKENS = int(
    os.getenv("REASONING_MAX_OUTPUT_TOKENS", "1800")
)

REASONING_ANSWER_MAX_TOKENS = int(
    os.getenv("REASONING_ANSWER_MAX_TOKENS", "900")
)

# Exhaustive retrieval configuration
EXHAUSTIVE_BATCH_SIZE = int(
    os.getenv("EXHAUSTIVE_BATCH_SIZE", "500")
)

EXHAUSTIVE_CONTEXT_K = int(
    os.getenv("EXHAUSTIVE_CONTEXT_K", "8")
)