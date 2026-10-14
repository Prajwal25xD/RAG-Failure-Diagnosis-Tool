
"""Diagnose chunk quality using real PostgreSQL evidence."""

import re
from collections import Counter

from sqlalchemy import text

import config
from retrieval.vector_search import get_engine


def get_neighboring_chunks(chunk: dict, limit: int | None = None) -> list[dict]:
    """Retrieve neighboring chunks from the same document and source."""

    if limit is None:
        limit = config.CHUNK_NEIGHBOR_LIMIT

    if limit < 0:
        raise ValueError("Neighbor limit cannot be negative.")

    if limit == 0:
        return []

    document_id = chunk.get("document_id")
    chunk_index = chunk.get("chunk_index")

    if document_id is None or chunk_index is None:
        return []

    table = config.qualified_table()

    id_col = config.quote_identifier(
        config.CHUNK_ID_COLUMN, "CHUNK_ID_COLUMN"
    )
    doc_col = config.quote_identifier(
        config.DOCUMENT_ID_COLUMN, "DOCUMENT_ID_COLUMN"
    )
    index_col = config.quote_identifier(
        config.CHUNK_INDEX_COLUMN, "CHUNK_INDEX_COLUMN",
        allow_empty=True,
    )
    text_col = config.quote_identifier(
        config.CHUNK_TEXT_COLUMN, "CHUNK_TEXT_COLUMN"
    )

    if not index_col:
        return []

    type_col = config.quote_identifier(
        config.DOCUMENT_TYPE_COLUMN, "DOCUMENT_TYPE_COLUMN",
        allow_empty=True,
    )
    url_col = config.quote_identifier(
        config.DOCUMENT_URL_COLUMN, "DOCUMENT_URL_COLUMN",
        allow_empty=True,
    )

    # Avoid treating chunks from different source documents as neighbors.
    source_filters = []
    parameters = {
        "document_id": str(document_id),
        "chunk_index": int(chunk_index),
        "neighbor_limit": limit,
        "chunk_id": str(chunk["id"]),
    }

    if type_col:
        source_filters.append(
            f"c.{type_col} IS NOT DISTINCT FROM :document_type"
        )
        parameters["document_type"] = chunk.get("document_type")

    if url_col:
        source_filters.append(
            f"c.{url_col} IS NOT DISTINCT FROM :document_url"
        )
        parameters["document_url"] = chunk.get("document_url")

    extra_filter = (
        " AND " + " AND ".join(source_filters)
        if source_filters else ""
    )

    sql = text(f"""
        SELECT
            c.{id_col} AS id,
            c.{doc_col} AS document_id,
            c.{index_col} AS chunk_index,
            c.{text_col} AS chunk_text
        FROM {table} c
        WHERE CAST(c.{doc_col} AS text) = :document_id
          AND CAST(c.{id_col} AS text) != :chunk_id
          AND c.{index_col} BETWEEN
              :chunk_index - :neighbor_limit
              AND :chunk_index + :neighbor_limit
          {extra_filter}
        ORDER BY c.{index_col}, c.{id_col}
    """)

    with get_engine().connect() as connection:
        rows = connection.execute(sql, parameters).mappings().all()

    return [dict(row) for row in rows]


def analyze_chunk(chunk: dict) -> dict:
    """Analyze one real chunk for possible quality issues."""

    raw_text = str(chunk.get("chunk_text") or "")
    cleaned_text = re.sub(r"\s+", " ", raw_text).strip()

    char_count = len(cleaned_text)
    raw_length = len(raw_text)

    whitespace_ratio = (
        sum(character.isspace() for character in raw_text) / raw_length
        if raw_length else 0.0
    )

    issues = []

    if not cleaned_text:
        issues.append("empty_chunk")

    elif char_count < config.CHUNK_MIN_CHARS:
        issues.append("short_chunk")

    if char_count > config.CHUNK_MAX_CHARS:
        issues.append("long_chunk")

    if whitespace_ratio > config.CHUNK_WHITESPACE_RATIO:
        issues.append("excessive_whitespace")

    if (
        cleaned_text
        and cleaned_text[0].islower()
    ):
        issues.append("possible_incomplete_start")

    if (
        cleaned_text
        and cleaned_text[-1].isalnum()
        and not cleaned_text.endswith((".", "!", "?", ":", ";"))
    ):
        issues.append("possible_incomplete_end")

    return {
        "chunk_id": chunk.get("id"),
        "document_id": chunk.get("document_id"),
        "chunk_index": chunk.get("chunk_index"),
        "character_count": char_count,
        "whitespace_ratio": round(whitespace_ratio, 4),
        "issues": issues,
    }


def diagnose_chunking(
    retrieved_chunks: list[dict],
    inspect_neighbors: bool = True,
) -> dict:
    """Analyze retrieved chunks and optionally inspect their neighbors."""

    if not retrieved_chunks:
        return {
            "status": "no_chunks",
            "analyzed_chunks": 0,
            "issue_counts": {},
            "chunk_reports": [],
        }

    reports = []
    issue_counts = Counter()
    seen_texts = {}

    for chunk in retrieved_chunks:
        report = analyze_chunk(chunk)

        normalized_text = re.sub(
            r"\s+", " ", str(chunk.get("chunk_text") or "")
        ).strip().casefold()

        if normalized_text:
            if normalized_text in seen_texts:
                report["issues"].append("duplicate_chunk_text")
                report["duplicate_of"] = seen_texts[normalized_text]
            else:
                seen_texts[normalized_text] = chunk.get("id")

        if inspect_neighbors:
            neighbors = get_neighboring_chunks(chunk)
            report["neighbor_count"] = len(neighbors)
            report["neighbor_chunk_ids"] = [
                neighbor["id"] for neighbor in neighbors
            ]

            if chunk.get("chunk_index") is not None:
                current_index = int(chunk["chunk_index"])
                neighbor_indices = {
                    int(neighbor["chunk_index"])
                    for neighbor in neighbors
                    if neighbor["chunk_index"] is not None
                }

                report["previous_chunk_found"] = (
                    current_index - 1 in neighbor_indices
                )
                report["next_chunk_found"] = (
                    current_index + 1 in neighbor_indices
                )

        for issue in report["issues"]:
            issue_counts[issue] += 1

        reports.append(report)

    return {
        "status": (
            "potential_issues_detected"
            if issue_counts else "no_issues_detected"
        ),
        "analyzed_chunks": len(reports),
        "issue_counts": dict(issue_counts),
        "chunk_reports": reports,
        "note": (
            "Heuristic findings require inspection of source context. "
            "A missing neighbor can also indicate a document boundary."
        ),
    }
