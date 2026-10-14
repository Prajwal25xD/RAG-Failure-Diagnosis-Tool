"""Dynamic routing between structured tender records and the existing RAG pipeline.

No user questions, tender IDs, or evidence are hardcoded. Read-only SQL.
"""
from __future__ import annotations

import os
import re
from datetime import datetime

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import URL

_ID_PATTERN = re.compile(r'\b(?:tender\s*(?:id|number|no\.?|#)?|ten_id)\s*[:#=\-]?\s*(\d+)\b', re.I)
_DOC_WORDS = re.compile(r'\b(pdf|document|clause|specification|requirement|cement|concrete|technical|scope|boq|rfp|corrigendum|bid document|explain|summariz(?:e|ation))\b', re.I)
_META_WORDS = re.compile(r'\b(title|date|deadline|opening|closing|pre.bid|eligib|notes|url|link|details?|information|status|list|show|find|which|compare)\b', re.I)


def database_engine():
    import config
    from retrieval.vector_search import get_engine
    return get_engine()


def table_name():
    return os.getenv('RAG_TENDER_METADATA_TABLE', 'gujrat_tender')


def _metadata_table(conn):
    """Resolve only an existing, introspected table; quote identifiers safely."""
    configured = table_name()
    if '.' in configured:
        schema, table = configured.split('.', 1)
    else:
        schema, table = 'public', configured
    inspector = inspect(conn)
    if not inspector.has_table(table, schema=schema):
        raise RuntimeError(f'Metadata table {schema}.{table} does not exist')
    cols = {col['name'] for col in inspector.get_columns(table, schema=schema)}
    if 'ten_id' not in cols:
        raise RuntimeError(f'Metadata table {schema}.{table} must contain ten_id')
    preparer = conn.dialect.identifier_preparer
    return f'{preparer.quote_schema(schema)}.{preparer.quote(table)}', cols


def _records(sql, params):
    with database_engine().connect() as conn:
        table, cols = _metadata_table(conn)
        query = sql(table, cols)
        return [dict(row) for row in conn.execute(text(query), params).mappings()]


def get_tenders(ids):
    ids = list(dict.fromkeys(int(x) for x in ids))
    if not ids:
        return []
    def sql(table, cols):
        return f'SELECT * FROM {table} WHERE ten_id = ANY(:ids) ORDER BY ten_id'
    return _records(sql, {'ids': ids})


def search_metadata(question, limit=20):
    """Search title, eligibility and notes using literal keyword matching."""
    words = re.findall(r'[\w]+', question.lower())
    stop = {'which','show','give','me','all','the','tenders','tender','related','to','about','with','have','find','list','what','are','is','in','for','details','information','of','and','on','whose','that','mention'}
    keywords = [w for w in words if len(w) >= 3 and w not in stop][:6]
    if not keywords:
        return []
    def sql(table, cols):
        fields = [col for col in ('title', 'notes', 'eligibility_note') if col in cols]
        if not fields:
            return f'SELECT * FROM {table} WHERE false'
        preparer = database_engine().dialect.identifier_preparer
        clause = ' OR '.join(f'COALESCE({preparer.quote(c)}, \'\') ILIKE :term' for c in fields)
        return f'SELECT * FROM {table} WHERE ({clause}) ORDER BY ten_id LIMIT :lim'
    return _records(sql, {'term': '%' + max(keywords, key=len) + '%', 'lim': max(1, min(int(limit), 100))})


def extract_tender_ids(question):
    return list(dict.fromkeys(int(x) for x in _ID_PATTERN.findall(question)))


def route_question(question):
    """Use SQL for clear structured requests, RAG for document/general requests."""
    ids = extract_tender_ids(question)
    has_doc = bool(_DOC_WORDS.search(question))
    has_meta = bool(_META_WORDS.search(question))
    if ids and (has_meta or not has_doc):
        if has_doc and not re.search(r'\b(details?|information|title|date|deadline|eligib|notes|compare)\b', question, re.I):
            return 'rag', ids
        return 'metadata_ids', ids
    if ids:
        return 'rag', ids
    if has_meta and re.search(r'\b(title|eligib|notes|list|show|find|which)\b', question, re.I):
        return 'metadata_search', []
    return 'rag', []


def metadata_report(question, records, mode, route):
    if not records:
        answer = 'No matching tender metadata records were found in the configured PostgreSQL metadata table.'
    else:
        lines = [f'Found {len(records)} matching tender record(s) in PostgreSQL.']
        for record in records:
            lines.append(f"\n### Tender {record.get('ten_id')}")
            for key, val in record.items():
                if val is not None and key not in {'data_json', 'documents'}:
                    lines.append(f'- **{key.replace("_", " ").title()}**: {val}')
            for key in ('documents', 'data_json'):
                if record.get(key) is not None:
                    lines.append(f'- **{key}**: Available in structured record (see JSON report)')
        answer = '\n'.join(lines)
    return {
        'metadata': {'started_at': datetime.now().isoformat(), 'mode': mode, 'question': question,
                     'model': 'PostgreSQL structured query (no LLM)', 'candidate_count': len(records),
                     'context_count': 0, 'query_route': route},
        'answer': {'text': answer, 'error': None},
        'retrieved_chunks': [],
        'structured_records': records,
        'diagnostics': {},
        'diagnostic_summary': {},
        'overall_status': 'structured_query_complete' if records else 'no_matching_records',
    }


def run_dynamic_query(question, mode='fast', document_id=None, supporting_chunk_ids=None):
    """Return a report compatible with app.py. Do not fabricate diagnostic results."""
    from main import run_rag_pipeline
    if document_id is not None and str(document_id).strip():
        return run_rag_pipeline(question=question, document_id=str(document_id).strip(), mode=mode,
                                supporting_chunk_ids=supporting_chunk_ids)
    route, ids = route_question(question)
    if route == 'metadata_ids':
        return metadata_report(question, get_tenders(ids), mode, route)
    if route == 'metadata_search':
        records = search_metadata(question)
        if records:
            return metadata_report(question, records, mode, route)
        # No lexical metadata hit does not imply absence from PDF content.
        return run_rag_pipeline(question=question, mode=mode, supporting_chunk_ids=supporting_chunk_ids)
    if len(ids) == 1:
        return run_rag_pipeline(question=question, document_id=str(ids[0]), mode=mode,
                                supporting_chunk_ids=supporting_chunk_ids)
    # Multiple IDs: preserve user question and search broadly rather than choosing one arbitrarily.
    return run_rag_pipeline(question=question, mode=mode, supporting_chunk_ids=supporting_chunk_ids)
