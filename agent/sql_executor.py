"""Read-only parameterized metadata retrieval. No LLM-authored SQL is executed."""
import os
from sqlalchemy import inspect, text
from retrieval.vector_search import get_engine


def metadata_schema():
    schema = os.getenv('RAG_TENDER_METADATA_SCHEMA', 'public')
    table = os.getenv('RAG_TENDER_METADATA_TABLE', 'gujrat_tender')
    engine = get_engine()
    with engine.connect() as conn:
        inspector = inspect(conn)
        if not inspector.has_table(table, schema=schema):
            raise RuntimeError(f'Metadata table {schema}.{table} not found')
        columns = {c['name']: c for c in inspector.get_columns(table, schema=schema)}
    return schema, table, columns


def execute_metadata_plan(plan):
    schema, table, cols = metadata_schema()
    engine = get_engine()
    max_rows = max(1, min(int(os.getenv('RAG_SQL_MAX_ROWS', '100')), 500))
    timeout_ms = max(100, min(int(os.getenv('RAG_SQL_TIMEOUT_MS', '10000')), 120000))
    with engine.connect() as conn:
        quote = conn.dialect.identifier_preparer.quote
        name = f'{quote(schema)}.{quote(table)}'
        id_col = os.getenv('RAG_TENDER_ID_COLUMN', 'ten_id')
        if id_col not in cols:
            raise RuntimeError(f'Missing tender ID column: {id_col}')
        clauses, params = [], {}
        if plan['tender_ids']:
            clauses.append(f'{quote(id_col)} = ANY(:ids)')
            params['ids'] = plan['tender_ids']
        for idx, flt in enumerate(plan['filters']):
            col = flt['column']
            if col not in cols:
                raise ValueError(f'Column not permitted: {col}')
            op, value = flt['operator'], flt['value']
            key = f'p{idx}'
            # Do not compare dates numerically while their PostgreSQL type is TEXT.
            if op in {'gte', 'lte'} and str(cols[col]['type']).upper() in {'TEXT', 'VARCHAR', 'CHAR'}:
                raise ValueError(f'Cannot safely compare text-based dates in {col}; normalize the column first')
            if op == 'contains':
                clauses.append(f'CAST({quote(col)} AS TEXT) ILIKE :{key}')
                params[key] = '%' + str(value).replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_') + '%'
                clauses[-1] += " ESCAPE '\\'"
            else:
                clauses.append(f'{quote(col)} { {"eq":"=", "gte":">=", "lte":"<="}[op] } :{key}')
                params[key] = value
        if plan['search_text'] and not plan['tender_ids']:
            fields = [c for c in ('title', 'notes', 'eligibility_note') if c in cols]
            if fields:
                clauses.append('(' + ' OR '.join(f'CAST({quote(c)} AS TEXT) ILIKE :search' for c in fields) + ')')
                params['search'] = '%' + plan['search_text'] + '%'
        where = ' WHERE ' + ' AND '.join(clauses) if clauses else ''
        if plan['operation'] == 'count':
            sql = f'SELECT COUNT(*) AS matching_tenders FROM {name}{where}'
        else:
            sql = f'SELECT * FROM {name}{where} ORDER BY {quote(id_col)} LIMIT :limit'
            params['limit'] = max_rows
        # Transaction-local timeout; use a dedicated read-only DB account in production.
        with conn.begin():
            conn.execute(text('SET TRANSACTION READ ONLY'))
            conn.execute(text("SELECT set_config('statement_timeout', :timeout, true)"), {'timeout': str(timeout_ms)})
            records = [dict(row) for row in conn.execute(text(sql), params).mappings()]
    return records
