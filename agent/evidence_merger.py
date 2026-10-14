"""Generate grounded answers from structured metadata and indexed PDF evidence."""
import json
import os
import requests


def synthesize(question, records, chunks):
    import config
    max_records = int(os.getenv('RAG_ANSWER_MAX_RECORDS', '20'))
    safe_records = [{k: v for k, v in r.items() if k not in {'data_json', 'documents'}} for r in records[:max_records]]
    safe_chunks = [{'chunk_id': c.get('id'), 'tender_id': c.get('document_id', c.get('tender_id')),
                    'document_type': c.get('document_type'), 'text': str(c.get('chunk_text', ''))[:2200]}
                   for c in chunks[:int(os.getenv('RAG_ANSWER_CONTEXT_K', '8'))]]
    if not safe_records and not safe_chunks:
        return 'No matching evidence was found in the selected PostgreSQL metadata and indexed documents.'
    evidence = json.dumps({'metadata_records': safe_records, 'document_chunks': safe_chunks}, default=str, ensure_ascii=False)
    messages = [
        {'role': 'system', 'content': 'Answer only from the supplied PostgreSQL records and document chunks. '
         'Cite tender IDs for metadata and chunk IDs for PDF statements. '
         'If evidence is incomplete, state the limitation. Never invent tender details.'},
        {'role': 'user', 'content': f'Question: {question}\n\nEvidence:\n{evidence[:60000]}'},
    ]
    base = os.getenv('LM_STUDIO_BASE_URL', 'http://localhost:1234/v1').rstrip('/')
    response = requests.post(f'{base}/chat/completions', json={
        'model': config.LM_STUDIO_MODEL, 'messages': messages, 'temperature': 0.1,
        'max_tokens': int(os.getenv('LLM_MAX_TOKENS', '1800')),
    }, timeout=int(os.getenv('LM_STUDIO_TIMEOUT', '600')))
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']
