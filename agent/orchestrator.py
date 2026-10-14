"""Dynamic SQL/RAG orchestration with existing diagnostic pipeline compatibility."""
from datetime import datetime
from agent.query_planner import plan_question
from agent.sql_executor import metadata_schema, execute_metadata_plan
from agent.evidence_merger import synthesize


def run_dynamic_query(question, mode='fast', document_id=None, supporting_chunk_ids=None):
    from main import run_rag_pipeline, retrieve_evidence, run_diagnostics, determine_overall_status
    from diagnostics.diagnostic_summary import build_diagnostic_summary
    if not question or not question.strip():
        raise ValueError('Question cannot be empty')
    _, _, columns = metadata_schema()
    plan = plan_question(question, list(columns))
    if document_id is not None and str(document_id).strip():
        manual_id = int(document_id)
        plan['tender_ids'] = [manual_id]
    route = plan['route']
    records = execute_metadata_plan(plan) if route in {'sql', 'hybrid'} else []
    # Preserve existing RAG diagnostics for document-only queries.
    if route == 'rag' and len(plan['tender_ids']) <= 1:
        result = run_rag_pipeline(question=plan['document_query'],
            document_id=str(plan['tender_ids'][0]) if plan['tender_ids'] else None,
            mode=mode, supporting_chunk_ids=supporting_chunk_ids)
        result['query_plan'] = plan
        return result
    candidates, chunks = [], []
    if route in {'rag', 'hybrid'}:
        # With multiple IDs, query each separately to prevent cross-tender mixing.
        ids = plan['tender_ids'] or [None]
        for tender_id in ids:
            found, selected, _ = retrieve_evidence(plan['document_query'],
                document_id=str(tender_id) if tender_id is not None else None, mode=mode)
            candidates.extend(found)
            chunks.extend(selected)
        seen = set()
        chunks = [c for c in chunks if not (c.get('id') in seen or seen.add(c.get('id')))]
        chunks = chunks[:int(__import__('os').getenv('RAG_ANSWER_CONTEXT_K', '8'))]
    answer = synthesize(question, records, chunks)
    diagnostics, summary = {}, {}
    if chunks:
        diagnostics = run_diagnostics(question, answer, candidates, chunks,
            len(chunks), supporting_chunk_ids=supporting_chunk_ids)
        summary = build_diagnostic_summary(
            retrieval_report=diagnostics.get('retrieval', {}),
            chunking_report=diagnostics.get('chunking', {}),
            conflict_report=diagnostics.get('conflict', {}),
            reasoning_report=diagnostics.get('reasoning', {}))
        status = determine_overall_status(
            diagnostics.get('retrieval', {}), diagnostics.get('chunking', {}),
            diagnostics.get('conflict', {}), diagnostics.get('reasoning', {}))
    else:
        status = 'structured_query_complete' if records else 'no_matching_records'
    return {
        'metadata': {'started_at': datetime.now().isoformat(), 'mode': mode,
            'question': question, 'model': 'LM Studio', 'candidate_count': len(candidates),
            'context_count': len(chunks), 'query_route': route},
        'answer': {'text': answer, 'error': None},
        'structured_records': records,
        'retrieved_chunks': [{k: c.get(k) for k in ('id', 'document_id', 'document_type',
            'document_url', 'chunk_index', 'chunk_text', 'vector_rank', 'keyword_rank', 'rrf_score')}
            for c in chunks],
        'diagnostics': diagnostics, 'diagnostic_summary': summary,
        'overall_status': status, 'query_plan': plan,
    }
