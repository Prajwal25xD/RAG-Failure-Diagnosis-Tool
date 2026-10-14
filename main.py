"""Main orchestration pipeline for the RAG Failure Diagnosis Tool.

Uses real PostgreSQL/pgvector retrieval, LM Studio answer generation,
four diagnostic modules, and JSON reports. All source identifiers are
configured outside this file.
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import config
from retrieval.hybrid_search import hybrid_search
from diagnostics.retrieval_diagnosis import diagnose_retrieval
from diagnostics.chunking_diagnosis import diagnose_chunking
from diagnostics.conflict_diagnosis import diagnose_conflicts
from diagnostics.reasoning_diagnosis import generate_rag_answer, diagnose_reasoning
from diagnostics.diagnostic_summary import build_diagnostic_summary


REPORT_DIR = Path(os.getenv("RAG_REPORT_DIR", "reports"))


def env_enabled(name, default="true"):
    return os.getenv(name, default).strip().lower() in {"true", "1", "yes", "on"}


SAVE_REPORTS = env_enabled("RAG_SAVE_REPORTS")
RUN_CHUNKING_DIAGNOSIS = env_enabled("RAG_RUN_CHUNKING")
RUN_CONFLICT_DIAGNOSIS = env_enabled("RAG_RUN_CONFLICT")
RUN_REASONING_DIAGNOSIS = env_enabled("RAG_RUN_REASONING")
INSPECT_CHUNK_NEIGHBORS = env_enabled("RAG_INSPECT_NEIGHBORS")


def print_section(title):
    print("\n" + "=" * 65)
    print(title)
    print("=" * 65)


def get_retrieval_settings(mode):
    """Set final context size; exhaustive retrieval has no candidate cap."""
    if mode == "fast":
        settings = {
            "candidate_limit": (
                config.VECTOR_CANDIDATE_LIMIT + config.KEYWORD_CANDIDATE_LIMIT
            ),
            "context_limit": config.FINAL_TOP_K,
        }
    elif mode == "exhaustive":
        settings = {
            "candidate_limit": None,
            "context_limit": int(
                os.getenv("RAG_EXHAUSTIVE_CONTEXT_K", str(config.FINAL_TOP_K))
            ),
        }
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if settings["context_limit"] <= 0:
        raise ValueError("Context limit must be positive.")
    return settings


def retrieve_evidence(question, document_id=None, mode="fast"):
    settings = get_retrieval_settings(mode)
    candidate_chunks = hybrid_search(
        question=question,
        document_id=document_id,
        top_k=settings["candidate_limit"],
        exhaustive=(mode == "exhaustive"),
    )
    context_chunks = candidate_chunks[: settings["context_limit"]]
    return candidate_chunks, context_chunks, settings


def run_diagnostics(
    question,
    answer,
    candidate_chunks,
    context_chunks,
    context_limit,
    supporting_chunk_ids=None,
):
    """Run retrieval over candidates and other checks over answer context."""
    results = {}

    print_section("1. RETRIEVAL DIAGNOSIS")
    try:
        retrieval_report = diagnose_retrieval(
            retrieved_chunks=candidate_chunks,
            supporting_chunk_ids=supporting_chunk_ids,
            top_k=context_limit,
        )
        results["retrieval"] = retrieval_report
        print("Status:", retrieval_report.get("failure_type", retrieval_report.get("status", "unknown")))
        print("Recall@K:", retrieval_report.get("recall_at_k"))
        print("Reason:", retrieval_report.get("reason"))
    except Exception as exc:
        results["retrieval"] = {"status": "error", "error": str(exc)}
        print("Retrieval diagnosis error:", exc)

    print_section("2. CHUNKING DIAGNOSIS")
    if RUN_CHUNKING_DIAGNOSIS:
        try:
            chunking_report = diagnose_chunking(
                retrieved_chunks=context_chunks,
                inspect_neighbors=INSPECT_CHUNK_NEIGHBORS,
            )
            results["chunking"] = chunking_report
            print("Status:", chunking_report.get("status"))
            print("Issue counts:", chunking_report.get("issue_counts"))
        except Exception as exc:
            results["chunking"] = {"status": "error", "error": str(exc)}
            print("Chunking diagnosis error:", exc)
    else:
        results["chunking"] = {"status": "skipped"}
        print("Skipped by configuration.")

    print_section("3. CONFLICT DIAGNOSIS")
    if RUN_CONFLICT_DIAGNOSIS:
        try:
            conflict_report = diagnose_conflicts(
                question=question,
                retrieved_chunks=context_chunks,
            )
            results["conflict"] = conflict_report
            print("Status:", conflict_report.get("status"))
            print("Potential conflicts:", conflict_report.get("total_conflicts"))
            for document_report in conflict_report.get("document_reports", []):
                print(
                    "Document:", document_report.get("document_id"),
                    "| Status:", document_report.get("status"),
                )
        except Exception as exc:
            results["conflict"] = {"status": "error", "error": str(exc)}
            print("Conflict diagnosis error:", exc)
    else:
        results["conflict"] = {"status": "skipped"}
        print("Skipped by configuration.")

    print_section("4. REASONING DIAGNOSIS")
    if RUN_REASONING_DIAGNOSIS and answer:
        try:
            reasoning_report = diagnose_reasoning(
                question=question,
                generated_answer=answer,
                retrieved_chunks=context_chunks,
            )
            results["reasoning"] = reasoning_report
            print("Status:", reasoning_report.get("status"))
            print("Findings:", len(reasoning_report.get("findings") or []))
            print("Summary:", reasoning_report.get("summary"))
        except Exception as exc:
            results["reasoning"] = {"status": "error", "error": str(exc)}
            print("Reasoning diagnosis error:", exc)
    else:
        results["reasoning"] = {
            "status": "skipped",
            "reason": "Disabled or no generated answer.",
        }
        print("Skipped.")

    return results


def normalized_reports(diagnostics, supporting_chunk_ids=None):
    """Normalize retrieval's failure_type key for consistent classification.

    Never infer ground-truth evidence from model-generated citations.
    """
    reports = {
        name: dict(diagnostics.get(name) or {})
        for name in ("retrieval", "chunking", "conflict", "reasoning")
    }
    retrieval = reports["retrieval"]
    if retrieval.get("status") in (None, "unknown") and retrieval.get("failure_type"):
        retrieval["status"] = retrieval["failure_type"]
    # The summary module can recognize explicitly provided ground truth.
    if supporting_chunk_ids:
        retrieval["verified_supporting_chunk_ids"] = list(supporting_chunk_ids)
    return reports


def determine_overall_status(
    retrieval_report,
    chunking_report,
    conflict_report,
    reasoning_report,
    verified_supporting_ids=None,
):
    """Keep unverified findings and chunking heuristics out of confirmed failures."""
    retrieval_status = retrieval_report.get("status", "unknown")
    chunking_status = chunking_report.get("status", "unknown")
    conflict_status = conflict_report.get("status", "unknown")
    reasoning_status = reasoning_report.get("status", "unknown")

    retrieval_failures = {
        "retrieval_miss", "ranking_failure", "mixed_retrieval_failure"
    }
    if retrieval_status in retrieval_failures and verified_supporting_ids:
        return "confirmed_failure"
    if retrieval_status in retrieval_failures:
        return "potential_failure"

    reasoning_failure_statuses = {
        "reasoning_error", "unsupported_claim", "contradicted_claim",
        "incorrect_interpretation", "missing_important_evidence",
        "issues_detected", "reasoning_issues_detected",
        "potential_reasoning_error",
    }
    if reasoning_status in reasoning_failure_statuses or reasoning_report.get("findings"):
        return "potential_failure"

    conflict_failure_statuses = {
        "conflict_detected", "conflicts_detected", "potential_conflicts_detected"
    }
    if (
        conflict_status in conflict_failure_statuses
        or (conflict_report.get("total_conflicts") or 0) > 0
    ):
        return "potential_failure"

    if (
        retrieval_status == "retrieval_success"
        and chunking_status in {"no_issues_detected", "no_issues", "healthy"}
        and conflict_status in {"no_conflict_detected", "no_conflicts_detected"}
        and reasoning_status == "supported"
    ):
        return "all_checks_passed"

    return "no_confirmed_failure_detected"


def print_diagnostic_summary(diagnostic_summary):
    print_section("WARNING COUNT AND DIAGNOSTIC CONFIDENCE")
    warnings = diagnostic_summary["warning_summary"]
    confidence = diagnostic_summary["diagnostic_confidence"]

    print("\nWARNING SUMMARY")
    print("Total warnings:", warnings["total_warnings"])
    print("Chunking warnings:", warnings["chunking_warnings"])
    print("Potential conflicts:", warnings["conflict_findings"])
    print("Reasoning findings:", warnings["reasoning_findings"])
    print("Chunking issue types:", warnings["chunking_issue_counts"])

    print("\nDIAGNOSTIC CONFIDENCE")
    print("Level:", confidence["level"])
    print("Check coverage:", f'{confidence["check_coverage_percent"]}%')
    print("Checks passed:", f'{confidence["passed_checks"]}/{confidence["total_checks"]}')
    print("Unresolved checks:", ", ".join(confidence["unresolved_checks"]) or "None")
    print("Flagged checks:", ", ".join(confidence["flagged_checks"]) or "None")
    print("Explanation:", confidence["explanation"])

    print("\nCHECK DETAILS")
    for name, detail in diagnostic_summary["check_details"].items():
        print(f'{name.capitalize()}: {detail["result"]} - {detail["explanation"]}')


def save_report(report):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = REPORT_DIR / f"rag_diagnosis_{timestamp}.json"
    with path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False, default=str)
    return str(path)


def run_rag_pipeline(question, document_id=None, mode="fast", supporting_chunk_ids=None):
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    started_at = datetime.now().isoformat()
    print_section("RAG FAILURE DIAGNOSIS TOOL")
    print("Mode:", mode)
    print("Question:", question)
    print("Document filter:", document_id or "All documents")

    print_section("STEP 1 — HYBRID RETRIEVAL")
    candidate_chunks, context_chunks, settings = retrieve_evidence(
        question=question, document_id=document_id, mode=mode
    )
    print("Candidate chunks:", len(candidate_chunks))
    print("Answer context chunks:", len(context_chunks))

    print_section("STEP 2 — ANSWER GENERATION")
    answer = None
    answer_error = None
    if context_chunks:
        try:
            answer = generate_rag_answer(question=question, retrieved_chunks=context_chunks)
            print("\nGENERATED ANSWER:\n")
            print(answer)
        except Exception as exc:
            answer_error = str(exc)
            print("Answer generation failed:", exc)
    else:
        print("No evidence retrieved. Cannot generate an answer.")

    print_section("STEP 3 — DIAGNOSTIC ANALYSIS")
    diagnostics = run_diagnostics(
        question=question,
        answer=answer,
        candidate_chunks=candidate_chunks,
        context_chunks=context_chunks,
        context_limit=settings["context_limit"],
        supporting_chunk_ids=supporting_chunk_ids,
    )

    reports = normalized_reports(diagnostics, supporting_chunk_ids)
    diagnostic_summary = build_diagnostic_summary(
        retrieval_report=reports["retrieval"],
        chunking_report=reports["chunking"],
        conflict_report=reports["conflict"],
        reasoning_report=reports["reasoning"],
    )
    overall_status = determine_overall_status(
        reports["retrieval"],
        reports["chunking"],
        reports["conflict"],
        reports["reasoning"],
        verified_supporting_ids=supporting_chunk_ids,
    )
    if answer_error:
        overall_status = "answer_generation_error"
    elif not context_chunks:
        overall_status = "no_evidence_retrieved"

    report = {
        "metadata": {
            "started_at": started_at,
            "completed_at": datetime.now().isoformat(),
            "mode": mode,
            "question": question,
            "document_id": document_id,
            "model": config.LM_STUDIO_MODEL,
            "embedding_model": config.EMBEDDING_MODEL_NAME,
            "embedding_column": config.EMBEDDING_COLUMN,
            "candidate_count": len(candidate_chunks),
            "context_count": len(context_chunks),
        },
        "answer": {"text": answer, "error": answer_error},
        "retrieved_chunks": [
            {
                "id": chunk.get("id"),
                "document_id": chunk.get("document_id"),
                "document_type": chunk.get("document_type"),
                "document_url": chunk.get("document_url"),
                "chunk_index": chunk.get("chunk_index"),
                "vector_rank": chunk.get("vector_rank"),
                "keyword_rank": chunk.get("keyword_rank"),
                "rrf_score": chunk.get("rrf_score"),
                "chunk_text": chunk.get("chunk_text"),
            }
            for chunk in context_chunks
        ],
        "diagnostics": diagnostics,
        "diagnostic_summary": diagnostic_summary,
        "overall_status": overall_status,
    }

    print_section("FINAL DIAGNOSIS")
    print("Overall status:", overall_status)
    for name, result in reports.items():
        print(f'{name.capitalize()}: {result.get("status", "unknown")}')
    print_diagnostic_summary(diagnostic_summary)

    if SAVE_REPORTS:
        try:
            report_path = save_report(report)
            report["report_path"] = report_path
            print("\nReport saved:", report_path)
        except Exception as exc:
            report["report_save_error"] = str(exc)
            print("Report save failed:", exc)
    return report


def main():
    parser = argparse.ArgumentParser(description="Run an end-to-end RAG failure diagnosis.")
    parser.add_argument("--question", type=str, default=None, help="Question to ask the RAG system.")
    parser.add_argument("--document-id", type=str, default=None, help="Optional document ID filter.")
    parser.add_argument("--mode", choices=["fast", "exhaustive"], default=None, help="Retrieval mode.")
    parser.add_argument(
        "--supporting-chunk-ids", nargs="+", default=None,
        help="Independently verified supporting chunk IDs for retrieval diagnosis.",
    )
    args = parser.parse_args()

    question = args.question
    if question is None:
        question = input("Enter your question: ").strip()

    document_id = args.document_id
    if document_id is None and args.question is None:
        document_id = input("Document ID (optional): ").strip() or None

    mode = args.mode
    if mode is None:
        mode = input("Mode [fast/exhaustive] (default fast): ").strip().lower() or "fast"
    if mode not in {"fast", "exhaustive"}:
        raise ValueError("Mode must be fast or exhaustive.")

    run_rag_pipeline(
        question=question,
        document_id=document_id,
        mode=mode,
        supporting_chunk_ids=args.supporting_chunk_ids,
    )


if __name__ == "__main__":
    main()
