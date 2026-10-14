
"""
Diagnostic Summary
==================

Summarizes the four RAG diagnostic modules.

Features:
- Warning counts
- Confirmed and potential failure counts
- Check coverage
- Diagnostic confidence classification
- Unresolved checks
- Human-readable explanations

No hardcoded document IDs, questions, or chunk IDs.
"""

from collections import Counter


def _as_dict(value):
    return value if isinstance(value, dict) else {}


def _as_list(value):
    return value if isinstance(value, list) else []


def _as_count(value):
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def build_diagnostic_summary(
    retrieval_report,
    chunking_report,
    conflict_report,
    reasoning_report,
):
    retrieval = _as_dict(retrieval_report)
    chunking = _as_dict(chunking_report)
    conflict = _as_dict(conflict_report)
    reasoning = _as_dict(reasoning_report)

    statuses = {
        "retrieval": retrieval.get("status", "unknown"),
        "chunking": chunking.get("status", "unknown"),
        "conflict": conflict.get("status", "unknown"),
        "reasoning": reasoning.get("status", "unknown"),
    }

    # -----------------------------------------
    # 1. Chunking warnings
    # -----------------------------------------

    raw_issue_counts = _as_dict(
        chunking.get("issue_counts")
    )

    chunking_issue_counts = {
        str(name): _as_count(count)
        for name, count in raw_issue_counts.items()
    }

    chunking_warning_count = sum(
        chunking_issue_counts.values()
    )

    # -----------------------------------------
    # 2. Reasoning findings
    # -----------------------------------------

    findings = _as_list(
        reasoning.get("findings")
    )

    reasoning_issue_types = Counter()

    for finding in findings:
        if isinstance(finding, dict):
            issue_type = (
                finding.get("type")
                or finding.get("issue_type")
                or "unspecified"
            )
        else:
            issue_type = "unspecified"

        reasoning_issue_types[str(issue_type)] += 1

    reasoning_warning_count = len(findings)

    # -----------------------------------------
    # 3. Evidence conflicts
    # -----------------------------------------

    conflict_warning_count = _as_count(
        conflict.get("total_conflicts", 0)
    )

    # -----------------------------------------
    # 4. Retrieval
    # -----------------------------------------

    retrieval_failure_statuses = {
        "retrieval_miss",
        "ranking_failure",
        "mixed_retrieval_failure",
    }

    retrieval_failure = (
        statuses["retrieval"]
        in retrieval_failure_statuses
    )

    # Retrieval failures are considered verified
    # only when supporting IDs were independently
    # supplied and validated by the calling pipeline.

    verified_retrieval_failure = (
        retrieval_failure
        and bool(
            retrieval.get("supporting_chunk_ids")
            or retrieval.get("verified_supporting_chunk_ids")
        )
    )

    # -----------------------------------------
    # 5. Diagnostic check classification
    # -----------------------------------------

    check_details = {}

    retrieval_status = statuses["retrieval"]

    if verified_retrieval_failure:
        check_details["retrieval"] = {
            "result": "confirmed_failure",
            "explanation": (
                "A retrieval failure was reported "
                "against independently verified evidence."
            ),
        }
    elif retrieval_failure:
        check_details["retrieval"] = {
            "result": "potential_failure",
            "explanation": (
                "A retrieval failure was reported, "
                "but evidence verification is not "
                "established in this summary."
            ),
        }
    elif retrieval_status == "retrieval_success":
        check_details["retrieval"] = {
            "result": "passed",
            "explanation": (
                "Verified supporting evidence was "
                "retrieved successfully."
            ),
        }
    else:
        check_details["retrieval"] = {
            "result": "unknown",
            "explanation": (
                "Retrieval cannot be evaluated without "
                "verified supporting evidence."
            ),
        }

    chunking_status = statuses["chunking"]

    if (
        chunking_warning_count > 0
        or chunking_status == "potential_issues_detected"
    ):
        check_details["chunking"] = {
            "result": "warning",
            "explanation": (
                f"{chunking_warning_count} potential "
                "chunking issues were reported. "
                "These are heuristic warnings."
            ),
        }
    elif chunking_status in {
        "no_issues_detected",
        "no_issues",
        "healthy",
    }:
        check_details["chunking"] = {
            "result": "passed",
            "explanation": (
                "No chunking issues were detected "
                "by the configured checks."
            ),
        }
    else:
        check_details["chunking"] = {
            "result": "unknown",
            "explanation": (
                "Chunking analysis did not produce "
                "a conclusive status."
            ),
        }

    conflict_status = statuses["conflict"]

    if (
        conflict_warning_count > 0
        or conflict_status in {
            "conflict_detected",
            "conflicts_detected",
            "potential_conflicts_detected",
        }
    ):
        check_details["conflict"] = {
            "result": "potential_failure",
            "explanation": (
                f"{conflict_warning_count} potential "
                "evidence conflicts were reported."
            ),
        }
    elif conflict_status in {
        "no_conflict_detected",
        "no_conflicts_detected",
    }:
        check_details["conflict"] = {
            "result": "passed",
            "explanation": (
                "No conflicts were detected "
                "in the analyzed evidence."
            ),
        }
    else:
        check_details["conflict"] = {
            "result": "inconclusive",
            "explanation": (
                "The conflict analysis could not "
                "reach a conclusive result."
            ),
        }

    reasoning_status = statuses["reasoning"]

    if (
        reasoning_warning_count > 0
        or reasoning_status in {
            "reasoning_error",
            "issues_detected",
            "reasoning_issues_detected",
            "potential_reasoning_error",
            "unsupported_claim",
            "contradicted_claim",
            "incorrect_interpretation",
            "missing_important_evidence",
        }
    ):
        check_details["reasoning"] = {
            "result": "potential_failure",
            "explanation": (
                f"{reasoning_warning_count} reasoning "
                "findings require review."
            ),
        }
    elif reasoning_status == "supported":
        check_details["reasoning"] = {
            "result": "passed",
            "explanation": (
                "The LLM audit found the generated "
                "answer supported by the supplied "
                "evidence. This is not independent "
                "proof of correctness."
            ),
        }
    else:
        check_details["reasoning"] = {
            "result": "inconclusive",
            "explanation": (
                "Reasoning analysis did not "
                "establish whether the answer "
                "was supported."
            ),
        }

    # -----------------------------------------
    # 6. Counts and coverage
    # -----------------------------------------

    result_counts = Counter(
        item["result"]
        for item in check_details.values()
    )

    total_checks = len(check_details)

    passed_checks = result_counts["passed"]

    unresolved_checks = [
        name
        for name, detail in check_details.items()
        if detail["result"] in {
            "unknown",
            "inconclusive",
        }
    ]

    flagged_checks = [
        name
        for name, detail in check_details.items()
        if detail["result"] in {
            "warning",
            "potential_failure",
            "confirmed_failure",
        }
    ]

    warning_count = (
        chunking_warning_count
        + conflict_warning_count
        + reasoning_warning_count
    )

    if retrieval_failure and not verified_retrieval_failure:
        warning_count += 1

    # This is coverage of reported check outcomes,
    # not statistical confidence or model accuracy.
    resolved_checks = (
        total_checks - len(unresolved_checks)
    )

    coverage_percent = round(
        100.0 * resolved_checks / total_checks,
        2,
    ) if total_checks else 0.0

    # -----------------------------------------
    # 7. Diagnostic confidence classification
    # -----------------------------------------

    if result_counts["confirmed_failure"] > 0:
        confidence_level = "verified_failure_evidence"

    elif unresolved_checks:
        confidence_level = "limited"

    elif (
        result_counts["potential_failure"] > 0
        or result_counts["warning"] > 0
    ):
        confidence_level = "qualified"

    elif passed_checks == total_checks:
        confidence_level = "all_checks_reported_pass"

    else:
        confidence_level = "limited"

    explanations = {
        "verified_failure_evidence": (
            "At least one check reports a failure "
            "against independently verified evidence."
        ),
        "limited": (
            "Some diagnostic checks are unknown or "
            "inconclusive. The overall assessment "
            "has incomplete evidence coverage."
        ),
        "qualified": (
            "All checks returned an assessment, "
            "but warnings or potential failures "
            "require further investigation."
        ),
        "all_checks_reported_pass": (
            "All configured checks reported passing. "
            "This does not guarantee that the "
            "generated answer is correct."
        ),
    }

    return {
        "warning_summary": {
            "total_warnings": warning_count,
            "chunking_warnings": chunking_warning_count,
            "conflict_findings": conflict_warning_count,
            "reasoning_findings": reasoning_warning_count,
            "chunking_issue_counts": chunking_issue_counts,
            "reasoning_issue_counts": dict(
                reasoning_issue_types
            ),
        },
        "diagnostic_confidence": {
            "level": confidence_level,
            "explanation": explanations[confidence_level],
            "check_coverage_percent": coverage_percent,
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "unresolved_checks": unresolved_checks,
            "flagged_checks": flagged_checks,
            "result_counts": dict(result_counts),
            "note": (
                "Confidence is a qualitative summary "
                "of diagnostic coverage, not a "
                "calibrated probability of correctness."
            ),
        },
        "check_details": check_details,
    }
