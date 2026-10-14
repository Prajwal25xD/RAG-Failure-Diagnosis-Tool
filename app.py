
"""
RAG Failure Diagnosis Tool
Streamlit dashboard for SQL, RAG and Hybrid queries.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import contextlib
import io
import json
import traceback
from datetime import datetime
from pathlib import Path

import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RAG Failure Diagnosis Tool",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SESSION STATE
# ============================================================

if "rag_last_report" not in st.session_state:
    st.session_state["rag_last_report"] = None

if "rag_last_logs" not in st.session_state:
    st.session_state["rag_last_logs"] = ""

if "rag_last_error" not in st.session_state:
    st.session_state["rag_last_error"] = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def as_dict(value):
    """Return a dictionary or an empty dictionary."""
    return value if isinstance(value, dict) else {}


def status_label(value):
    """Convert status identifiers to readable labels."""
    if value is None:
        return "Unknown"

    return str(value).replace("_", " ").strip().title()


def json_bytes(data):
    """Serialize reports safely, including dates and decimals."""
    return json.dumps(
        data,
        indent=2,
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def parse_verified_ids(raw):
    """
    Parse optional verified chunk IDs.

    Examples:
        123,456,789
        123 456 789
    """
    if not raw or not raw.strip():
        return None

    tokens = raw.replace(",", " ").split()

    ids = []
    for token in tokens:
        try:
            chunk_id = int(token)
        except ValueError as exc:
            raise ValueError(
                f"Invalid verified chunk ID: {token}"
            ) from exc

        if chunk_id < 0:
            raise ValueError(
                "Verified chunk IDs must be nonnegative."
            )

        if chunk_id not in ids:
            ids.append(chunk_id)

    return ids or None


def format_value(value):
    """Display structured metadata values."""
    if value is None:
        return "—"

    if isinstance(value, (dict, list)):
        return json.dumps(
            value,
            ensure_ascii=False,
            default=str,
        )

    return str(value)


def render_diagnostics(report):
    """Display all available diagnostic reports."""

    diagnostics = as_dict(
        report.get("diagnostics")
    )

    summary = as_dict(
        report.get("diagnostic_summary")
    )

    st.subheader("Diagnostic summary")

    if summary:
        st.json(summary, expanded=False)
    else:
        st.info(
            "No diagnostic summary was returned. "
            "SQL-only queries may not require RAG diagnostics."
        )

    if not diagnostics:
        st.info(
            "No RAG diagnostic checks were returned "
            "for this query."
        )
        return

    st.subheader("Individual diagnostic checks")

    preferred_order = [
        "retrieval",
        "chunking",
        "conflict",
        "reasoning",
    ]

    keys = list(diagnostics.keys())

    ordered_keys = [
        key for key in preferred_order
        if key in diagnostics
    ]

    ordered_keys.extend(
        key for key in keys
        if key not in ordered_keys
    )

    for name in ordered_keys:
        with st.expander(
            f"{status_label(name)} diagnosis",
            expanded=False,
        ):
            value = diagnostics[name]

            if isinstance(value, (dict, list)):
                st.json(value, expanded=True)
            else:
                st.write(value)


def render_structured_records(report):
    """Display metadata returned by PostgreSQL."""

    records = report.get("structured_records")

    if records is None or records == []:
        st.info("No structured PostgreSQL records returned.")
        return

    st.subheader("Structured PostgreSQL records")

    if isinstance(records, list):
        if records and isinstance(records[0], dict):
            st.dataframe(
                records,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.json(records, expanded=False)

    elif isinstance(records, dict):
        st.json(records, expanded=True)

    else:
        st.write(records)


def render_evidence(report):
    """Display retrieved PDF chunks and their source fields."""

    chunks = report.get("retrieved_chunks") or []

    if not chunks:
        st.info("No PDF evidence chunks were returned.")
        return

    st.subheader("Retrieved document evidence")

    st.caption(
        f"{len(chunks)} chunks included in the report."
    )

    for index, chunk in enumerate(chunks, start=1):

        if not isinstance(chunk, dict):
            with st.expander(f"Chunk {index}"):
                st.write(chunk)
            continue

        chunk_id = (
            chunk.get("id")
            or chunk.get("chunk_id")
            or chunk.get("chunk_uid")
            or index
        )

        tender_id = chunk.get("tender_id", "Unknown")
        document_type = chunk.get(
            "document_type",
            "Unknown",
        )

        title = (
            f"Chunk {chunk_id} | "
            f"Tender {tender_id} | "
            f"{document_type}"
        )

        with st.expander(title, expanded=False):

            col1, col2 = st.columns(2)

            col1.write(
                f"**Tender ID:** {tender_id}"
            )

            col2.write(
                f"**Document type:** {document_type}"
            )

            if chunk.get("document_url"):
                st.write("**Document URL:**")
                st.code(
                    str(chunk["document_url"]),
                    language="text",
                )

            if chunk.get("chunk_index") is not None:
                st.write(
                    "**Chunk index:**",
                    chunk["chunk_index"],
                )

            if chunk.get("score") is not None:
                st.write(
                    "**Retrieval score:**",
                    chunk["score"],
                )

            chunk_text = (
                chunk.get("chunk_text")
                or chunk.get("text")
                or chunk.get("content")
            )

            if chunk_text:
                st.write("**Original chunk text:**")
                st.text_area(
                    "Chunk content",
                    value=str(chunk_text),
                    height=230,
                    disabled=True,
                    key=f"chunk_text_{index}",
                    label_visibility="collapsed",
                )

            with st.expander("Complete chunk metadata"):
                st.json(chunk, expanded=False)


def render_answer(report):
    """Display generated answer or structured results."""

    answer = as_dict(
        report.get("answer")
    )

    st.subheader("Evidence-grounded answer")

    if answer.get("error"):
        st.error(str(answer["error"]))

    answer_text = (
        answer.get("text")
        or answer.get("answer")
        or answer.get("response")
    )

    if answer_text:
        st.markdown(str(answer_text))
        return

    # If the LLM returned no text, show available SQL records.
    records = report.get("structured_records")

    if records:
        st.warning(
            "No generated answer text was returned. "
            "Showing the retrieved PostgreSQL records instead."
        )

        render_structured_records(report)

    else:
        st.warning(
            "The pipeline completed but returned no answer text."
        )

        with st.expander("Inspect raw answer"):
            st.json(answer)


def render_report(report):
    """Display report downloads, query plan and logs."""

    st.subheader("Complete execution report")

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    st.download_button(
        label="Download complete JSON report",
        data=json_bytes(report),
        file_name=f"rag_diagnosis_{timestamp}.json",
        mime="application/json",
        use_container_width=True,
    )

    if report.get("report_path"):
        st.success(
            f"Report saved locally: {report['report_path']}"
        )

    if report.get("report_save_error"):
        st.warning(
            f"Report save error: {report['report_save_error']}"
        )

    if report.get("query_plan"):
        with st.expander(
            "AI query plan",
            expanded=True,
        ):
            st.json(
                report["query_plan"],
                expanded=True,
            )

    with st.expander("Full report JSON"):
        st.json(report, expanded=False)

    with st.expander("Execution logs"):
        st.code(
            st.session_state.get(
                "rag_last_logs",
                "",
            ) or "No logs captured.",
            language="text",
        )


def render_dashboard(report):
    """Render a complete pipeline report."""

    metadata = as_dict(
        report.get("metadata")
    )

    overall_status = report.get(
        "overall_status",
        "unknown",
    )

    query_plan = as_dict(
        report.get("query_plan")
    )

    st.divider()

    st.subheader("Latest run")

    metrics = st.columns(4)

    metrics[0].metric(
        "Overall status",
        status_label(overall_status),
    )

    metrics[1].metric(
        "Candidates searched",
        metadata.get("candidate_count", "—"),
    )

    metrics[2].metric(
        "Context chunks",
        metadata.get("context_count", "—"),
    )

    metrics[3].metric(
        "Mode",
        status_label(metadata.get("mode")),
    )

    st.caption(
        f"Question: {metadata.get('question', 'Unknown')} "
        f"· Model: {metadata.get('model', 'LM Studio')}"
    )

    if query_plan:
        route = query_plan.get(
            "route",
            "unknown",
        )

        operation = query_plan.get(
            "operation",
            "unknown",
        )

        st.write(
            f"**Query route:** {route.upper()} "
            f" | **Operation:** {operation}"
        )

    (
        tab_answer,
        tab_diagnostics,
        tab_evidence,
        tab_report,
    ) = st.tabs(
        [
            "💬 Answer",
            "🩺 Diagnostics",
            "📚 Evidence",
            "📄 Report & logs",
        ]
    )

    with tab_answer:
        render_answer(report)

    with tab_diagnostics:
        render_diagnostics(report)

    with tab_evidence:
        records = report.get("structured_records")

        if records:
            render_structured_records(report)

        if report.get("retrieved_chunks"):
            render_evidence(report)

        if not records and not report.get("retrieved_chunks"):
            st.info(
                "No matching PostgreSQL records "
                "or PDF evidence chunks were returned."
            )

    with tab_report:
        render_report(report)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Run configuration")

    st.caption(
        "Uses your existing .env, PostgreSQL database, "
        "embedding model and LM Studio server."
    )

    mode = st.radio(
        "Retrieval mode",
        options=["fast", "exhaustive"],
        format_func=lambda value: (
            "⚡ Fast — limited candidates"
            if value == "fast"
            else "🔬 Exhaustive — all eligible candidates"
        ),
    )

    document_id = st.text_input(
        "Document ID (optional)",
        value="",
        placeholder="Blank = all documents",
    )

    with st.expander("Advanced evaluation"):

        verified_ids_raw = st.text_input(
            "Verified supporting chunk IDs",
            value="",
            placeholder="Example: 5679, 5680",
            help=(
                "Optional known supporting chunk IDs "
                "for evaluating retrieval coverage."
            ),
        )

    st.divider()

    st.subheader("Execution notes")

    st.caption(
        "Exhaustive mode may take longer and load "
        "many candidate records into memory. "
        "The answer and LLM audits still use "
        "the top context chunks."
    )

    st.caption(
        "Diagnostic confidence is qualitative coverage, "
        "not the probability that an answer is correct."
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.caption(
    "EVIDENCE INTELLIGENCE / DIAGNOSTIC CONSOLE"
)

st.title("RAG Failure Diagnosis Tool")

st.write(
    "Investigate retrieval, chunking, conflicting evidence, "
    "and reasoning in one workspace."
)


# ============================================================
# QUESTION FORM
# ============================================================

with st.form("rag_question_form"):

    question = st.text_area(
        "Ask a question about your indexed documents",
        placeholder=(
            "Example: Give me the bid end date "
            "of tender 12247."
        ),
        height=110,
    )

    submitted = st.form_submit_button(
        "Run RAG diagnosis",
        use_container_width=True,
        type="primary",
    )


# ============================================================
# PIPELINE EXECUTION
# ============================================================

if submitted:

    if not question.strip():

        st.error(
            "Please enter a question."
        )

    else:

        # Clear the previous result only when
        # a valid new question is submitted.
        st.session_state["rag_last_report"] = None
        st.session_state["rag_last_logs"] = ""
        st.session_state["rag_last_error"] = None

        with st.spinner(
            "Running retrieval, answer generation "
            "and diagnostic checks..."
        ):

            buffer = io.StringIO()

            try:

                from agent.orchestrator import run_dynamic_query

                with contextlib.redirect_stdout(buffer):

                    result = run_dynamic_query(
                        question=question.strip(),
                        document_id=(
                            document_id.strip() or None
                        ),
                        mode=mode,
                        supporting_chunk_ids=(
                            parse_verified_ids(
                                verified_ids_raw
                            )
                        ),
                    )

                # Validate the result.
                if result is None:
                    raise RuntimeError(
                        "Pipeline returned None."
                    )

                if not isinstance(result, dict):
                    raise TypeError(
                        "Pipeline returned an unexpected "
                        f"type: {type(result).__name__}"
                    )

                if not result:
                    raise RuntimeError(
                        "Pipeline returned an empty dictionary."
                    )

                # Store the result in session state.
                st.session_state["rag_last_report"] = result

                st.session_state["rag_last_logs"] = (
                    buffer.getvalue()
                )

                st.success(
                    "Pipeline completed successfully!"
                )

            except Exception as exc:

                error_details = traceback.format_exc()

                st.session_state["rag_last_report"] = None

                st.session_state["rag_last_error"] = (
                    error_details
                )

                st.session_state["rag_last_logs"] = (
                    buffer.getvalue()
                    + "\n"
                    + error_details
                )

                st.error(
                    "Pipeline failed: "
                    f"{type(exc).__name__}: {exc}"
                )

                with st.expander(
                    "View detailed error",
                    expanded=True,
                ):
                    st.code(
                        error_details,
                        language="text",
                    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

# Read the latest report from session state.
report = st.session_state.get(
    "rag_last_report"
)

if isinstance(report, dict):

    # Render the complete report.
    render_dashboard(report)

else:

    # Do not display a previous answer after failure.
    if not submitted:

        st.info(
            "Enter a question and select "
            "**Run RAG diagnosis** to begin."
        )
