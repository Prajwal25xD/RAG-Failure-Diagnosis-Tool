
"""
Dynamic, schema-aware query planner for the RAG Failure Diagnosis Tool.

Features:
- Uses LM Studio through the shared client.
- Supports SQL, RAG, and Hybrid query routing.
- Dynamically understands available PostgreSQL columns.
- Extracts tender IDs and metadata filters.
- Validates LLM-generated JSON.
- Never generates or executes SQL.
- No hardcoded tender IDs or questions.
"""

import json
import re

from agent.lm_studio_client import chat_completion


ALLOWED_ROUTES = {"sql", "rag", "hybrid"}

ALLOWED_OPS = {"lookup", "search", "count", "list"}

ALLOWED_FILTER_OPS = {"eq", "contains", "gte", "lte"}

MAX_TENDER_IDS = 30
MAX_FILTERS = 8


def extract_json(raw):
    """
    Extract a JSON object from the LM Studio response.
    Supports plain JSON and Markdown code fences.
    """

    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(
            "LM Studio returned an empty query plan."
        )

    raw = raw.strip()

    # Remove Markdown code fences.
    raw = re.sub(
        r"^```(?:json)?\s*",
        "",
        raw,
        flags=re.IGNORECASE,
    )

    raw = re.sub(r"\s*```$", "", raw)

    try:
        return json.loads(raw)

    except json.JSONDecodeError:
        decoder = json.JSONDecoder()

        # Find the first valid JSON object.
        for match in re.finditer(r"\{", raw):
            try:
                result, _ = decoder.raw_decode(
                    raw[match.start():]
                )

                if isinstance(result, dict):
                    return result

            except json.JSONDecodeError:
                continue

    raise ValueError(
        "Query planner returned invalid JSON.\n"
        f"Response: {raw[:1500]}"
    )


def validate_tender_ids(tender_ids):
    """
    Validate and deduplicate tender IDs.
    """

    if tender_ids is None:
        return []

    if not isinstance(tender_ids, list):
        raise ValueError(
            "tender_ids must be a list."
        )

    if len(tender_ids) > MAX_TENDER_IDS:
        raise ValueError(
            f"Maximum {MAX_TENDER_IDS} tender IDs allowed."
        )

    validated_ids = []

    for value in tender_ids:

        if isinstance(value, bool):
            raise ValueError(
                f"Invalid tender ID: {value}"
            )

        try:
            tender_id = int(value)

        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid tender ID: {value}"
            )

        if tender_id < 0:
            raise ValueError(
                "Tender ID cannot be negative."
            )

        if tender_id not in validated_ids:
            validated_ids.append(tender_id)

    return validated_ids


def validate_filters(filters, columns):
    """
    Validate metadata filters against actual
    PostgreSQL column names.
    """

    if filters is None:
        return []

    if not isinstance(filters, list):
        raise ValueError(
            "filters must be a list."
        )

    if len(filters) > MAX_FILTERS:
        raise ValueError(
            f"Maximum {MAX_FILTERS} filters allowed."
        )

    allowed_columns = set(columns)

    validated_filters = []

    for item in filters:

        if not isinstance(item, dict):
            raise ValueError(
                "Each filter must be a JSON object."
            )

        column = item.get("column")
        operator = item.get("operator")
        value = item.get("value")

        if column not in allowed_columns:
            raise ValueError(
                f"Unknown metadata column: {column}"
            )

        if operator not in ALLOWED_FILTER_OPS:
            raise ValueError(
                f"Unsupported filter operator: {operator}"
            )

        if isinstance(value, bool) or not isinstance(
            value,
            (str, int, float),
        ):
            raise ValueError(
                f"Invalid filter value for {column}"
            )

        validated_filters.append(
            {
                "column": column,
                "operator": operator,
                "value": value,
            }
        )

    return validated_filters


def build_planner_prompt(question, columns):
    """
    Build a dynamic planning prompt based on
    the actual PostgreSQL metadata schema.
    """

    columns_json = json.dumps(
        columns,
        ensure_ascii=False,
    )

    prompt = f"""
You are an intelligent query planner for a PostgreSQL
and document-based RAG system.

Your task is to understand the user's question
and decide how the system should retrieve information.

The system contains:

1. PostgreSQL tender metadata.
2. Indexed PDF document chunks.
3. Vector search for semantic document retrieval.

AVAILABLE METADATA COLUMNS:

{columns_json}

USER QUESTION:

{question}

QUERY ROUTING RULES:

SQL:
Use when the question asks for:
- Tender details stored in metadata.
- Tender titles.
- Bid dates.
- Tender counts.
- Metadata filtering.
- Listing records.

RAG:
Use when the question asks for:
- Eligibility requirements in PDF documents.
- Contract clauses.
- Technical specifications.
- PDF document contents.
- Semantic document search.

HYBRID:
Use when the question needs both:
- Structured metadata.
- Information from PDF chunks.

OPERATIONS:

lookup:
Retrieve information about specific tender IDs.

search:
Search metadata or document content.

count:
Count matching tender records.

list:
Return matching tender records.

FILTER OPERATORS:

eq:
Exact equality.

contains:
Text contains.

gte:
Greater than or equal.

lte:
Less than or equal.

IMPORTANT RULES:

1. Never generate SQL queries.
2. Never invent PostgreSQL columns.
3. Never invent tender IDs.
4. Use only available metadata columns.
5. Extract explicitly mentioned tender IDs.
6. If no tender ID is mentioned, use an empty array.
7. For PDF questions, preserve the original question
   in document_query.
8. For metadata questions, use SQL.
9. For PDF questions, use RAG.
10. For combined questions, use Hybrid.
11. For dates stored as TEXT, prefer contains unless
    the date format is established.
12. Return ONLY a valid JSON object.
13. Do not include explanations outside JSON.

REQUIRED JSON FORMAT:

{{
    "route": "sql",
    "operation": "lookup",
    "tender_ids": [],
    "filters": [],
    "search_text": "",
    "document_query": "",
    "explanation": ""
}}

EXAMPLE 1:

Question:
Give me the bid end date of tender 12247

Response:
{{
    "route": "sql",
    "operation": "lookup",
    "tender_ids": [12247],
    "filters": [],
    "search_text": "",
    "document_query": "",
    "explanation": "Retrieve the tender bid end date from metadata."
}}

EXAMPLE 2:

Question:
What are the eligibility requirements in tender 12247?

Response:
{{
    "route": "rag",
    "operation": "search",
    "tender_ids": [12247],
    "filters": [],
    "search_text": "",
    "document_query": "What are the eligibility requirements in tender 12247?",
    "explanation": "Retrieve eligibility evidence from PDF chunks."
}}

EXAMPLE 3:

Question:
How many tenders are stored in the database?

Response:
{{
    "route": "sql",
    "operation": "count",
    "tender_ids": [],
    "filters": [],
    "search_text": "",
    "document_query": "",
    "explanation": "Count the tender metadata records."
}}

EXAMPLE 4:

Question:
Find IT maintenance tenders and summarize their eligibility requirements.

Response:
{{
    "route": "hybrid",
    "operation": "search",
    "tender_ids": [],
    "filters": [],
    "search_text": "IT maintenance",
    "document_query": "Eligibility requirements for IT maintenance tenders",
    "explanation": "Find matching tender metadata and retrieve PDF eligibility evidence."
}}

Return the JSON query plan now.
"""

    return prompt.strip()


def validate_plan(plan, columns, question):
    """
    Validate the complete query plan.
    """

    if not isinstance(plan, dict):
        raise ValueError(
            "Planner output must be a JSON object."
        )

    route = plan.get("route", "hybrid")
    operation = plan.get("operation", "search")

    if route not in ALLOWED_ROUTES:
        raise ValueError(
            f"Unsupported route: {route}"
        )

    if operation not in ALLOWED_OPS:
        raise ValueError(
            f"Unsupported operation: {operation}"
        )

    tender_ids = validate_tender_ids(
        plan.get("tender_ids", [])
    )

    filters = validate_filters(
        plan.get("filters", []),
        columns,
    )

    search_text = str(
        plan.get("search_text") or ""
    )[:500]

    document_query = str(
        plan.get("document_query") or question
    )[:2000]

    explanation = str(
        plan.get("explanation") or ""
    )[:500]

    return {
        "route": route,
        "operation": operation,
        "tender_ids": tender_ids,
        "filters": filters,
        "search_text": search_text,
        "document_query": document_query,
        "explanation": explanation,
    }


def plan_question(question, columns, model=None):
    """
    Main query planning function.

    Parameters
    ----------
    question : str
        Natural-language question.

    columns : list
        Actual PostgreSQL metadata columns.

    model : str, optional
        LM Studio model override.

    Returns
    -------
    dict
        Validated query plan.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError(
            "Question cannot be empty."
        )

    columns = list(columns)

    if not columns:
        raise ValueError(
            "No PostgreSQL metadata columns provided."
        )

    prompt = build_planner_prompt(
        question=question,
        columns=columns,
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a structured query planner. "
                "Return exactly one valid JSON object. "
                "Never generate SQL. "
                "Do not return Markdown."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    # Shared LM Studio client.
    # The client must accept a model override if supplied.
    if model is None:
        raw_response = chat_completion(
            messages=messages,
            max_tokens=2048,
            temperature=0,
        )
    else:
        raw_response = chat_completion(
            messages=messages,
            max_tokens=2048,
            temperature=0,
            model=model,
        )

    plan = extract_json(raw_response)

    validated_plan = validate_plan(
        plan=plan,
        columns=columns,
        question=question,
    )

    return validated_plan
