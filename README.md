# 🔎 RAG Failure Diagnosis Tool

**A beginner-friendly, evidence-aware debugging tool for Retrieval-Augmented Generation (RAG) systems.**

This project does more than answer questions from documents. It helps investigate **why a RAG answer may be wrong**: Did the search miss the right document? Was the text split badly? Do different chunks disagree? Did the language model misunderstand the evidence?

The current application uses **PostgreSQL + pgvector**, **Hugging Face embeddings**, a **locally served LLM through LM Studio**, and a **Streamlit dashboard**. It can also answer questions from structured tender metadata using a bounded SQL query builder.

> **Project status:** Working local prototype. Some diagnostic results are heuristic or LLM-assisted, not guaranteed proof of correctness. The application assumes the PostgreSQL tender metadata and document chunks have already been populated; the README does not claim that the current app automatically scrapes and indexes new PDFs.

## 📌 Contents

1. [What problem does this solve?](#-what-problem-does-this-solve)
2. [What can it do?](#-what-can-it-do)
3. [Tech stack in simple words](#-tech-stack-in-simple-words)
4. [System architecture](#-system-architecture)
5. [How a question moves through the system](#-how-a-question-moves-through-the-system)
6. [SQL vs RAG vs Hybrid](#-sql-vs-rag-vs-hybrid)
7. [Fast Mode vs Exhaustive Mode](#-fast-mode-vs-exhaustive-mode)
8. [The four failure checks](#-the-four-failure-checks)
9. [Project folder structure](#-project-folder-structure)
10. [File-by-file explanation and connections](#-file-by-file-explanation-and-connections)
11. [Database structure](#-database-structure)
12. [Installation and setup](#-installation-and-setup)
13. [Example questions](#-example-questions)
14. [Understanding the dashboard](#-understanding-the-dashboard)
15. [Troubleshooting](#-troubleshooting)
16. [Current limitations and future improvements](#-current-limitations-and-future-improvements)

---

## 🎯 What problem does this solve?

Imagine you ask an AI assistant:

> "What are the eligibility requirements for tender 12247?"

A normal RAG chatbot searches PDF text, selects a few paragraphs, and generates an answer. If that answer is incomplete or incorrect, it can be difficult to tell **where the problem happened**.

This project helps separate the possible causes:

| Problem | Simple meaning | Example |
|---|---|---|
| **Retrieval miss** | The search did not find the necessary text | The eligibility clause exists, but it was not retrieved |
| **Chunking problem** | The document was split into unhelpful pieces | A condition begins in one chunk and ends in another |
| **Conflicting evidence** | Retrieved passages appear to disagree | Two passages mention different experience requirements |
| **Reasoning problem** | The LLM's answer does not match the supplied evidence | The PDF says 3 years, but the answer says 5 |

The goal is to make a document-question-answering pipeline **easier to inspect and debug**, not to promise that every error will be detected automatically.

## ✨ What can it do?

- Ask questions about **tender metadata** such as titles, bid dates, and record counts.
- Ask questions about **indexed tender PDF text** such as eligibility, technical specifications, and contract conditions.
- Combine metadata and PDF evidence for **Hybrid** questions.
- Search chunks using both **semantic vector similarity** and **keyword search**.
- Choose between **Fast** and **Exhaustive** candidate retrieval.
- Display source chunks, tender IDs, document types, and other available evidence metadata.
- Run checks for retrieval, chunking, conflicting evidence, and reasoning.
- Display a diagnostic summary and download a JSON report.
- Run locally with a model served by **LM Studio**.

## 🧰 Tech stack in simple words

| Technology | What it does here |
|---|---|
| **Python** | Connects the components and runs the pipeline |
| **Streamlit** | Creates the browser-based dashboard |
| **PostgreSQL** | Stores tender metadata and indexed document chunks |
| **pgvector** | Stores and searches embedding vectors inside PostgreSQL |
| **Hugging Face / Sentence Transformers** | Turns a question into a numerical vector (embedding) |
| **Nomic embedding model** | The configured embedding model; default is `nomic-ai/nomic-embed-text-v1.5` with 768 dimensions |
| **LM Studio** | Hosts the local language model used for planning, answering, and some audits |
| **SQLAlchemy / psycopg2** | Lets Python communicate with PostgreSQL |
| **Reciprocal Rank Fusion (RRF)** | Combines vector-search and keyword-search rankings |
| **JSON** | Stores or exports machine-readable reports |

**In simple words:** PostgreSQL is the library, embeddings help find similar meanings, LM Studio writes or evaluates answers, and Streamlit is the screen you interact with.

## 🏗 System architecture

```mermaid
flowchart TD
    U[User enters question] --> UI[app.py / Streamlit]
    UI --> O[agent/orchestrator.py]
    O --> P[agent/query_planner.py]
    P --> L[LM Studio]
    P --> D{Query route}
    D -->|SQL| S[agent/sql_executor.py]
    S --> DB[(PostgreSQL metadata)]
    D -->|RAG| M[main.py retrieval pipeline]
    D -->|Hybrid| S
    D -->|Hybrid| M
    M --> H[retrieval/hybrid_search.py]
    H --> V[retrieval/vector_search.py]
    H --> K[retrieval/keyword_search.py]
    V --> C[(PostgreSQL + pgvector chunks)]
    K --> C
    V --> E[Hugging Face embeddings]
    S --> A[agent/evidence_merger.py]
    M --> A
    A --> L
    M --> X[Four diagnostic modules]
    X --> Y[diagnostics/diagnostic_summary.py]
    A --> R[Answer and evidence report]
    Y --> R
    R --> UI
```

**Note:** The RAG-only path for a single tender can use `main.py`'s original answer-generation function instead of `agent/evidence_merger.py`. The diagram is a simplified overview of the main connections.

## 🔄 How a question moves through the system

Consider:

> **"Give me the title, bid end date, and eligibility requirements for tender 12247."**

1. **`app.py` receives the question.** The user chooses Fast or Exhaustive Mode and optionally enters a document ID.
2. **`agent/orchestrator.py` starts the workflow.** It coordinates the other modules.
3. **`agent/query_planner.py` asks LM Studio to create a JSON plan.** The plan chooses `sql`, `rag`, or `hybrid`, extracts the tender ID, and identifies allowed metadata filters. It does **not** ask the model to write SQL.
4. **`agent/sql_executor.py` handles metadata** when the plan includes SQL. It inspects the actual table columns and constructs a limited, parameterized, read-only query.
5. **`main.py` and the retrieval modules handle PDF questions.** They look for relevant chunks using vector and keyword search.
6. **The answer generator uses retrieved evidence.** Depending on the route, this is the existing RAG answer generator or `agent/evidence_merger.py`.
7. **Diagnostic modules examine the RAG process** when document evidence is used.
8. **`app.py` displays the answer, diagnostics, evidence, and JSON report.**

For a metadata-only SQL query, the application may legitimately show **0 PDF chunks**. That does not automatically mean retrieval failed; PDF retrieval was not required.

## 🧭 SQL vs RAG vs Hybrid

| Route | When to use it | Example | Main code path |
|---|---|---|---|
| **SQL** | Information already stored in metadata columns | "What is the bid end date of tender 12247?" | `orchestrator.py` → `sql_executor.py` → `evidence_merger.py` |
| **RAG** | Details found inside indexed PDF text | "What eligibility documents are required?" | `orchestrator.py` → `main.py` → `retrieval/` → RAG answer + diagnostics |
| **Hybrid** | A question needs both metadata and PDF passages | "Show the tender title, closing date, and eligibility rules" | `orchestrator.py` → SQL + retrieval → `evidence_merger.py` + diagnostics |

The planner's JSON is a **request plan**, not raw SQL. The executor only accepts supported operations and validated columns. This is safer and more predictable than directly executing SQL generated by an LLM.

## ⚡ Fast Mode vs Exhaustive Mode

| Mode | What happens | Best for |
|---|---|---|
| **Fast** | Retrieves a limited number of candidate chunks using configured vector and keyword limits | Normal questions and faster iteration |
| **Exhaustive** | Considers all eligible indexed candidates for retrieval ranking, which can require substantially more time and memory | Investigating possible retrieval misses |

**Important:** Exhaustive Mode does **not** send every database chunk to the LLM. It can search/rank a much larger candidate set, while answer generation still uses a smaller number of top context chunks. It is also not a guarantee that every possible piece of evidence will be recognized as relevant.

## 🩺 The four failure checks

### 1. Retrieval diagnosis

**Question it asks:** Did the search return the evidence that should have been found?

This is especially useful when you provide **verified supporting chunk IDs** in the dashboard. Without independently known supporting evidence, the tool cannot always establish whether a missing chunk represents a true retrieval failure.

### 2. Chunking diagnosis

**Question it asks:** Were the retrieved text pieces split in a potentially problematic way?

It can flag suspicious chunk characteristics such as unusually short/long text or other configured quality indicators. These are **warnings**, not automatic proof that chunking caused an incorrect answer.

### 3. Conflict diagnosis

**Question it asks:** Do the retrieved passages contradict one another?

For example, two retrieved sections might give different eligibility conditions. The model-assisted check may flag potential contradictions, but the original documents still need human verification.

### 4. Reasoning diagnosis

**Question it asks:** Does the generated answer appear to be supported by the retrieved evidence?

For example, if the answer says "5 years of experience" but the provided evidence says "3 years," the check should try to identify the mismatch. Because an LLM assists with this audit, it can also miss mistakes.

### Diagnostic summary

`diagnostics/diagnostic_summary.py` collects the four checks into a more readable result: warnings, possible or confirmed failures, checks that passed, and checks that remain inconclusive.

**"No Confirmed Failure Detected" does not mean "the answer is guaranteed correct."** Likewise, diagnostic confidence describes the coverage and quality of the checks, **not** a statistical probability that the answer is right.

## 📁 Project folder structure

The following reflects the current core application files. Some local setups may also include older modules, scripts, saved reports, or additional tests.

```text
RAG-Failure-Diagnosis-Tool/
│
├── app.py                         # Streamlit user interface
├── main.py                        # Core document-RAG pipeline
├── config.py                      # Loads settings from .env
├── dynamic_query.py               # Older metadata-query module; not the current UI entry point
├── requirements.txt               # Python packages
├── .gitignore                     # Files Git should not upload
├── .env                           # Local secrets/settings (do NOT commit)
│
├── agent/
│   ├── __init__.py
│   ├── orchestrator.py            # Decides which components to run
│   ├── query_planner.py           # LLM creates a structured query plan
│   ├── sql_executor.py            # Safe metadata queries
│   ├── evidence_merger.py         # Combines metadata and PDF evidence into an answer
│   └── lm_studio_client.py        # Shared LM Studio API helper
│
├── retrieval/
│   ├── __init__.py
│   ├── vector_search.py           # Finds semantically similar chunks
│   ├── keyword_search.py          # Finds matching words/phrases
│   └── hybrid_search.py           # Combines both rankings with RRF
│
├── diagnostics/
│   ├── retrieval_diagnosis.py     # Checks retrieval coverage
│   ├── chunking_diagnosis.py      # Checks chunk quality
│   ├── conflict_diagnosis.py      # Looks for conflicting evidence
│   ├── reasoning_diagnosis.py     # Generates/audits RAG answers
│   └── diagnostic_summary.py     # Combines diagnostic results
│
├── tests/                         # Individual diagnostic/retrieval tests
└── reports/                       # Generated JSON reports (local output)
```

## 🧩 File-by-file explanation and connections

This section is intended for someone opening the project for the **first time**.

### Main application files

| File | What it means in simple words | Connects to |
|---|---|---|
| **`app.py`** | The **front desk**. Displays the question box, settings, answer, diagnostics, evidence, and download button. | Calls `agent/orchestrator.py`; reads the returned report |
| **`main.py`** | The **RAG manager**. Retrieves document evidence, generates document answers, runs diagnostic checks, and can save reports. | Calls `retrieval/hybrid_search.py` and modules in `diagnostics/` |
| **`config.py`** | The **settings book**. Reads `.env` for database, table, embedding, search, and model settings. | Imported by `main.py`, retrieval code, and agent modules |
| **`requirements.txt`** | The **shopping list for Python**. Lists packages needed to run the application. | Used by `pip install -r requirements.txt` |
| **`.env`** | The **private settings file**. Holds your database connection and local model configuration. | Read through `config.py` and some `os.getenv(...)` calls |
| **`.gitignore`** | The **do-not-upload list**. Helps prevent `.env`, `.venv`, caches, and generated outputs from being committed. | Used by Git, not by the RAG runtime |
| **`dynamic_query.py`** | An earlier query-routing implementation retained in the repository. | The current `app.py` uses `agent/orchestrator.py` instead |

### `agent/` — the decision-making and answering layer

| File | Simple explanation | Receives from → Sends to |
|---|---|---|
| **`agent/__init__.py`** | Marks `agent` as a Python package. | Enables imports like `from agent.orchestrator import ...` |
| **`agent/orchestrator.py`** | The **traffic controller**: chooses SQL, RAG, or both, and assembles the result. | `app.py` → `query_planner.py`, `sql_executor.py`, `main.py`, `evidence_merger.py` |
| **`agent/query_planner.py`** | The **question interpreter**: asks LM Studio for a JSON plan, such as `route=sql` and `tender_ids=[12247]`. | `orchestrator.py` → `lm_studio_client.py` |
| **`agent/sql_executor.py`** | The **database reader**: builds restricted, parameterized SQL using validated metadata columns. | `orchestrator.py` → PostgreSQL, using the engine from `retrieval/vector_search.py` |
| **`agent/evidence_merger.py`** | The **answer writer**: supplies structured records and selected PDF chunks to the LLM to produce an evidence-based response. | `orchestrator.py` → LM Studio API |
| **`agent/lm_studio_client.py`** | The **LM Studio messenger**: sends chat-completion requests and reports API errors. | Used by `query_planner.py` in the current implementation |

**Small implementation detail:** `evidence_merger.py` currently makes its own LM Studio HTTP request; it does not yet reuse `lm_studio_client.py`. This is a possible refactoring improvement.

### `retrieval/` — the document search layer

| File | Simple explanation | Connects to |
|---|---|---|
| **`retrieval/__init__.py`** | Makes the retrieval folder importable as a Python package. | Other Python modules importing `retrieval.*` |
| **`retrieval/vector_search.py`** | The **meaning searcher**. Embeds the question and finds chunks whose vectors are similar. Also provides a database engine used elsewhere. | Hugging Face embedding model + PostgreSQL/pgvector |
| **`retrieval/keyword_search.py`** | The **word searcher**. Looks for chunks matching terms in the question. | PostgreSQL indexed text |
| **`retrieval/hybrid_search.py`** | The **ranking combiner**. Uses Reciprocal Rank Fusion to merge the two result lists. | Called by `main.py`; calls both search modules |

**Why use both searches?** Vector search is useful when the wording differs but the meaning is similar. Keyword search helps when exact names, numbers, or technical phrases matter. Hybrid search tries to benefit from both.

### `diagnostics/` — the debugging layer

| File | Simple explanation | Called by |
|---|---|---|
| **`retrieval_diagnosis.py`** | Checks whether expected supporting chunks were retrieved. | `main.py` |
| **`chunking_diagnosis.py`** | Flags potential problems in how text chunks were formed. | `main.py` |
| **`conflict_diagnosis.py`** | Looks for disagreements in retrieved evidence. | `main.py` |
| **`reasoning_diagnosis.py`** | Contains document-answer generation and reasoning/evidence auditing functions. | `main.py` |
| **`diagnostic_summary.py`** | Collects diagnostic outcomes into one readable summary. | `main.py` and `agent/orchestrator.py` |

### `tests/` — checking individual components

| File | What it tests |
|---|---|
| `tests/test_retrieval.py` | Retrieval-related behavior |
| `tests/test_retrieval_diagnosis.py` | Retrieval diagnosis logic |
| `tests/test_chunking_diagnosis.py` | Chunking diagnosis logic |
| `tests/test_conflict_diagnosis.py` | Conflict diagnosis logic |
| `tests/test_reasoning_diagnosis.py` | Reasoning diagnosis logic |

The tests are useful for checking code changes, but passing unit tests does not prove the complete live PostgreSQL + LM Studio workflow is correct.

### One-line connection map

```text
app.py
  └── agent/orchestrator.py
       ├── agent/query_planner.py
       │    └── agent/lm_studio_client.py → LM Studio
       ├── agent/sql_executor.py → PostgreSQL metadata
       ├── main.py
       │    ├── retrieval/hybrid_search.py
       │    │    ├── retrieval/vector_search.py → Hugging Face + pgvector
       │    │    └── retrieval/keyword_search.py → PostgreSQL
       │    ├── diagnostics/retrieval_diagnosis.py
       │    ├── diagnostics/chunking_diagnosis.py
       │    ├── diagnostics/conflict_diagnosis.py
       │    └── diagnostics/reasoning_diagnosis.py
       ├── agent/evidence_merger.py → LM Studio
       └── diagnostics/diagnostic_summary.py
            ↓
       Report returned to app.py → Answer / Diagnostics / Evidence / Logs
```

## 🗃 Database structure

This application reads **two kinds of stored information**:

### 1. Tender metadata: `public.gujrat_tender`

One row represents a tender. Example fields include:

| Column | Meaning |
|---|---|
| `ten_id` | Tender identifier |
| `title` | Tender title |
| `bid_end_date_time` | Deadline for submitting bids |
| `bid_opening_date_time` | Scheduled opening date/time |
| `notes` | Additional notes, if available |
| `eligibility_note` | Eligibility information stored as metadata |
| `documents`, `data_json` | JSON-formatted source information |

### 2. Indexed document chunks: `public.tender_chunks`

One row represents a **piece of text** extracted from a tender document. Important fields include:

| Column | Meaning |
|---|---|
| `id` | Unique chunk identifier |
| `tender_id` | Tender associated with the chunk |
| `document_type` | Type of source document |
| `document_url` | Source document link, if stored |
| `chunk_index` | Chunk's position in its source |
| `chunk_text` | Actual extracted text |
| `embedding_hf` (or configured column) | Numerical vector used for semantic search |

**Why split documents into chunks?** A PDF may contain many pages. Searching smaller passages is usually more manageable than sending an entire PDF to the LLM. However, splitting at the wrong location can hide important context, which is one reason the chunking diagnosis exists.

> The current application **queries pre-existing records and embeddings**. To use a new document, an upstream ingestion process must first extract its text, split it into chunks, generate compatible embeddings, and store them in the configured tables.

## 🚀 Installation and setup

### Prerequisites

- Python (the project has been developed with **Python 3.12**).
- PostgreSQL with the **pgvector** extension installed.
- Existing metadata and indexed document chunks in the configured tables.
- LM Studio installed, with a compatible chat model loaded and its local API server running.
- Enough local memory to load the embedding and language models.

### 1. Clone the repository

```bash
git clone https://github.com/Prajwal25xD/RAG-Failure-Diagnosis-Tool.git
cd RAG-Failure-Diagnosis-Tool
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**Git Bash on Windows:**

```bash
source .venv/Scripts/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure `.env`

Create `.env` in the project root. Replace placeholders with your **own local values**:

```dotenv
# PostgreSQL connection
DATABASE_URL=postgresql+psycopg2://YOUR_USER:YOUR_PASSWORD@localhost:5432/tender
DB_SCHEMA=public
CHUNKS_TABLE=tender_chunks
DOCUMENT_ID_COLUMN=tender_id
CHUNK_TEXT_COLUMN=chunk_text
EMBEDDING_COLUMN=embedding_hf

# Metadata table used for SQL queries
RAG_TENDER_METADATA_SCHEMA=public
RAG_TENDER_METADATA_TABLE=gujrat_tender
RAG_TENDER_ID_COLUMN=ten_id
RAG_SQL_MAX_ROWS=100
RAG_SQL_TIMEOUT_MS=10000

# Embeddings: must match stored vectors
EMBEDDING_MODEL_NAME=nomic-ai/nomic-embed-text-v1.5
EMBEDDING_DIMENSIONS=768
EMBEDDING_QUERY_PREFIX=search_query:

# LM Studio
LM_STUDIO_BASE_URL=http://localhost:1234/v1
LM_STUDIO_MODEL=YOUR_LOADED_MODEL_ID
LM_STUDIO_TIMEOUT=600

# Search / answer settings
VECTOR_CANDIDATE_LIMIT=30
KEYWORD_CANDIDATE_LIMIT=30
FINAL_TOP_K=8
RAG_ANSWER_CONTEXT_K=8
RAG_PLANNER_TIMEOUT=120
RAG_PLANNER_MAX_TOKENS=2048

# Reports
RAG_SAVE_REPORTS=true
RAG_REPORT_DIR=reports
```

**Important:** `LM_STUDIO_MODEL` must match the ID shown by your running LM Studio server. `EMBEDDING_COLUMN` and `EMBEDDING_DIMENSIONS` must match the actual PostgreSQL column and stored vectors. Do not commit `.env` to GitHub.

### 5. Start LM Studio

1. Load your chosen compatible chat model in LM Studio.
2. Start the local server (commonly `http://localhost:1234`).
3. Check the model endpoint:

```powershell
Invoke-RestMethod http://localhost:1234/v1/models
```

### 6. Start the dashboard

```bash
streamlit run app.py
```

Open the local Streamlit URL shown in the terminal (usually `http://localhost:8501`).

### 7. Optional: Run the original RAG pipeline directly

`main.py` also contains a command-line entry point. Inspect its `--help` options before running:

```bash
python main.py --help
```

### 8. Optional: Run tests

```bash
python -m pytest tests/
```

If `pytest` is not installed, install it separately with `pip install pytest`.

## 💬 Example questions

Try the following in the dashboard **after verifying that the tender IDs exist**.

| Type | Example question | Expected route |
|---|---|---|
| Metadata | Give me the bid end date of tender 12247. | SQL |
| Metadata | How many tenders are stored in the database? | SQL |
| PDF evidence | What eligibility requirements are mentioned in the PDF for tender 12247? | RAG |
| PDF evidence | What technical specifications are required for tender 12247? | RAG |
| Combined | Give me the title, bid end date, and eligibility requirements for tender 12247. | Hybrid |
| Comparison | Compare eligibility requirements for two existing tender IDs. | RAG/Hybrid, depending on plan |

The query planner determines the actual route, so these are **expected behaviors**, not guaranteed outputs.

## 🖥 Understanding the dashboard

| Section | What it shows |
|---|---|
| **Run configuration** | Fast/Exhaustive mode, optional document ID, optional verified supporting chunk IDs |
| **Latest run** | Overall status, candidate count, context chunk count, mode, and query route |
| **💬 Answer** | LLM-generated answer or fallback structured records |
| **🩺 Diagnostics** | Retrieval, chunking, conflict, and reasoning findings when available |
| **📚 Evidence** | Structured database rows and/or retrieved PDF chunks |
| **📄 Report & logs** | JSON report, AI query plan, and execution logs |

### What do "Candidates searched" and "Context chunks" mean?

- **Candidates searched:** Chunks considered by the retrieval/ranking pipeline.
- **Context chunks:** The smaller set selected to supply evidence to the answer generator.

For example, **17 candidates and 8 context chunks** means the search considered 17 chunks but passed 8 selected chunks into the answer context. The numbers change by query and mode.

### What does "Structured Query Complete" mean?

A metadata-oriented request completed through the structured query path. It does **not** mean the system searched PDFs or ran all four RAG diagnostics.

## 🛠 Troubleshooting

| Symptom | Likely area to check | First action |
|---|---|---|
| `HTTP 400` from `/chat/completions` | LM Studio request, model ID, context size, or API parameters | Inspect the **response body**; verify `/v1/models` |
| PostgreSQL connection error | `.env`, server status, credentials, database | Check `DATABASE_URL` and whether PostgreSQL is running |
| Metadata table not found | Metadata schema/table settings | Check `RAG_TENDER_METADATA_SCHEMA` and `RAG_TENDER_METADATA_TABLE` |
| Vector dimension mismatch | Embedding model or database column | Confirm stored vector dimensions and `EMBEDDING_DIMENSIONS` |
| Zero PDF chunks for SQL question | May be expected | Check the `query_plan.route` |
| Zero chunks for PDF question | Retrieval/filter/index issue | Verify the tender has indexed chunks and the ID matches |
| Streamlit shows previous answer after failure | Session-state clearing | Check `rag_last_report` is reset only when a new valid run starts |
| Pipeline succeeds but answer not displayed | Dashboard rendering or report structure | Inspect `report['answer']` and `rag_last_report` |
| Slow Exhaustive Mode | Large candidate set | Test Fast Mode first and monitor memory |

### Why does an answer sometimes look correct but still need checking?

LLMs can produce convincing explanations even when the evidence is incomplete. Always inspect the **Evidence** tab and confirm that cited chunk IDs and metadata fields are actually present in the retrieved records.

## ⚠️ Current limitations and future improvements

### Current limitations

- The current repository focuses on **querying and diagnosing already indexed documents**, not a fully self-contained new-PDF ingestion workflow.
- The planner can misunderstand a question or return invalid JSON.
- The metadata executor supports a bounded set of operations; it is **not** a general-purpose SQL analytics engine.
- PDF retrieval quality depends on extraction quality, chunking, stored embeddings, and query wording.
- Some metadata date columns may be stored as text, limiting safe date comparisons.
- Hybrid searches should be reviewed carefully when metadata filters need to restrict which document chunks are searched.
- Diagnostics can be inconclusive and cannot guarantee factual correctness.
- Exhaustive retrieval can consume significant time and memory.
- The shared LM Studio client is not yet used by every model-calling module.

### Possible future improvements

- Add a complete, configurable ingestion pipeline for new PDFs and OCR.
- Constrain Hybrid document retrieval to the tender IDs matched by SQL metadata filters.
- Validate every answer citation against actual retrieved chunk IDs.
- Add automatic evaluation datasets with known correct supporting chunks.
- Add richer dashboards for failure trends and retrieval comparisons.
- Centralize all LM Studio requests, timeouts, and error reporting.
- Normalize text-based dates in the metadata schema.
- Add Docker-based deployment and CI testing.

---

## 👨‍💻 Repository

**GitHub:** [Prajwal25xD/RAG-Failure-Diagnosis-Tool](https://github.com/Prajwal25xD/RAG-Failure-Diagnosis-Tool)

### In one sentence

**This project is a RAG chatbot with a debugging layer: it retrieves evidence, answers questions, and helps explain which stage may have caused a wrong or incomplete answer.**
