# 🔍 RAG Failure Diagnosis Tool

### An AI-powered debugging and evaluation framework for Retrieval-Augmented Generation (RAG) systems

A configurable, project-independent tool designed to identify potential failures in RAG pipelines, evaluate retrieved evidence, validate citations, and check whether generated answers are supported by source documents.

Unlike a standard RAG chatbot that focuses only on answering questions, this project also examines **how the answer was generated, whether the retrieved evidence is relevant, and what may have caused an unreliable response.**

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Why This Project?](#-why-this-project)
- [Key Highlights](#-key-highlights)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [How RAG Works](#-how-rag-works)
- [Fast Mode vs Exhaustive Mode](#-fast-mode-vs-exhaustive-mode)
- [RAG Failure Diagnosis](#-rag-failure-diagnosis)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Prerequisites](#-prerequisites)
- [Installation and Setup](#-installation-and-setup)
- [Environment Configuration](#-environment-configuration)
- [Database Requirements](#-database-requirements)
- [How to Run](#-how-to-run)
- [Example Usage](#-example-usage)
- [Understanding the Diagnostic Report](#-understanding-the-diagnostic-report)
- [How to Adapt It to Other Projects](#-how-to-adapt-it-to-other-projects)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [Resume Description](#-resume-description)

---

## 📖 Project Overview

Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with Large Language Models (LLMs).

Instead of answering only from their pre-trained knowledge, RAG systems retrieve relevant information from external documents and provide that information to an LLM as context.

However, a RAG system can still produce incorrect, incomplete, or unsupported answers.

For example:

- The search engine retrieves irrelevant document passages.
- The correct information exists in the database but is not included in the retrieved results.
- The answer contains information that cannot be verified from the supplied evidence.
- Citations are missing or refer to chunks that were not retrieved.
- Important details are lost because documents were split into unsuitable chunks.
- Different source passages contain conflicting information.

This project adds a diagnostic layer to the RAG workflow. It retrieves document chunks, generates answers using an LLM, and evaluates several aspects of retrieval and answer quality.

The goal is to make RAG systems more **transparent, debuggable, and easier to evaluate**.

### Main Objectives

1. Retrieve relevant document chunks using hybrid search.
2. Generate answers grounded in retrieved source material.
3. Evaluate the relevance of retrieved chunks.
4. Validate citations against the retrieved evidence.
5. Audit whether factual claims are supported by source passages.
6. Flag potentially unsupported numerical values and dates.
7. Measure retrieval recall when ground-truth evidence is available.
8. Support both focused questions and comprehensive document analysis.
9. Keep the core workflow configurable so it can be adapted to compatible document-based RAG projects.

---

## 🎯 Why This Project?

A traditional RAG application primarily focuses on producing an answer.

But when an answer is incorrect, several different problems may have caused the failure:

- Was the correct document passage retrieved?
- Did the retriever rank irrelevant passages too highly?
- Did the LLM ignore relevant evidence?
- Did the model invent an unsupported fact?
- Did the citation point to the correct chunk?
- Was the required information missing from the indexed data?

Without a diagnostic workflow, identifying the source of an error can require manually inspecting the entire pipeline.

This project helps investigate these questions by evaluating retrieval, citations, answer support, and potential failure causes.

| Traditional RAG System | RAG Failure Diagnosis Tool |
|---|---|
| Retrieves document passages | Evaluates whether retrieved passages appear relevant |
| Generates an answer | Audits factual claims against retrieved evidence |
| May include citations | Checks whether citation IDs exist in the retrieved set |
| May produce unsupported details | Flags some potentially unsupported numbers and dates |
| Can miss important evidence | Supports optional ground-truth retrieval recall |
| Usually uses a limited context | Provides an Exhaustive Mode for all indexed chunks of a selected document |

**Important:** These evaluations provide diagnostic signals, not guaranteed proof that an answer is correct or incorrect.

---

## 🌟 Key Highlights

### 1. Hybrid Retrieval

Combines two retrieval techniques:

- **Vector Search:** Finds passages with semantically similar meaning.
- **Keyword Search:** Finds passages using lexical matches through PostgreSQL full-text search.

The two rankings are combined using Reciprocal Rank Fusion (RRF).

### 2. RAG Answer Evaluation

Uses an LLM evaluator to inspect whether generated factual claims are supported by the retrieved document passages.

### 3. Citation Validation

Checks whether the citation IDs included in the answer correspond to chunks retrieved for that query.

### 4. Fast Mode

Retrieves a limited number of high-ranking chunks for ordinary questions, helping reduce the amount of context sent to the LLM.

### 5. Exhaustive Mode

Processes all indexed chunks associated with a selected document in batches, then combines the extracted findings into a consolidated answer with citations.

### 6. Potential Failure Identification

Highlights possible retrieval problems, missing citations, unsupported claims, and other issues requiring investigation.

### 7. Ground-Truth Evaluation

Can calculate evidence recall when the expected relevant chunk IDs are known.

### 8. Configurable Architecture

Database connection details, table and column names, embedding settings, LLM endpoints, and retrieval limits can be configured through environment variables.

---

## ⚙️ Features

| Feature | Description |
|---|---|
| Hybrid Search | Combines vector similarity and PostgreSQL full-text search |
| Reciprocal Rank Fusion | Merges the vector and keyword rankings |
| Semantic Embeddings | Converts questions into vectors for semantic retrieval |
| LLM Answer Generation | Generates answers using retrieved source passages |
| Citation Validation | Checks recognized citation IDs against retrieved chunk IDs |
| Retrieval Evaluation | Estimates the relevance of retrieved chunks |
| Answer Support Audit | Evaluates whether factual claims are supported by source evidence |
| Numeric/Date Warnings | Flags some values that deserve manual verification |
| Ground-Truth Recall | Measures retrieval recall when expected evidence IDs are provided |
| Fast Mode | Answers questions using a limited top-k set of chunks |
| Exhaustive Mode | Processes all indexed chunks for a selected document in batches |
| Environment Configuration | Allows settings to be changed without hardcoding a project ID |

---

## 🏗️ System Architecture

The application follows a retrieval, generation, and evaluation workflow.

```text
                    USER QUESTION
                          |
                          v
                 LOAD CONFIGURATION
                          |
                          v
               DETECT DOCUMENT ID
                    IF PRESENT
                          |
                          v
                   SELECT MODE
                    /       \
                   /         \
                  v           v
             FAST MODE    EXHAUSTIVE MODE
                  |           |
                  v           v
          CREATE QUERY     LOAD ALL INDEXED
           EMBEDDING       DOCUMENT CHUNKS
                  |           |
                  v           v
           VECTOR SEARCH   PROCESS IN BATCHES
                  |           |
                  v           v
           KEYWORD SEARCH  EXTRACT EVIDENCE
                  |           |
                  v           v
            RECIPROCAL RANK  COMBINE FINDINGS
             FUSION (RRF)    WITH CITATIONS
                  |           |
                  v           |
            SELECT TOP-K       |
              CHUNKS           |
                  |           |
                  +-----+-----+
                        |
                        v
                 GENERATE ANSWER
                        |
                        v
                 DISPLAY SOURCES
                        |
                        v
               DIAGNOSTIC MODE?
                  /         \
                 NO          YES
                 |            |
                 v            v
          CHECK CITATIONS   EVALUATE RETRIEVAL
          AND DISPLAY      VALIDATE CITATIONS
          THE ANSWER       AUDIT ANSWER SUPPORT
                           CHECK NUMBER/DATE CLAIMS
                           CHECK GROUND-TRUTH RECALL
                           REPORT POSSIBLE CAUSES
```

### Workflow Explanation

**Step 1: Receive a question**

The user enters a question through the command-line interface.

**Step 2: Identify the required document**

If the question contains a recognizable document ID, the application can filter retrieval to that document.

**Step 3: Retrieve evidence**

Fast Mode combines vector and keyword retrieval. Exhaustive Mode loads all indexed chunks associated with the selected document.

**Step 4: Generate an answer**

The retrieved passages are passed to the LLM, which is instructed to answer from the supplied evidence and preserve citations.

**Step 5: Evaluate the answer**

When diagnostic mode is requested, additional evaluators inspect the retrieval results, citation markers, and answer claims.

**Step 6: Report potential problems**

The application displays diagnostic results and possible root causes that a developer can investigate.

---

## 🧠 How RAG Works

RAG stands for **Retrieval-Augmented Generation**.

It combines an information retrieval system with a generative language model.

### Retrieval

The system searches a collection of indexed document chunks to find passages relevant to the user's question.

For example, a question about eligibility requirements should retrieve passages describing eligibility conditions rather than unrelated document metadata.

### Augmentation

The retrieved passages are added to the prompt as context for the LLM.

### Generation

The LLM uses the supplied context to construct an answer and include source citations.

### Diagnosis

This project adds an evaluation stage that checks whether the retrieved evidence appears relevant and whether the generated answer is supported by that evidence.

The system works with **already indexed document chunks**. Document ingestion, PDF extraction, OCR, chunking, and embedding generation must be handled by an upstream pipeline.

---

## ⚡ Fast Mode vs Exhaustive Mode

The project supports two answering strategies.

### Fast Mode — Focused Retrieval

Fast Mode is intended for ordinary questions where a limited number of relevant passages should provide enough context.

#### Workflow

1. Convert the question into an embedding.
2. Retrieve similar chunks through vector search.
3. Retrieve keyword matches through PostgreSQL full-text search.
4. Merge the rankings using Reciprocal Rank Fusion.
5. Select the highest-ranking chunks.
6. Send the selected chunks to the LLM.
7. Generate an answer with citations.

#### Example questions

- What is the purpose of this document?
- What are the eligibility requirements?
- What is the contract duration?

#### Advantages

- Sends less context to the LLM.
- Usually requires fewer tokens than exhaustive processing.
- Suitable for specific, focused questions.

#### Trade-off

The relevant evidence may be missed if it does not appear in the retrieved top-k results.

The default `FINAL_TOP_K` value in the example configuration is 8. Set it to 10 if you want Fast Mode to send the top 10 merged chunks.

### Exhaustive Mode — Comprehensive Document Analysis

Exhaustive Mode is intended for questions requiring broad coverage of one document.

#### Workflow

1. Detect the document ID from the question.
2. Retrieve all indexed chunks associated with that document.
3. Divide the chunks into batches.
4. Ask the LLM to extract relevant facts from each batch.
5. Preserve source citations during extraction.
6. Combine the batch findings into a consolidated answer.

#### Example questions

- List all eligibility requirements for this document.
- Provide the complete list of exceptions.
- Identify every condition mentioned in the document.

#### Advantages

- Reviews more indexed evidence than ordinary top-k retrieval.
- Can help uncover relevant details distributed across different chunks.
- Supports comprehensive questions that require evidence from multiple passages.

#### Trade-offs

- Requires more model calls.
- Can take longer than Fast Mode.
- Uses more tokens.
- May still omit details during extraction or final synthesis.

**Important:** Exhaustive Mode processes all indexed chunks for the selected document. It cannot recover information missing from the original extraction or indexing pipeline.

---

## 🔎 RAG Failure Diagnosis

The diagnostic workflow is activated by adding `diagnose:` before the question.

Example:

```text
diagnose: What are the eligibility requirements?
```

The tool evaluates several aspects of the RAG pipeline.

### 1. Retrieval Relevance

The evaluator examines whether the retrieved passages appear useful for answering the question.

Possible overall assessments include:

- `LIKELY_RELEVANT`
- `MIXED_RELEVANCE`
- `LIKELY_RETRIEVAL_FAILURE`

Individual chunks may be rated as `HIGH`, `MEDIUM`, or `LOW` relevance.

**Example:** If the question asks for a financial eligibility threshold but the retrieved chunks discuss unrelated document details, the evaluator may flag a potential retrieval problem.

However, irrelevant retrieved passages do not prove that the correct evidence exists elsewhere. Additional searches or ground-truth evidence are required to confirm a retrieval miss.

### 2. Citation Validation

The answer may include markers such as:

```text
[DB CHUNK ID: 123]
```

The citation checker compares the recognized IDs against the chunks retrieved for that question.

Possible statuses:

- `NO_CITATIONS` — No recognized citation markers were found.
- `INVALID_CITATIONS` — At least one citation refers to a chunk outside the retrieved set.
- `CITATIONS_PRESENT` — Recognized citations were found and their IDs belong to the retrieved set.

A valid citation ID does not prove that the cited passage supports the claim. Semantic support is examined separately.

### 3. Answer Support Audit

The LLM evaluator compares the answer's factual claims with the supplied evidence.

Possible overall assessments:

- `SUPPORTED`
- `PARTIALLY_SUPPORTED`
- `UNSUPPORTED`
- `NOT_EVALUABLE`

At claim level, the evaluator may use:

- `SUPPORTED`
- `CONTRADICTED`
- `NOT_FOUND`
- `UNCLEAR`

The report includes claims, evidence chunk IDs, and explanations.

These are LLM-generated judgments and should be reviewed against the actual source passages.

### 4. Numeric and Date Warnings

The tool applies a lightweight text-matching heuristic to flag some numerical values and dates that do not appear in cited source text.

This can help identify values that deserve additional review.

However, different number formats, punctuation, currency symbols, or date representations may produce false positives or false negatives.

**A warning does not prove that a number or date is incorrect.**

### 5. Ground-Truth Retrieval Recall

When expected evidence chunk IDs are known, the system can calculate retrieval recall.

The formula is:

```text
Recall = Expected evidence chunks retrieved
         ----------------------------------
         Total expected evidence chunks
```

For example, suppose a test case has four expected evidence chunks and retrieval returns three of them.

```text
Recall = 3 / 4 = 0.75 = 75%
```

Ground-truth recall is optional because the system cannot automatically know the correct evidence for every arbitrary question.

### 6. Possible Root Causes

The diagnostic report combines its signals to highlight potential issues such as:

- No retrieval results.
- Missing or invalid citations.
- Irrelevant retrieved passages.
- Unsupported answer claims.
- Missing expected evidence in a labeled test.
- Numeric or date values requiring verification.

These are possible root causes rather than guaranteed diagnoses.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application logic and orchestration |
| PostgreSQL | Stores document chunks and metadata |
| pgvector | Vector similarity search |
| Sentence Transformers | Generates query embeddings |
| Nomic Embeddings | Default configurable embedding model |
| PostgreSQL Full-Text Search | Keyword-based retrieval |
| Reciprocal Rank Fusion | Combines retrieval rankings |
| LM Studio | Runs a local LLM through an API |
| SQLAlchemy | Database connections and SQL execution |
| Requests | Communicates with the LLM API |
| python-dotenv | Loads environment variables |

The default embedding configuration uses `nomic-ai/nomic-embed-text-v1.5`, with an expected vector dimension of 768.

If a different embedding model is used, its output dimensions and query-prefix requirements must match the stored embeddings and model documentation.

---

## 📁 Project Structure

A recommended project layout is:

```text
rag-failure-diagnosis-tool/
│
├── rag_failure_diagnosis.py
├── README.md
├── requirements.txt
├── .env
├── .env.example
│
└── tests/
    ├── test_citations.py
    ├── test_document_id_extraction.py
    └── test_diagnostic_cases.py
```

The test files shown above are suggested additions and must be created separately if they do not already exist.

### File Descriptions

- `rag_failure_diagnosis.py` — Main application.
- `.env` — Local database and model configuration; must not be committed with real credentials.
- `.env.example` — Configuration template with placeholder values.
- `requirements.txt` — Python dependencies.
- `README.md` — Project documentation.
- `tests/` — Suggested location for automated tests.

---

## 💻 Prerequisites

Before running the application, ensure you have:

- Python 3.10 or newer.
- PostgreSQL with a table containing indexed document chunks.
- The PostgreSQL `vector` extension enabled for vector search.
- A compatible embedding model.
- A running LM Studio server or another compatible OpenAI-style API endpoint.
- A model loaded in the LLM server.
- Valid database connection details.
- Embedding dimensions compatible with the database vector column.

The embedding model may need to be downloaded the first time the application runs.

---

## 🚀 Installation and Setup

### Step 1: Create a Project Directory

Create a folder for the project and place `rag_failure_diagnosis.py` inside it.

### Step 2: Create a Virtual Environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install python-dotenv sqlalchemy "psycopg[binary]" requests sentence-transformers
```

For reproducible installations, record the dependency versions that you successfully test in `requirements.txt`.

### Step 4: Configure Environment Variables

Create a `.env` file in the project root.

Copy the configuration template below and update it to match your own database and model server.

### Step 5: Start LM Studio

1. Open LM Studio.
2. Select and load a compatible chat model.
3. Start the local server.
4. Confirm that the server URL matches your configuration.

### Step 6: Verify the Database

Ensure that the configured chunk table contains document text and embeddings, and that the database user has permission to query the table.

### Step 7: Run the Application

```bash
python rag_failure_diagnosis.py
```

---

## 🔐 Environment Configuration

Use environment variables to adapt the application to different projects.

Example `.env` template:

```dotenv
# Database connection
DATABASE_URL=postgresql+psycopg://USER:PASSWORD@HOST:5432/DATABASE

# Chunk table configuration
DB_SCHEMA=public
CHUNKS_TABLE=tender_chunks

# Required logical columns
CHUNK_ID_COLUMN=id
DOCUMENT_ID_COLUMN=tender_id
CHUNK_TEXT_COLUMN=chunk_text
EMBEDDING_COLUMN=embedding_hf

# Optional metadata columns
DOCUMENT_TYPE_COLUMN=document_type
DOCUMENT_URL_COLUMN=document_url
CHUNK_INDEX_COLUMN=chunk_index

# Embedding configuration
EMBEDDING_MODEL_NAME=nomic-ai/nomic-embed-text-v1.5
EMBEDDING_DIMENSIONS=768
EMBEDDING_QUERY_PREFIX=search_query:

# LLM endpoint
LM_STUDIO_BASE_URL=http://localhost:1234/v1
LM_STUDIO_MODEL=
LM_STUDIO_TIMEOUT=600

# Retrieval configuration
VECTOR_CANDIDATE_LIMIT=30
KEYWORD_CANDIDATE_LIMIT=30
FINAL_TOP_K=8
RRF_K=60
EXHAUSTIVE_BATCH_SIZE=4

# Document ID detection
DOCUMENT_ID_LABELS=document,record,tender,contract,case,project
```

### Configuration Notes

- `DATABASE_URL` specifies the database connection.
- `DB_SCHEMA` and `CHUNKS_TABLE` specify where the chunks are stored.
- The column variables map the application's logical fields to your table.
- Optional metadata columns should be configured to match the actual table; use empty values only if the script supports that field being absent.
- `EMBEDDING_MODEL_NAME` selects the embedding model.
- `EMBEDDING_DIMENSIONS` must match both the model output and the database vector column.
- `EMBEDDING_QUERY_PREFIX` must match the selected model's documented query-encoding requirements.
- `LM_STUDIO_BASE_URL` specifies the OpenAI-compatible API base URL.
- `LM_STUDIO_MODEL` can be left blank if the application supports automatic model discovery.
- `VECTOR_CANDIDATE_LIMIT` controls the number of vector-search candidates.
- `KEYWORD_CANDIDATE_LIMIT` controls the number of keyword-search candidates.
- `FINAL_TOP_K` controls the number of final chunks passed to the LLM in Fast Mode.
- `RRF_K` controls the rank-fusion smoothing constant.
- `EXHAUSTIVE_BATCH_SIZE` controls the number of chunks processed per exhaustive extraction call.
- `DOCUMENT_ID_LABELS` specifies words that may introduce a document ID in a question.

The example values are starting points. Confirm that each variable is supported by your copy of the script before relying on it; configuration names must match the implementation.

**Security:** Never commit a `.env` file containing real database passwords or secrets. Commit only a placeholder `.env.example`.

---

## 🗄️ Database Requirements

The configured table must contain indexed document chunks.

At minimum, the application needs the following logical fields:

| Logical Field | Purpose |
|---|---|
| Chunk ID | Unique identifier for each chunk |
| Document ID | Groups chunks belonging to the same document |
| Chunk Text | Text passage used for retrieval and generation |
| Embedding Vector | Vector representation used for similarity search |

Optional metadata may include:

- Document type.
- Source URL.
- Chunk index.

The example names in the configuration are defaults, not universal requirements.

### Vector Dimension Compatibility

The embedding model output dimension must match the configured PostgreSQL vector column.

For example, a 768-dimensional embedding requires a compatible 768-dimensional vector column.

### Full-Text Search

The current retrieval approach uses PostgreSQL full-text search with `to_tsvector` and `plainto_tsquery`.

The configured chunk-text column must contain searchable text.

### Database Permissions

The database account must have the required permissions to read the configured chunk table and execute the search queries.

---

## ▶️ How to Run

Activate your environment and run:

```bash
python rag_failure_diagnosis.py
```

The application will display a question prompt.

### Ordinary Question

```text
Ask a question: What is the purpose of this document?
```

The application retrieves evidence, generates an answer, displays source information, and checks citations.

### Diagnostic Question

```text
Ask a question: diagnose: What are the eligibility requirements?
```

The application generates an answer and prints the diagnostic report.

### Exit the Application

```text
quit
```

Depending on the implementation, `exit` or `q` may also be accepted.

---

## 🧪 Example Usage

Use questions relevant to the documents in your database.

### Example 1: General Question

```text
What is the purpose of this document?
```

Expected workflow:

1. Search for relevant chunks.
2. Generate an answer.
3. Display the retrieved sources.
4. Check citation IDs.

### Example 2: Unsupported Information

```text
diagnose: What is the exact annual turnover threshold?
```

This can test whether the model acknowledges missing evidence instead of inventing a value.

The diagnosis should also be checked to determine whether the retrieved chunks were irrelevant or whether the indexed collection genuinely lacks the information.

### Example 3: Comprehensive Question

```text
List all eligibility requirements for document ABC123.
```

If the ID format is recognized and exhaustive-mode conditions are satisfied, the system can process all indexed chunks associated with that document in batches.

### Example 4: Numeric or Date Claims

```text
diagnose: What are the important dates and monetary values?
```

The diagnostic report may flag values that deserve manual verification against the cited passages.

These examples illustrate testing ideas, not guaranteed results.

---

## 📊 Understanding the Diagnostic Report

The report includes several sections.

| Section | Purpose |
|---|---|
| Retrieval Relevance | Estimates whether retrieved chunks are useful |
| Citation Validation | Checks recognized citations against retrieved IDs |
| Answer Support Audit | Evaluates support for factual claims |
| Number/Date Warnings | Flags some potentially unsupported values |
| Ground-Truth Recall | Measures expected evidence retrieval when labels are provided |
| Possible Root Causes | Summarizes issues that may deserve investigation |

### How to Investigate a Failure

1. Read the generated answer.
2. Inspect the cited passages.
3. Search the full indexed document if the relevant evidence is missing.
4. If the evidence exists but was not retrieved, investigate embeddings, search queries, ranking, and filters.
5. If the evidence was retrieved but the answer misrepresents it, investigate the prompt and generation process.
6. If citations are missing, inspect the answer-generation instructions and citation parser.
7. If the original document text is incomplete, investigate extraction, OCR, and chunking.
8. Record the question, expected evidence, retrieved chunks, and observed failure as a regression test.

---

## 🔄 How to Adapt It to Other Projects

The application is designed to avoid hardcoding a particular project or document ID.

To reuse it:

1. Identify a table containing indexed chunks.
2. Configure the database schema and table name.
3. Map the chunk ID, document ID, text, and embedding columns.
4. Select an embedding model compatible with the stored vectors.
5. Configure the model's embedding dimensions and query prefix.
6. Configure the LLM API endpoint.
7. Adjust document-ID detection for your dataset's identifier format.
8. Test vector and keyword search separately.
9. Create labeled evaluation cases with expected evidence IDs.
10. Run diagnostic tests and inspect their accuracy.

### What Project-Independent Means

The application can be adapted to compatible PostgreSQL-backed RAG systems through configuration.

It is not automatically compatible with every database, arbitrary schema, or every vector store. A different database engine or retrieval backend may require a dedicated adapter.

Document-ID detection is also based on configurable patterns rather than a universal identifier standard.

---

## ⚠️ Limitations

This project is a diagnostic aid, not a guaranteed correctness engine.

### 1. No Automatic Document Ingestion

The application expects document chunks and embeddings to exist already. PDF extraction, OCR, chunking, and indexing are handled separately.

### 2. LLM Evaluators Can Be Wrong

An LLM evaluator can incorrectly judge relevance or answer support. Human review and labeled test cases remain important.

### 3. Retrieval Failure Is Not Always Confirmed

Irrelevant retrieved chunks may indicate a retrieval problem, but the correct evidence might also be missing from the indexed collection.

### 4. Citation Validation Is Syntactic

A citation ID can exist in the retrieved set without actually supporting the associated claim.

### 5. Numeric/Date Warnings Are Heuristic

The warning mechanism may produce false positives and false negatives because text matching does not fully understand semantic equivalence.

### 6. Exhaustive Mode Has Limits

It processes indexed chunks, not necessarily every fact in the original document. Information lost during ingestion can remain unavailable.

### 7. Ground-Truth Evaluation Requires Labels

Retrieval recall cannot be measured accurately without known expected evidence IDs.

### 8. Not Every Failure Type Has a Dedicated Classifier

The current approach identifies possible root causes from a combination of LLM evaluations and heuristic checks. A specialized, independently validated classifier for every failure category would require additional development and testing.

---

## 🚀 Future Improvements

Potential improvements include:

- Building a labeled RAG failure benchmark.
- Adding automated regression tests for known failure cases.
- Measuring Recall@K, Precision@K, Mean Reciprocal Rank, and nDCG.
- Implementing a dedicated conflicting-evidence detector.
- Detecting potential chunk-boundary problems using neighboring passages.
- Adding a reranker to improve retrieval ranking.
- Storing diagnostic reports for historical comparison.
- Developing a dashboard to inspect evidence, citations, and failure reports.
- Adding adapters for additional vector databases.
- Improving structured output validation for LLM evaluators.
- Benchmarking Fast Mode and Exhaustive Mode for latency, token usage, and answer coverage.
- Adding document-level extraction and indexing quality checks.
- Evaluating diagnostic accuracy against human-labeled test cases.

These improvements could turn the project into a more comprehensive RAG evaluation and observability platform.

---

## 💼 Resume Description

**RAG Failure Diagnosis Tool | Python, PostgreSQL, pgvector, Sentence Transformers, LM Studio**

- Developed a configurable RAG evaluation and debugging tool combining vector similarity search and PostgreSQL full-text search using Reciprocal Rank Fusion.
- Integrated a local LLM endpoint to generate evidence-grounded answers and evaluate retrieval relevance, citation validity, and claim support.
- Implemented Fast Mode for focused top-k retrieval and Exhaustive Mode for batched analysis of all indexed chunks associated with a selected document.
- Added numeric/date warning heuristics and optional ground-truth evidence recall to support systematic RAG evaluation.

Only include performance improvements or accuracy metrics after measuring them on a defined test set.

---

## 🎤 Interview Explanation

"I built a RAG Failure Diagnosis Tool to help investigate why a Retrieval-Augmented Generation system might return incorrect, incomplete, or unsupported answers.

The tool uses hybrid retrieval, combining vector similarity search with PostgreSQL full-text search. It merges the rankings using Reciprocal Rank Fusion and sends the retrieved evidence to a locally hosted LLM through LM Studio.

It supports two modes. Fast Mode retrieves a limited number of relevant chunks for normal questions, while Exhaustive Mode processes all indexed chunks for a selected document in batches to answer comprehensive questions.

Beyond answering questions, the tool evaluates retrieval relevance, validates citations, audits factual claims against the retrieved evidence, and flags potentially unsupported numbers and dates. It can also measure retrieval recall when expected evidence chunks are known.

The goal is to make RAG pipelines easier to debug by examining not only the generated answer, but also the evidence and potential reasons why that answer may be unreliable."

---

## 🏁 Conclusion

The RAG Failure Diagnosis Tool extends a conventional RAG workflow with diagnostic checks around retrieval and generation.

Its key value is helping developers inspect retrieved evidence, validate citations, evaluate claim support, and identify potential causes of unreliable answers.

The combination of **hybrid retrieval, Fast and Exhaustive modes, citation validation, answer-support auditing, and optional ground-truth evaluation** provides a foundation for systematically testing and improving RAG applications.

The next step toward a production-ready evaluation platform is to build a labeled benchmark, measure diagnostic accuracy, and validate each failure category against known examples.

**The objective is not merely to generate answers—it is to make RAG systems easier to inspect, evaluate, and improve.**
