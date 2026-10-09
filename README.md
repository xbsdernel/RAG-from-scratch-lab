# RAG From Scratch

A complete, end-to-end Retrieval-Augmented Generation application implementing all nine topics from the "RAG From Scratch" series, built with Python 3.12, LangChain, ChromaDB, and Google Gemini.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Installation — Windows (PowerShell)](#installation--windows-powershell)
4. [Configuration](#configuration)
5. [Running the Application](#running-the-application)
6. [The Nine RAG Modes](#the-nine-rag-modes)
7. [Running Tests](#running-tests)
8. [Sample Documents and Evaluation](#sample-documents-and-evaluation)
9. [Troubleshooting](#troubleshooting)
10. [Privacy and Security](#privacy-and-security)
11. [Switching LLM Providers](#switching-llm-providers)
12. [Limitations and Known Issues](#limitations-and-known-issues)

---

## Architecture Overview

```
D:\RAG\
├── app.py                        # Streamlit entry point
├── requirements.txt
├── .env.example
├── .gitignore
├── rag_app/
│   ├── core/
│   │   ├── config.py             # All configuration from .env
│   │   ├── loaders.py            # PDF, DOCX, TXT, Markdown loaders
│   │   ├── chunker.py            # Text splitting + metadata
│   │   ├── embeddings.py         # E5 multilingual embeddings
│   │   ├── vector_store.py       # ChromaDB wrapper + deduplication
│   │   ├── indexing.py           # High-level indexing orchestrator
│   │   ├── llm_provider.py       # Gemini/Ollama/Groq adapter
│   │   ├── prompts.py            # All prompt templates
│   │   └── tracing.py            # Internal observability
│   └── pipelines/
│       ├── basic_rag.py          # Mode 1 & 3
│       ├── multi_query.py        # Mode 4
│       ├── rag_fusion.py         # Mode 5
│       ├── rrf.py                # RRF algorithm (pure logic)
│       ├── sequential_decomp.py  # Mode 6
│       ├── parallel_decomp.py    # Mode 7
│       ├── step_back.py          # Mode 8
│       ├── hyde.py               # Mode 9
│       └── query_generator.py    # Shared query/decomp generation
├── tests/
│   ├── test_chunking.py          # pytest: chunking tests
│   ├── test_rrf.py               # pytest: RRF unit tests
│   ├── test_loaders.py           # pytest: loader tests
│   ├── test_pipelines.py         # pytest: pipeline tests (mocked LLM)
│   └── run_stdlib_tests.py       # Standalone runner (no pytest)
├── data/
│   ├── sample_docs/              # Sample PDF/TXT/MD for testing
│   └── eval/
│       └── evaluation.py         # Recall@k and MRR evaluation
└── chroma_db/                    # Persistent vector store (auto-created)
```

### Component Responsibilities

| Component | Purpose |
|-----------|---------|
| `config.py` | Single source of truth for all settings; reads `.env` |
| `loaders.py` | File type detection, text extraction, metadata assignment |
| `chunker.py` | Recursive text splitting with `chunk_id` assignment |
| `embeddings.py` | `intfloat/multilingual-e5-small` with `query:`/`passage:` prefixes |
| `vector_store.py` | ChromaDB CRUD: add, search, delete, list, clear |
| `indexing.py` | Orchestrates: load → chunk → embed → store with deduplication |
| `llm_provider.py` | Provider-agnostic LLM factory with retry and error handling |
| `prompts.py` | All prompt templates isolated from pipeline logic |
| `tracing.py` | Per-run observability; never logs API keys |
| `rrf.py` | Pure Reciprocal Rank Fusion implementation; testable independently |
| `basic_rag.py` | Modes 1 & 3: standard retrieve → prompt → generate |
| `multi_query.py` | Mode 4: N query variations → merge → generate |
| `rag_fusion.py` | Mode 5: N queries → per-list ranks → RRF → generate |
| `sequential_decomp.py` | Mode 6: ordered subquestions with cumulative history |
| `parallel_decomp.py` | Mode 7: concurrent subquestions via `ThreadPoolExecutor` |
| `step_back.py` | Mode 8: specific + abstract dual retrieval |
| `hyde.py` | Mode 9: hypothetical passage embedding → real doc retrieval |
| `app.py` | Streamlit UI with 4 tabs: Chat, Documents, Inspect, Index Inspect |

---

## Prerequisites

- Python 3.12 (installed at `C:\Users\<you>\AppData\Local\Programs\Python\Python312\`)
- Internet access for first-time model download (~120 MB)
- A Google account to get a Gemini API key (free tier available)

---

## Installation — Windows (PowerShell)

### Step 1: Create and activate a virtual environment

```powershell
# Navigate to the project directory
cd D:\RAG

# Create virtual environment using Python 3.12
& "C:\Users\mohamed\AppData\Local\Programs\Python\Python312\python.exe" -m venv .venv

# Activate the virtual environment
.\.venv\Scripts\Activate.ps1
```

> **Note:** If you see a script execution error, run:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### Step 2: Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

> **First run note:** The embedding model (`intfloat/multilingual-e5-small`, ~120 MB) is downloaded automatically from HuggingFace on first use. Subsequent runs use the local cache. CPU inference is used by default — no GPU required, though inference will be slower than GPU.

---

## Configuration

### Step 1: Get your Gemini API key

1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Sign in with your Google account
3. Click **"Create API key"**
4. Copy the key

> **Free tier:** As of the time of writing, `gemini-2.0-flash` is available on the free tier with rate limits. Check current limits at [aistudio.google.com/rate-limit](https://aistudio.google.com/rate-limit). Free tier inputs/outputs may be used by Google to improve their products.

### Step 2: Create your `.env` file

```powershell
Copy-Item .env.example .env
```

Open `.env` in any text editor and replace `your_google_api_key_here` with your actual key:

```
GOOGLE_API_KEY=AIza...your_actual_key_here
GEMINI_MODEL=gemini-2.0-flash
```

> ⚠️ **Never commit `.env` to version control.** It is already in `.gitignore`.

### Optional: LangSmith tracing

If you have a [LangSmith](https://smith.langchain.com) account:

```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__your_langsmith_key_here
LANGCHAIN_PROJECT=rag-from-scratch
```

LangSmith is entirely optional. The application works fully without it.

---

## Running the Application

### Launch the Streamlit UI

```powershell
# Make sure your virtual environment is activated
.\.venv\Scripts\Activate.ps1

# Start the app
streamlit run app.py
```

The app opens at `http://localhost:8501` in your browser.

### Application Tabs

| Tab | Purpose |
|-----|---------|
| **💬 Chat** | Ask questions using the selected RAG strategy |
| **📂 Documents** | Upload, index, list, and delete documents |
| **🔬 Inspect Last Run** | View trace events, retrieved chunks, prompts, and strategy details |
| **🧪 Index Inspection** | Mode 2 diagnostic view: browse indexed chunks, test queries |

### Upload your first document

1. Click the **📂 Documents** tab
2. Click **Browse files** and select a PDF, DOCX, TXT, or Markdown file
3. Click **📥 Index Selected Files**
4. Switch to the **💬 Chat** tab and ask a question

---

## The Nine RAG Modes

Select the mode from the **RAG Strategy** dropdown in the sidebar.

### Mode 1 — Basic RAG
Standard pipeline: embed query → retrieve top-k → format prompt → generate answer.
The model is instructed to cite sources and acknowledge when evidence is insufficient.

### Mode 2 — Indexing & Retrieval Inspection
Diagnostic mode. Shows indexed chunks, their sizes, metadata, embedding model used, and retrieved results with L2 distance scores.

> **Score note:** L2 distances are NOT calibrated confidence percentages. Lower = more similar.

### Mode 3 — Generation & Prompt Inspection
Same as Mode 1, but exposes the full formatted prompt (with secrets removed), the retrieved context, and the raw model output in the Inspect tab.

### Mode 4 — Multi-Query Retrieval
Generates N alternative phrasings of your question using the LLM, retrieves for each, deduplicates by `chunk_id`, and generates one final answer from the merged context.

> **Parameter:** *Alternative queries* (sidebar) controls how many variations are generated.

### Mode 5 — RAG Fusion (RRF)
Extends Multi-Query with Reciprocal Rank Fusion. Each query produces a ranked list; RRF aggregates them into a single ranking. The formula:

```
RRF_score(d) = Σ  1 / (c + rank_r(d))
```

> **Parameter:** *RRF constant (c)* — default 60. Higher c gives more equal weighting across ranks.

### Mode 6 — Sequential Query Decomposition
Decomposes a complex question into ordered sub-questions. Each step receives previous Q&A history as context, enabling dependent reasoning chains.

> **Parameters:** *Sub-questions* controls decomposition depth. History is bounded to prevent context overflow.

### Mode 7 — Parallel Query Decomposition
Decomposes into independent sub-questions and answers them concurrently using `ThreadPoolExecutor`. A final synthesis step combines all partial answers.

> **Parameter:** *Max concurrency* — controls how many sub-questions are processed simultaneously.

### Mode 8 — Step-Back Prompting
1. Generates an abstract "step-back" question about the underlying principle.
2. Retrieves independently for both the original and step-back queries.
3. Combines both evidence sets in the final generation prompt.

### Mode 9 — Hypothetical Document Embeddings (HyDE)
1. Generates a hypothetical passage that would answer the question.
2. Embeds the hypothetical passage as a retrieval signal.
3. Uses that embedding to retrieve **real** documents.
4. Answers using only the real retrieved documents.

> ⚠️ **The hypothetical passage is AI-generated and NOT factual evidence.** It is displayed separately and never included in the final context.

---

## Running Tests

### With pytest (after installing dependencies):

```powershell
python -m pytest tests/ -v
```

### Without pytest (works immediately):

```powershell
python tests/run_stdlib_tests.py
```

### What each test file covers:

| File | Coverage |
|------|---------|
| `test_rrf.py` | 13 RRF unit tests: formula, aggregation, deduplication, tie-breaking, c-parameter effect |
| `test_chunking.py` | 11 chunking tests: splitting, metadata preservation, chunk IDs, overlap |
| `test_loaders.py` | 12 loader tests: UTF-8, Arabic text, metadata, empty files, unsupported formats |
| `test_pipelines.py` | 15 pipeline tests with mocked LLMs: Basic RAG, Multi-Query, Sequential, HyDE, Step-Back |

All pipeline tests use **mocked LLM responses** — no API key required.

---

## Sample Documents and Evaluation

### Sample Documents

Located in `data/sample_docs/`:
- `rag_overview.txt` — Comprehensive guide to RAG concepts
- `langchain_guide.md` — LangChain framework reference

### Running the Evaluation

```powershell
# Index the sample documents first (do this in the app UI)
# Then run:
python data/eval/evaluation.py
```

This reports **Recall@5** and **MRR** for predefined sample questions.

> **Important:** Meaningful evaluation requires accurate ground-truth `chunk_id` labels. The sample IDs in `evaluation.py` are illustrative — update them after indexing to match the actual assigned chunk IDs. A high score does NOT guarantee factual accuracy.

---

## Troubleshooting

### "GOOGLE_API_KEY is not configured"
Create `.env` from `.env.example` and set your key.

### "429 RESOURCE_EXHAUSTED"
You've hit the Gemini free-tier rate limit. Wait a few minutes and retry. The app implements exponential backoff automatically. Consider upgrading to a paid tier or switching to Ollama (local, free).

### "No module named 'langchain_core'"
Dependencies are not installed. Run:
```powershell
python -m pip install -r requirements.txt
```

### Embedding model download fails
The first run downloads ~120 MB from HuggingFace. Ensure you have an internet connection. The model is cached in `~/.cache/huggingface/` after the first download.

### "UnicodeEncodeError" on Windows terminal
Run PowerShell with:
```powershell
$env:PYTHONIOENCODING = "utf-8"
```

### Streamlit port conflict
```powershell
streamlit run app.py --server.port 8502
```

### ChromaDB error on first run
Ensure the `chroma_db/` directory is writable. Do not delete this directory unless you want to clear the entire index.

---

## Privacy and Security

- **API keys:** Stored in `.env` only. Never hardcoded, never logged, never exposed in the UI.
- **Data sent to Google:** When using the Gemini API, your prompts (including retrieved document text) are sent to Google's servers. Google may use free-tier inputs to improve their products. For sensitive documents, consider using a local model (see [Switching LLM Providers](#switching-llm-providers)).
- **Local embeddings:** `intfloat/multilingual-e5-small` runs entirely on your machine. No text is sent externally for embedding.
- **ChromaDB:** All indexed vectors are stored locally on disk in `chroma_db/`.

---

## Switching LLM Providers

The LLM provider is modular. Change the **LLM Provider** dropdown in the sidebar to switch between:

| Provider | Requirements |
|----------|-------------|
| `gemini` (default) | `GOOGLE_API_KEY` in `.env` |
| `ollama` | Ollama running locally (`ollama serve`), model pulled (`ollama pull llama3.2`) |
| `groq` | `GROQ_API_KEY` in `.env`, `pip install langchain-groq` |

To add a new provider, implement it in `rag_app/core/llm_provider.py` in the `get_llm()` function.

---

## Limitations and Known Issues

| Item | Detail |
|------|--------|
| LLM not installed | Requires `langchain-google-genai` and other packages from `requirements.txt` |
| GPU not used for embeddings | CPU-only inference. Expect 1–5s per embedding batch for large documents |
| Free-tier rate limits | Gemini free tier has RPM/TPD limits that vary over time. Not permanently free |
| HyDE hallucination | Hypothetical passages may be irrelevant or incorrect. They are retrieval aids only |
| Evaluation labels | Sample `chunk_id` values in `evaluation.py` must be updated after indexing |
| Parallel concurrency | Limited by Gemini API rate limits. High concurrency may trigger 429 errors |
| Arabic language | Supported by the embedding model. LLM output language depends on Gemini |
