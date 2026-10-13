# Reliable RAG API

An AI-powered knowledge assistant that answers questions from a collection of documents.  
Answers are **grounded in retrieved evidence**, clearly **abstain when the answer is absent**, and **never invent facts**.

---

## Table of Contents

- [Architecture](#architecture)
- [Setup](#setup)
- [Running the API](#running-the-api)
- [Docker](#docker)
- [API Reference](#api-reference)
- [Automated Tests](#automated-tests)
- [Evaluation](#evaluation)
- [Limitations](#limitations)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         FastAPI Application                     │
│                                                                 │
│  POST /documents          POST /ask            GET /health      │
│       │                        │                               │
│  ┌────▼──────────┐   ┌─────────▼──────────┐                   │
│  │ IngestionSvc  │   │     QAService       │                   │
│  │               │   │                    │                   │
│  │ 1. Load doc   │   │ 1. Retrieve chunks  │                   │
│  │ 2. Chunk text │   │ 2. Filter by score  │                   │
│  │ 3. Embed      │   │ 3. Call LLM         │                   │
│  │ 4. Store      │   │ 4. Validate cites   │                   │
│  └──────┬────────┘   └─────────┬──────────┘                   │
│         │                      │                               │
│  ┌──────▼──────────────────────▼──────────┐                   │
│  │           VectorStore (ChromaDB)        │                   │
│  └────────────────────────────────────────┘                   │
│         │                      │                               │
│  ┌──────▼──────┐     ┌─────────▼──────────┐                   │
│  │  Embeddings │     │     Groq LLM        │                   │
│  │ (MiniLM-L6) │     │ (llama-3.1-8b-inst) │                   │
│  └─────────────┘     └────────────────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| **Vector store** | ChromaDB (persistent) | Zero-infrastructure, file-backed, cosine distance built-in |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` | 384-dim, fast, high-quality for English retrieval |
| **LLM** | Groq API (`llama-3.1-8b-instant`) | Free tier, sub-second latency, JSON mode supported |
| **Chunking** | Character sliding window (2000 chars, 300 overlap) | Paragraph-aware breaks; page metadata preserved for PDFs |
| **Reliability** | Citation validation layer | LLM-cited chunk IDs are cross-checked against retrieved set; hallucinated IDs cause abstention |
| **Abstention** | Two-stage | (1) No chunks pass relevance filter → abstain before LLM call; (2) LLM says `found_in_documents=false` or cites no valid IDs → abstain |
| **Deduplication** | SHA-256 content hash | Re-uploading identical content returns `status: duplicate` without re-indexing |

---

## Setup

### Prerequisites

- Python 3.11+
- A [Groq API key](https://console.groq.com/) (free)

### 1. Clone and create virtual environment

```bash
git clone <repo-url>
cd RagAPI
python -m venv venv
```

### 2. Install dependencies

```bash
# Windows
venv\Scripts\pip install -r requirements.txt

# Linux / macOS
venv/bin/pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set your Groq API key:

```env
GROQ_API_KEY=gsk_your_key_here
```

All other defaults work out of the box.

---

## Running the API

```bash
# Windows
venv\Scripts\uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Linux / macOS
venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

| URL | Description |
|---|---|
| http://localhost:8000/docs | Interactive Swagger UI |
| http://localhost:8000/health | Health check (`{"status":"ok"}`) |
| http://localhost:8000/ready | Readiness (embedding backend + chroma directory) |

---

## Docker

```bash
# Build and start
docker-compose up --build

# Detached
docker-compose up -d --build
```

The compose file mounts named volumes for ChromaDB data and the Hugging Face model cache so both survive container restarts.

> **Note**: Edit `docker-compose.yml` to add `GROQ_API_KEY` under `environment`, or create a `.env` file that Docker Compose picks up automatically.

---

## API Reference

### `POST /documents` — Ingest a document

Upload a PDF, TXT, or Markdown file.

```bash
curl -X POST http://localhost:8000/documents \
  -F "file=@sample_documents/employee_handbook.md"
```

**Response**
```json
{
  "status": "ingested",
  "filename": "employee_handbook.md",
  "document_id": "sha256-hash",
  "chunk_count": 3,
  "is_duplicate": false
}
```

Status is `"duplicate"` if the same content was already indexed (checked by content hash).

---

### `POST /ask` — Ask a question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many annual leave days do employees receive?"}'
```

**Response when answer is found**
```json
{
  "question": "How many annual leave days do employees receive?",
  "answer": "Employees receive 18 days of annual leave each calendar year.",
  "found_in_documents": true,
  "sources": [
    {
      "document": "employee_handbook.md",
      "excerpt": "Employees receive 18 days of annual leave each calendar year...",
      "chunk_id": "abc123:0",
      "page": null
    }
  ]
}
```

**Response when answer is not found**
```json
{
  "question": "What is the 401k match?",
  "answer": "The information is unavailable in the provided documents.",
  "found_in_documents": false,
  "sources": []
}
```

Optional request parameters:
- `top_k` (int, 1–20): number of chunks to retrieve (overrides server default)
- `relevance_threshold` (float, 0–2): cosine distance cutoff (lower = stricter)

---

### `GET /health` and `GET /ready`

Liveness and readiness probes used by Docker and load balancers.

---

## Automated Tests

Tests use `FakeEmbeddings` (deterministic hash vectors) and `ScriptedLLM` (configurable mock) — **no API key or network access required**.

```bash
# Run all 42 tests
venv\Scripts\python -m pytest          # Windows
venv/bin/python -m pytest              # Linux/macOS

# Verbose output
python -m pytest -v

# Unit tests only
python -m pytest tests/unit

# Integration tests only
python -m pytest tests/integration
```

### Test coverage summary

| Suite | File | What is tested |
|---|---|---|
| Unit | `test_chunking.py` | Chunk IDs, page assignment, overlap windows |
| Unit | `test_document_loader.py` | PDF/TXT/MD loading, empty files, bad extensions |
| Unit | `test_llm_parse.py` | JSON extraction from plain and fenced LLM output |
| Unit | `test_qa.py` | Grounded answers, abstention, citation validation, error propagation |
| Unit | `test_retrieval.py` | Relevance filtering, empty DB, duplicate indexing, model mismatch |
| Integration | `test_documents_api.py` | Upload success, duplicate, oversize, malformed PDF, bad extension |
| Integration | `test_questions_api.py` | Happy path, empty KB, empty question, response schema |
| Integration | `test_failure_cases.py` | 503 on LLM outage, no hallucination on malformed output, fabricated chunk IDs rejected |

---

## Evaluation

### Running the evaluation

```bash
# Fast mode: FakeEmbeddings + PassthroughLLM (no API key needed)
venv\Scripts\python evaluation\run_evaluation.py

# Real embeddings + PassthroughLLM (downloads MiniLM model)
venv\Scripts\python evaluation\run_evaluation.py --real-embeddings

# Full quality: real embeddings + Groq LLM (requires GROQ_API_KEY in .env)
venv\Scripts\python evaluation\run_evaluation.py --real-embeddings --live-llm
```

Results are written to `evaluation/results/latest.json`.

### Evaluation dataset (`evaluation/questions.json`)

12 questions across 4 categories:

| Category | Questions | Description |
|---|---|---|
| `answerable_direct` | q01, q02, q03, q05, q12 | Single fact retrievable from one document |
| `answerable_paraphrase` | q04 | Same fact asked with different wording |
| `multi_evidence` | q06, q07 | Requires facts from two separate chunks |
| `unanswerable_absent` | q08, q09 | Benefit/fact that simply does not exist in documents |
| `unanswerable_similar` | q10, q11 | Sounds like something that exists but is not mentioned |

### Limitations

1. **PassthroughLLM quality**: The offline fallback concatenates raw excerpts rather than synthesizing a coherent answer. It passes structural checks but does not measure LLM phrasing quality.

2. **FakeEmbeddings retrieval**: Hash-based vectors produce higher cosine distances than real embeddings. The evaluation uses a relaxed `relevance_threshold=1.05` for the fake embedder. With real MiniLM embeddings and `threshold=0.55`, retrieval is significantly more precise.

3. **Chunk boundary effects**: Long answers that span two chunks may only be partially retrieved, causing `multi_evidence` questions to fail when the `chunk_size` is too large relative to the document.

4. **Prompt injection resistance**: The system prompt instructs the model to treat document content as untrusted, and the sample documents include explicit anti-injection text. This mitigates but does not eliminate the risk of adversarial documents.

5. **No re-ranking**: Retrieved chunks are ordered by cosine distance only. A cross-encoder re-ranker would improve precision, especially for longer documents.

6. **Single collection**: All documents share one ChromaDB collection. Multi-tenant or access-controlled knowledge bases would require per-user collections.

7. **Stateless scaling**: The current design is stateless at the API layer (state is in ChromaDB files). Horizontal scaling requires a shared filesystem or a remote ChromaDB server.
