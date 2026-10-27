# RAG Knowledge Assistant

A production-grade, document-grounded Retrieval-Augmented Generation (RAG) API built with FastAPI, LangChain, ChromaDB, HuggingFace embeddings, and Groq LLM.

Answers questions strictly from uploaded documents, provides exact source citations with excerpts, and **reliably abstains** whenever information is absent (strongly mitigating hallucinations via citation validation).

---

## Features 
- **Industry-Standard Architecture:** Built with modern LangChain ecosystem (`langchain-core`, `langchain-chroma`, `langchain-groq`, `langchain-text-splitters`, `langchain-huggingface`).
- **Document Ingestion:** Ingests PDF (with per-page metadata), TXT, and Markdown files with SHA-256 deduplication.
- **Smart Chunking:** Utilizes LangChain's `RecursiveCharacterTextSplitter` with configurable chunk sizes, overlap, and paragraph/sentence boundary preservation.
- **Semantic Retrieval:** Dense embeddings (`sentence-transformers/all-MiniLM-L6-v2`) and persistent cosine vector search with `langchain-chroma`.
- **Grounded LLM Answering:** Integrates with Groq via `ChatGroq` using Pydantic structured outputs (`with_structured_output(LLMStructuredResponse)`).
- **Evidence & Citations:** Every answer cites source document names, page numbers (for PDFs), chunk IDs, and text excerpts.
- **Reliable Abstention:** Two-stage abstention filters out unanswerable queries before or after LLM generation.
- **FastAPI Endpoints:** Clean endpoints for ingestion (`/documents`), asking questions (`/ask`), health checks (`/health`), and readiness (`/ready`).
- **Interactive Web UI:** Minimal, elegant web interface at `http://localhost:8000/` for dragging & dropping documents and chatting live.
- **14-Question Benchmark:** Automated evaluation testing answerable, unanswerable, and multi-evidence questions.

---

## Setup

### 1. Prerequisites
- Python 3.11+
- A free [Groq API key](https://console.groq.com/)

### 2. Install Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure API Key
Copy the example environment file and insert your Groq API key:
```bash
cp .env.example .env
```
Edit `.env`:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```

---

## Running the Application

Start the FastAPI application:
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

| URL | Description |
|---|---|
| `http://localhost:8000/` | Interactive Web Demonstration UI |
| `http://localhost:8000/docs` | Swagger / OpenAPI Documentation |
| `http://localhost:8000/health` | Health Check Endpoint |

---

## API Usage Examples

### 1. Ingest a Document (`POST /documents`)
Upload PDF, TXT, or Markdown:
```bash
curl -X POST http://localhost:8000/documents \
  -F "file=@sample_documents/company_policies.txt"
```
Response:
```json
{
  "status": "ingested",
  "filename": "company_policies.txt",
  "document_id": "8959214135...",
  "chunk_count": 2,
  "is_duplicate": false
}
```

### 2. Ask a Question (`POST /ask`)
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many days per week are employees allowed to work remotely?"}'
```
Response (Answerable):
```json
{
  "question": "How many days per week are employees allowed to work remotely?",
  "answer": "Employees are permitted to work remotely up to two days per week with manager approval.",
  "found_in_documents": true,
  "sources": [
    {
      "document": "company_policies.txt",
      "excerpt": "Remote work is permitted up to two days per week with prior manager approval...",
      "chunk_id": "8959214135...:0",
      "page": null
    }
  ]
}
```

Response (Unanswerable / Abstention):
```json
{
  "question": "What is the 401(k) company match percentage?",
  "answer": "The information is unavailable in the provided documents.",
  "found_in_documents": false,
  "sources": []
}
```

---

## Automated Tests & Evaluation

### Run Test Suite
Runs unit and integration tests with fast, offline mocks (under 2 seconds):
```bash
pytest
```

### Run 14-Question Evaluation Benchmark
Evaluates the full RAG pipeline against 14 questions (answerable, unanswerable, and multi-evidence):
```bash
python evaluation/run_evaluation.py --real-embeddings
```
Results are saved to `evaluation/results/latest.json`.

---

## Docker (Optional)

Run the application in a Docker container:
```bash
docker-compose up --build
```

---

## Limitations

- **No Authentication & Database:** No user accounts, role-based access, or relational database (e.g. PostgreSQL) is connected.
- **Stateless (No Chat History):** Questions are answered as independent single-turn requests; multi-turn conversational context is not stored.
- **No SSE Streaming:** Answers are returned as a complete JSON response once generation finishes, rather than streamed token-by-token via Server-Sent Events (SSE).
- **Text-Only Documents:** Works with text-based PDF, TXT, and Markdown files. Scanned documents and images are not supported (no OCR).
- **Semantic Search Only:** Uses dense vector embeddings without hybrid keyword search (BM25), which can occasionally miss exact code numbers or specific IDs.
