"""Run the evaluation dataset against the RAG services."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import Settings
from app.factory import create_app

SAMPLE_DIR = ROOT / "sample_documents"
QUESTIONS_PATH = ROOT / "evaluation" / "questions.json"
RESULTS_DIR = ROOT / "evaluation" / "results"


def normalize(text: str) -> str:
    compact = re.sub(r"[,\s]+", "", text.lower())
    return compact


def fact_present(answer: str, fact: str) -> bool:
    if fact.isdigit():
        collapsed = answer.replace(",", "")
        return re.search(rf"\b{re.escape(fact)}\b", collapsed) is not None
    return fact.lower() in answer.lower()


def load_questions() -> list[dict]:
    return json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))


def ingest_samples(app) -> None:
    ingestion = app.state.ingestion_service
    for path in sorted(SAMPLE_DIR.iterdir()):
        if path.suffix.lower() not in {".md", ".txt", ".pdf", ".markdown"}:
            continue
        ingestion.ingest(path.name, path.read_bytes())


def evaluate_item(app, item: dict) -> dict:
    qa = app.state.qa_service
    settings = app.state.settings
    result = qa.ask(
        item["question"],
        top_k=settings.top_k,
        relevance_threshold=settings.relevance_threshold,
    )
    source_docs = [source.document for source in result.sources]
    fact_hits = {fact: fact_present(result.answer, fact) for fact in item.get("required_facts", [])}

    # Retrieve for document-level check
    vectorstore = app.state.vectorstore
    retrieval = vectorstore.similarity_search_with_score(
        item["question"], k=settings.top_k
    )
    retrieved_docs = sorted({doc.metadata.get("filename", "") for doc, _ in retrieval})

    expected_found = item["expected_found"]
    found_ok = result.found_in_documents is expected_found
    facts_ok = all(fact_hits.values()) if expected_found else True
    sources_ok = True
    if expected_found:
        expected_docs = set(item.get("expected_documents") or [])
        sources_ok = expected_docs.issubset(set(source_docs) | set(retrieved_docs))
    else:
        sources_ok = result.sources == []

    citations_ok = all(source.chunk_id for source in result.sources)
    passed = found_ok and facts_ok and sources_ok and citations_ok

    return {
        "id": item["id"],
        "question": item["question"],
        "category": item["category"],
        "expected_found": expected_found,
        "actual_found": result.found_in_documents,
        "answer": result.answer,
        "sources": [source.model_dump() for source in result.sources],
        "retrieved_documents": retrieved_docs,
        "required_facts": fact_hits,
        "checks": {
            "found_in_documents": found_ok,
            "required_facts": facts_ok,
            "sources": sources_ok,
            "citations_map_to_chunks": citations_ok,
        },
        "passed": passed,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline.")
    parser.add_argument("--real-embeddings", action="store_true", help="Use MiniLM instead of fake embeddings.")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    persist = RESULTS_DIR / "eval_chroma"
    settings = Settings(
        chroma_persist_directory=persist,
        chroma_collection_name="evaluation",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2" if args.real_embeddings else "fake",
        groq_api_key=Settings().groq_api_key,
        chunk_size=800,
        chunk_overlap=120,
        top_k=5,
        relevance_threshold=0.70 if args.real_embeddings else 1.05,
    )

    # Build embeddings
    if args.real_embeddings:
        from langchain_huggingface import HuggingFaceEmbeddings
        embedder = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    else:
        from tests.conftest import FakeEmbeddings
        embedder = FakeEmbeddings()

    embedding_backend = getattr(embedder, "model_name", "unknown")

    if not settings.groq_api_key:
        print("ERROR: GROQ_API_KEY is not configured in .env. Please set GROQ_API_KEY to run the evaluation.")
        return 1

    # Build LLM
    from langchain_groq import ChatGroq
    from app.services.qa import LangChainLLM

    chat_model = ChatGroq(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.groq_api_key,
    )
    llm = LangChainLLM(chat_model)

    if persist.exists():
        import shutil
        shutil.rmtree(persist, ignore_errors=True)

    app = create_app(settings=settings, embedder=embedder, llm=llm)
    ingest_samples(app)

    rows = []
    questions = load_questions()
    for idx, item in enumerate(questions):
        if idx > 0:
            time.sleep(2.0)
        rows.append(evaluate_item(app, item))
    passed = sum(1 for row in rows if row["passed"])
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "llm_mode": "groq",
        "embedding_backend": embedding_backend,
        "relevance_threshold": settings.relevance_threshold,
        "chunk_size": settings.chunk_size,
        "total": len(rows),
        "passed": passed,
        "failed": len(rows) - passed,
        "results": rows,
        "scoring_notes": (
            "found_in_documents, required_facts, source filenames, and citation IDs are "
            "deterministic checks. Passthrough LLM copies retrieved excerpts and does not "
            "measure hosted-model phrasing quality."
        ),
    }
    output_path = RESULTS_DIR / "latest.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"LLM mode: groq")
    print(f"Embeddings: {embedding_backend}")
    print(f"Passed {passed}/{len(rows)}")
    for row in rows:
        mark = "PASS" if row["passed"] else "FAIL"
        print(f"  [{mark}] {row['id']} {row['category']}: {row['question']}")
        if not row["passed"]:
            print(f"         checks={row['checks']} facts={row['required_facts']}")
    print(f"Wrote {output_path}")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
