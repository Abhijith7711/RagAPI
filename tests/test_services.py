"""Unit tests covering chunking, document loading, retrieval, and QA grounding."""

from __future__ import annotations

import fitz
import pytest

from app.exceptions import EmptyDocumentError, UnsupportedFileTypeError
from app.schemas import ExtractedDocument, ExtractedPage, LLMStructuredResponse, RetrievedChunk
from app.services.chunking import chunk_document, content_hash
from app.services.document_loader import load_document
from app.services.embeddings import FakeEmbeddings
from app.services.llm import parse_llm_json
from app.services.qa import ABSTENTION, QAService
from app.services.retrieval import RetrievalService
from app.services.vector_store import VectorStore
from tests.conftest import ScriptedLLM


def _make_chunk(chunk_id: str, text: str, page_number: int | None = 1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc-123",
        filename="handbook.md",
        chunk_index=0,
        text=text,
        page_number=page_number,
        document_hash="hash-123",
        distance=0.15,
    )


def test_chunking_sliding_window_and_metadata():
    text = "Paragraph one with some text.\n\n" + ("Sentence in second paragraph. " * 30)
    document = ExtractedDocument(
        filename="guide.md",
        text=text,
        pages=[ExtractedPage(page_number=1, text=text)],
    )
    chunks = chunk_document(document, chunk_size=200, chunk_overlap=40)
    assert len(chunks) >= 2
    assert chunks[0].filename == "guide.md"
    assert chunks[0].chunk_index == 0
    assert chunks[0].document_id == content_hash(text)
    assert chunks[0].chunk_id == f"{chunks[0].document_id}:0"


def test_chunking_pdf_page_mapping():
    page1 = "Page one content. " * 20
    page2 = "Page two content. " * 20
    document = ExtractedDocument(
        filename="policy.pdf",
        text=f"{page1}\n\n{page2}",
        pages=[
            ExtractedPage(page_number=1, text=page1.strip()),
            ExtractedPage(page_number=2, text=page2.strip()),
        ],
    )
    chunks = chunk_document(document, chunk_size=150, chunk_overlap=20)
    pages = {chunk.page_number for chunk in chunks}
    assert 1 in pages
    assert 2 in pages


def test_document_loader_formats():
    # Markdown
    md_doc = load_document("handbook.md", b"# Leave\n\n18 days annual leave.")
    assert md_doc.filename == "handbook.md"
    assert "18 days" in md_doc.text

    # Plain text
    txt_doc = load_document("terms.txt", b"Notice period is 30 days.")
    assert txt_doc.filename == "terms.txt"
    assert "30 days" in txt_doc.text

    # PDF with PyMuPDF
    fitz_doc = fitz.open()
    fitz_page = fitz_doc.new_page()
    fitz_page.insert_text((72, 72), "IT Security Policy Page 1")
    pdf_bytes = fitz_doc.tobytes()
    fitz_doc.close()

    pdf_doc = load_document("security.pdf", pdf_bytes)
    assert pdf_doc.filename == "security.pdf"
    assert "IT Security Policy Page 1" in pdf_doc.text
    assert pdf_doc.pages[0].page_number == 1


def test_document_loader_validations():
    with pytest.raises(UnsupportedFileTypeError):
        load_document("virus.exe", b"binary content")

    with pytest.raises(EmptyDocumentError):
        load_document("empty.txt", b"")

    # Path traversal characters stripped to basename
    traversal = load_document("../../etc/passwd.txt", b"safe content")
    assert traversal.filename == "passwd.txt"


def test_parse_llm_json_plain_and_fenced():
    # Plain JSON
    plain = '{"answer": "18 days", "found_in_documents": true, "supporting_chunk_ids": ["c:0"]}'
    res1 = parse_llm_json(plain)
    assert res1.answer == "18 days"
    assert res1.found_in_documents is True
    assert res1.supporting_chunk_ids == ["c:0"]

    # Fenced markdown codeblock JSON
    fenced = """Here is the response:
```json
{"answer": "Unavailable", "found_in_documents": false, "supporting_chunk_ids": []}
```
"""
    res2 = parse_llm_json(fenced)
    assert res2.found_in_documents is False
    assert res2.supporting_chunk_ids == []

    # Invalid JSON raises ValueError
    with pytest.raises(ValueError):
        parse_llm_json("Not a json at all")


def test_retrieval_and_relevance_threshold(tmp_path):
    embedder = FakeEmbeddings()
    store = VectorStore(tmp_path / "chroma", "retrieval_test", "fake-hash-embeddings")
    doc = load_document("policy.txt", b"Employees receive 18 days of annual leave each year.")
    chunks = chunk_document(doc, chunk_size=400, chunk_overlap=50)
    store.add_chunks(chunks, embedder.embed_documents([c.text for c in chunks]))

    retrieval = RetrievalService(embedder, store)

    # Permissive threshold retrieves the chunk
    matches = retrieval.search("How many leave days?", top_k=3, relevance_threshold=1.5)
    assert len(matches) >= 1
    assert "18 days" in matches[0].text

    # Strict zero threshold filters it out
    filtered = retrieval.search("How many leave days?", top_k=3, relevance_threshold=0.0)
    assert len(filtered) == 0


def test_qa_service_grounding_and_abstention():
    evidence = [_make_chunk("chunk:0", "Annual leave is 18 days.")]
    llm = ScriptedLLM(
        LLMStructuredResponse(
            answer="Employees receive 18 days of leave.",
            found_in_documents=True,
            supporting_chunk_ids=["chunk:0"],
        )
    )

    class StubRetrieval:
        def __init__(self, items):
            self.items = items
        def search(self, *args, **kwargs):
            return self.items

    # Successful grounded answer
    qa = QAService(StubRetrieval(evidence), llm)
    res = qa.ask("How much leave?", top_k=5, relevance_threshold=0.6)
    assert res.found_in_documents is True
    assert len(res.sources) == 1
    assert res.sources[0].chunk_id == "chunk:0"
    assert "18 days" in res.sources[0].excerpt

    # Abstention when no evidence passes retrieval filter
    empty_qa = QAService(StubRetrieval([]), llm)
    res_empty = empty_qa.ask("What is the 401k plan?", top_k=5, relevance_threshold=0.6)
    assert res_empty.found_in_documents is False
    assert res_empty.sources == []
    assert res_empty.answer == ABSTENTION


def test_qa_service_multi_evidence():
    evidence = [
        _make_chunk("c:0", "Annual leave is 18 days."),
        _make_chunk("c:1", "Sick leave is 10 days."),
    ]
    llm = ScriptedLLM(
        LLMStructuredResponse(
            answer="Annual leave is 18 days and sick leave is 10 days.",
            found_in_documents=True,
            supporting_chunk_ids=["c:0", "c:1"],
        )
    )

    class StubRetrieval:
        def search(self, *args, **kwargs):
            return evidence

    qa = QAService(StubRetrieval(), llm)
    res = qa.ask("What are leave allowances?", top_k=5, relevance_threshold=0.6)
    assert res.found_in_documents is True
    assert len(res.sources) == 2
    cited_ids = {s.chunk_id for s in res.sources}
    assert cited_ids == {"c:0", "c:1"}
