"""Unit tests covering document loading, chunking via LangChain, retrieval, and QA grounding."""

from __future__ import annotations

import fitz
import pytest

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.exceptions import EmptyDocumentError, UnsupportedFileTypeError
from app.schemas import LLMStructuredResponse
from app.services.document_loader import content_hash, load_document
from app.services.qa import ABSTENTION, QAService
from tests.conftest import FakeEmbeddings, ScriptedLLM


def _make_evidence(chunk_id: str, text: str, page_number: int | None = 1) -> dict:
    return {
        "chunk_id": chunk_id,
        "filename": "handbook.md",
        "page_number": page_number,
        "text": text,
    }


def test_document_loader_returns_langchain_documents():
    docs = load_document("handbook.md", b"# Leave\n\n18 days annual leave.")
    assert len(docs) >= 1
    assert docs[0].metadata["filename"] == "handbook.md"
    assert "18 days" in docs[0].page_content
    assert "document_hash" in docs[0].metadata


def test_langchain_text_splitter_produces_chunks():
    """RecursiveCharacterTextSplitter creates overlapping chunks with metadata."""
    text = "Paragraph one with some text.\n\n" + ("Sentence in second paragraph. " * 30)
    docs = load_document("guide.md", text.encode())
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
    chunks = splitter.split_documents(docs)
    assert len(chunks) >= 2
    assert chunks[0].metadata["filename"] == "guide.md"
    assert "document_hash" in chunks[0].metadata


def test_chunking_pdf_page_mapping():
    page1 = "Page one content. " * 20
    page2 = "Page two content. " * 20
    doc = fitz.open()
    p1 = doc.new_page()
    p1.insert_text((72, 72), page1)
    p2 = doc.new_page()
    p2.insert_text((72, 72), page2)
    pdf_bytes = doc.tobytes()
    doc.close()

    docs = load_document("policy.pdf", pdf_bytes)
    pages = {d.metadata["page_number"] for d in docs}
    assert 1 in pages
    assert 2 in pages


def test_document_loader_formats():
    # Markdown
    md_doc = load_document("handbook.md", b"# Leave\n\n18 days annual leave.")
    assert md_doc[0].metadata["filename"] == "handbook.md"
    assert "18 days" in md_doc[0].page_content

    # Plain text
    txt_doc = load_document("terms.txt", b"Notice period is 30 days.")
    assert txt_doc[0].metadata["filename"] == "terms.txt"
    assert "30 days" in txt_doc[0].page_content

    # PDF with PyMuPDF
    fitz_doc = fitz.open()
    fitz_page = fitz_doc.new_page()
    fitz_page.insert_text((72, 72), "IT Security Policy Page 1")
    pdf_bytes = fitz_doc.tobytes()
    fitz_doc.close()

    pdf_doc = load_document("security.pdf", pdf_bytes)
    assert pdf_doc[0].metadata["filename"] == "security.pdf"
    assert "IT Security Policy Page 1" in pdf_doc[0].page_content
    assert pdf_doc[0].metadata["page_number"] == 1


def test_document_loader_validations():
    with pytest.raises(UnsupportedFileTypeError):
        load_document("virus.exe", b"binary content")

    with pytest.raises(EmptyDocumentError):
        load_document("empty.txt", b"")

    # Path traversal characters stripped to basename
    traversal = load_document("../../etc/passwd.txt", b"safe content")
    assert traversal[0].metadata["filename"] == "passwd.txt"


def test_content_hash_deduplication():
    hash1 = content_hash("same content")
    hash2 = content_hash("same content")
    hash3 = content_hash("different content")
    assert hash1 == hash2
    assert hash1 != hash3


def test_retrieval_and_relevance_threshold(tmp_path):
    embedder = FakeEmbeddings()
    vectorstore = Chroma(
        collection_name="retrieval_test",
        embedding_function=embedder,
        persist_directory=str(tmp_path / "chroma"),
        collection_metadata={"hnsw:space": "cosine"},
    )
    docs = load_document("policy.txt", b"Employees receive 18 days of annual leave each year.")
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    for i, c in enumerate(chunks):
        c.metadata["chunk_index"] = i
        c.metadata["document_id"] = c.metadata["document_hash"]
        c.metadata["chunk_id"] = f"{c.metadata['document_hash']}:{i}"
        c.id = f"{c.metadata['document_hash']}:{i}"
    vectorstore.add_documents(chunks)

    # Permissive threshold retrieves the chunk
    results = vectorstore.similarity_search_with_score("How many leave days?", k=3)
    matches = [(doc, score) for doc, score in results if score <= 1.5]
    assert len(matches) >= 1
    assert "18 days" in matches[0][0].page_content

    # Strict zero threshold filters it out
    filtered = [(doc, score) for doc, score in results if score <= 0.0]
    assert len(filtered) == 0


def test_qa_service_grounding_and_abstention():
    evidence = [_make_evidence("chunk:0", "Annual leave is 18 days.")]
    llm = ScriptedLLM(
        LLMStructuredResponse(
            answer="Employees receive 18 days of leave.",
            found_in_documents=True,
            supporting_chunk_ids=["chunk:0"],
        )
    )

    class StubVectorStore:
        _collection = type("C", (), {"count": lambda self: 1})()
        def similarity_search_with_score(self, query, k=5):
            from langchain_core.documents import Document
            return [(Document(page_content=e["text"], metadata={
                "chunk_id": e["chunk_id"],
                "filename": e["filename"],
                "page_number": e["page_number"],
                "document_id": "doc-123",
                "document_hash": "hash-123",
                "chunk_index": idx,
            }), 0.15) for idx, e in enumerate(evidence)]

    # Successful grounded answer
    qa = QAService(StubVectorStore(), llm)
    res = qa.ask("How much leave?", top_k=5, relevance_threshold=0.6)
    assert res.found_in_documents is True
    assert len(res.sources) == 1
    assert "chunk:0" in res.sources[0].chunk_id or res.sources[0].chunk_id
    assert "18 days" in res.sources[0].excerpt

    # Abstention when no evidence passes retrieval filter
    class EmptyVectorStore:
        _collection = type("C", (), {"count": lambda self: 0})()
        def similarity_search_with_score(self, query, k=5):
            return []

    empty_qa = QAService(EmptyVectorStore(), llm)
    res_empty = empty_qa.ask("What is the 401k plan?", top_k=5, relevance_threshold=0.6)
    assert res_empty.found_in_documents is False
    assert res_empty.sources == []
    assert res_empty.answer == ABSTENTION


def test_qa_service_multi_evidence():
    evidence = [
        _make_evidence("c:0", "Annual leave is 18 days."),
        _make_evidence("c:1", "Sick leave is 10 days."),
    ]
    llm = ScriptedLLM(
        LLMStructuredResponse(
            answer="Annual leave is 18 days and sick leave is 10 days.",
            found_in_documents=True,
            supporting_chunk_ids=["c:0", "c:1"],
        )
    )

    class StubVectorStore:
        _collection = type("C", (), {"count": lambda self: 2})()
        def similarity_search_with_score(self, query, k=5):
            from langchain_core.documents import Document
            return [(Document(page_content=e["text"], metadata={
                "chunk_id": e["chunk_id"],
                "filename": e["filename"],
                "page_number": e["page_number"],
                "document_id": "doc-123",
                "document_hash": "hash-123",
                "chunk_index": idx,
            }), 0.15) for idx, e in enumerate(evidence)]

    qa = QAService(StubVectorStore(), llm)
    res = qa.ask("What are leave allowances?", top_k=5, relevance_threshold=0.6)
    assert res.found_in_documents is True
    assert len(res.sources) == 2
