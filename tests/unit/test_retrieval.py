from app.services.chunking import chunk_document
from app.services.document_loader import load_document
from app.services.embeddings import FakeEmbeddings
from app.services.retrieval import RetrievalService
from app.services.vector_store import VectorStore


def _store(tmp_path) -> VectorStore:
    return VectorStore(
        persist_directory=tmp_path / "chroma",
        collection_name="retrieval_test",
        embedding_model="fake-hash-embeddings",
    )


def _index(store: VectorStore, filename: str, text: str, embedder: FakeEmbeddings):
    extracted = load_document(filename, text.encode("utf-8"))
    chunks = chunk_document(extracted, chunk_size=400, chunk_overlap=50)
    store.add_chunks(chunks, embedder.embed_documents([chunk.text for chunk in chunks]))
    return chunks


def test_relevant_chunks_are_returned(tmp_path):
    embedder = FakeEmbeddings()
    store = _store(tmp_path)
    chunks = _index(store, "handbook.md", "Employees receive 18 days of annual leave each year.", embedder)
    retrieval = RetrievalService(embedder, store)
    results = retrieval.search("How many annual leave days do employees receive?", top_k=3, relevance_threshold=1.2)
    assert results
    assert any("18 days" in item.text for item in results)
    assert results[0].filename == "handbook.md"
    assert results[0].chunk_id == chunks[0].chunk_id
    assert results[0].distance >= 0


def test_empty_collection_returns_no_evidence(tmp_path):
    retrieval = RetrievalService(FakeEmbeddings(), _store(tmp_path))
    assert retrieval.search("anything", top_k=5, relevance_threshold=0.5) == []


def test_relevance_filter_drops_distant_chunks(tmp_path):
    embedder = FakeEmbeddings()
    store = _store(tmp_path)
    _index(store, "handbook.md", "Employees receive 18 days of annual leave.", embedder)
    retrieval = RetrievalService(embedder, store)
    strict = retrieval.search("annual leave days", top_k=5, relevance_threshold=0.0)
    relaxed = retrieval.search("annual leave days", top_k=5, relevance_threshold=1.5)
    assert strict == []
    assert relaxed


def test_duplicate_indexing_does_not_create_duplicate_chunks(tmp_path):
    embedder = FakeEmbeddings()
    store = _store(tmp_path)
    chunks = _index(store, "handbook.md", "Notice period is 30 days after probation.", embedder)
    store.add_chunks(chunks, embedder.embed_documents([chunk.text for chunk in chunks]))
    assert store.count() == len(chunks)


def test_embedding_model_mismatch(tmp_path):
    VectorStore(tmp_path / "chroma", "mismatch", "fake-hash-embeddings")
    from app.exceptions import EmbeddingModelMismatchError
    import pytest

    with pytest.raises(EmbeddingModelMismatchError):
        VectorStore(tmp_path / "chroma", "mismatch", "sentence-transformers/all-MiniLM-L6-v2")
