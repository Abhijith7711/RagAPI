import fitz

from app.services.ingestion import IngestionService


def _pdf(text: str) -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    payload = document.tobytes()
    document.close()
    return payload


def test_successful_upload_txt(client):
    response = client.post(
        "/documents",
        files={"file": ("company_policies.txt", b"Remote work is limited to 2 days each week.", "text/plain")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ingested"
    assert body["filename"] == "company_policies.txt"
    assert body["chunk_count"] >= 1
    assert body["is_duplicate"] is False
    assert body["document_id"]


def test_successful_upload_pdf_and_markdown(client):
    pdf_response = client.post(
        "/documents",
        files={"file": ("handbook.pdf", _pdf("Employees receive 18 days of annual leave."), "application/pdf")},
    )
    md_response = client.post(
        "/documents",
        files={"file": ("handbook.md", b"# Leave\n\nEmployees receive 10 days of sick leave.", "text/markdown")},
    )
    assert pdf_response.status_code == 200
    assert md_response.status_code == 200


def test_invalid_upload_extension(client):
    response = client.post("/documents", files={"file": ("notes.exe", b"binary", "application/octet-stream")})
    assert response.status_code == 400
    assert response.json()["error_code"] == "unsupported_file_type"


def test_empty_upload(client):
    response = client.post("/documents", files={"file": ("empty.txt", b"", "text/plain")})
    assert response.status_code == 400
    assert response.json()["error_code"] == "empty_document"


def test_malformed_pdf_upload(client):
    response = client.post(
        "/documents",
        files={"file": ("broken.pdf", b"%PDF-1.4 not a real pdf", "application/pdf")},
    )
    assert response.status_code == 422
    assert response.json()["error_code"] == "unreadable_document"


def test_duplicate_upload(client):
    payload = b"Notice period is 30 days after probation."
    first = client.post("/documents", files={"file": ("policy.txt", payload, "text/plain")})
    second = client.post("/documents", files={"file": ("policy-copy.txt", payload, "text/plain")})
    assert first.json()["status"] == "ingested"
    assert second.json()["status"] == "duplicate"
    assert second.json()["is_duplicate"] is True
    assert first.json()["document_id"] == second.json()["document_id"]
    store = client.app.state.vector_store
    assert store.count() == first.json()["chunk_count"]


def test_oversize_upload(client, settings):
    payload = b"x" * (settings.max_upload_bytes + 1)
    response = client.post("/documents", files={"file": ("huge.txt", payload, "text/plain")})
    assert response.status_code == 413
    assert response.json()["error_code"] == "payload_too_large"


def test_ingestion_service_direct(settings):
    from app.services.embeddings import FakeEmbeddings
    from app.services.vector_store import VectorStore

    store = VectorStore(settings.chroma_persist_directory / "nested", "ingest", "fake-hash-embeddings")
    service = IngestionService(FakeEmbeddings(), store, 400, 50)
    result = service.ingest("a.md", b"Visible text for indexing.")
    assert result.status == "ingested"
