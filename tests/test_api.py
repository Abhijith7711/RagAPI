"""API integration tests covering ingestion, question answering, and UI endpoints."""

from __future__ import annotations

import fitz

from app.exceptions import LLMUnavailableError
from app.schemas import LLMStructuredResponse


def _create_sample_pdf(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    payload = doc.tobytes()
    doc.close()
    return payload


def test_health_and_ready(client):
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}

    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"


def test_upload_txt_and_markdown(client):
    txt_resp = client.post(
        "/documents",
        files={"file": ("policies.txt", b"Remote work is limited to 2 days per week.", "text/plain")},
    )
    assert txt_resp.status_code == 200
    assert txt_resp.json()["status"] == "ingested"
    assert txt_resp.json()["filename"] == "policies.txt"
    assert txt_resp.json()["chunk_count"] >= 1

    md_resp = client.post(
        "/documents",
        files={"file": ("guide.md", b"# Handbook\n\nAnnual leave is 18 days.", "text/markdown")},
    )
    assert md_resp.status_code == 200
    assert md_resp.json()["status"] == "ingested"


def test_upload_pdf(client):
    pdf_bytes = _create_sample_pdf("IT Security: Passwords must be at least 14 characters.")
    response = client.post(
        "/documents",
        files={"file": ("security.pdf", pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "ingested"
    assert response.json()["filename"] == "security.pdf"


def test_upload_duplicate_detection(client):
    content = b"Probation period for new hires is exactly 90 days."
    first = client.post("/documents", files={"file": ("policy.txt", content, "text/plain")})
    second = client.post("/documents", files={"file": ("policy_copy.txt", content, "text/plain")})

    assert first.json()["status"] == "ingested"
    assert second.json()["status"] == "duplicate"
    assert second.json()["is_duplicate"] is True
    assert first.json()["document_id"] == second.json()["document_id"]


def test_upload_validation_errors(client):
    # Invalid extension
    bad_ext = client.post("/documents", files={"file": ("script.sh", b"echo hi", "text/plain")})
    assert bad_ext.status_code == 400
    assert bad_ext.json()["error_code"] == "unsupported_file_type"

    # Empty file
    empty = client.post("/documents", files={"file": ("empty.txt", b"", "text/plain")})
    assert empty.status_code == 400
    assert empty.json()["error_code"] == "empty_document"

    # Malformed PDF
    broken_pdf = client.post(
        "/documents",
        files={"file": ("broken.pdf", b"%PDF-1.4 invalid binary", "application/pdf")},
    )
    assert broken_pdf.status_code == 422
    assert broken_pdf.json()["error_code"] == "unreadable_document"


def test_ask_question_grounded(client, scripted_llm):
    client.post(
        "/documents",
        files={"file": ("handbook.md", b"Employees receive 18 days of annual leave.", "text/markdown")},
    )
    response = client.post("/ask", json={"question": "How many days of annual leave?"})
    assert response.status_code == 200
    body = response.json()
    assert body["found_in_documents"] is True
    assert len(body["sources"]) >= 1
    assert body["sources"][0]["document"] == "handbook.md"
    assert "18 days" in body["sources"][0]["excerpt"]
    assert body["sources"][0]["chunk_id"]


def test_ask_question_empty_knowledge_base(client):
    response = client.post("/ask", json={"question": "What is the dental plan?"})
    assert response.status_code == 200
    body = response.json()
    assert body["found_in_documents"] is False
    assert body["sources"] == []
    assert "unavailable" in body["answer"].lower()


def test_ask_question_validation(client):
    empty_q = client.post("/ask", json={"question": "   "})
    assert empty_q.status_code == 422
    assert empty_q.json()["error_code"] == "validation_error"


def test_ask_rejects_fabricated_citations(client, scripted_llm):
    client.post("/documents", files={"file": ("doc.txt", b"Valid policy text here.", "text/plain")})
    scripted_llm.response = LLMStructuredResponse(
        answer="Fabricated response.",
        found_in_documents=True,
        supporting_chunk_ids=["non-existent-chunk-id"],
    )
    response = client.post("/ask", json={"question": "What is the policy?"})
    body = response.json()
    assert body["found_in_documents"] is False
    assert body["sources"] == []
    assert "unavailable" in body["answer"].lower()


def test_ask_llm_outage_returns_503(client, scripted_llm):
    client.post("/documents", files={"file": ("doc.txt", b"Valid policy text here.", "text/plain")})
    scripted_llm.error = LLMUnavailableError("Groq service unreachable")
    response = client.post("/ask", json={"question": "What is the policy?"})
    assert response.status_code == 503
    assert response.json()["error_code"] == "llm_unavailable"


def test_ui_endpoints(client):
    # Main page HTML
    page = client.get("/")
    assert page.status_code == 200
    assert "text/html" in page.headers["content-type"]
    assert "RAG Knowledge Assistant" in page.text

    # Static CSS and JS
    css = client.get("/static/styles.css")
    assert css.status_code == 200
    assert "text/css" in css.headers["content-type"]

    js = client.get("/static/app.js")
    assert js.status_code == 200
