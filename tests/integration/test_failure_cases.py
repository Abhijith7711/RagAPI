from app.exceptions import LLMUnavailableError
from app.schemas import LLMStructuredResponse


def test_llm_unavailable_returns_503(client, scripted_llm):
    client.post("/documents", files={"file": ("a.txt", b"Employees receive 18 days of annual leave.", "text/plain")})
    scripted_llm.error = LLMUnavailableError("provider down")
    response = client.post("/ask", json={"question": "How many annual leave days?"})
    assert response.status_code == 503
    assert response.json()["error_code"] == "llm_unavailable"
    assert "18" not in response.json()["detail"]


def test_malformed_llm_output_does_not_invent_answer(client, scripted_llm):
    client.post("/documents", files={"file": ("a.txt", b"Employees receive 18 days of annual leave.", "text/plain")})
    scripted_llm.error = ValueError("not json")
    response = client.post("/ask", json={"question": "How many annual leave days?"})
    assert response.status_code == 200
    body = response.json()
    assert body["found_in_documents"] is False
    assert body["sources"] == []


def test_fabricated_chunk_id_rejected(client, scripted_llm):
    client.post("/documents", files={"file": ("a.txt", b"Employees receive 18 days of annual leave.", "text/plain")})
    scripted_llm.response = LLMStructuredResponse(
        answer="Something invented",
        found_in_documents=True,
        supporting_chunk_ids=["fabricated-id"],
    )
    response = client.post("/ask", json={"question": "How many annual leave days?"})
    body = response.json()
    assert body["found_in_documents"] is False
    assert body["sources"] == []


def test_error_body_shape_is_consistent(client):
    response = client.post("/documents", files={"file": ("x.bin", b"nope", "application/octet-stream")})
    assert set(response.json().keys()) == {"detail", "error_code"}
