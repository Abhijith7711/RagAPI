from app.schemas import LLMStructuredResponse


def test_health_and_ready(client):
    health = client.get("/health")
    ready = client.get("/ready")
    assert health.status_code == 200
    assert health.json() == {"status": "ok"}
    assert ready.status_code == 200
    assert ready.json()["status"] == "ready"
    assert ready.json()["embedding_backend"] == "fake-hash-embeddings"


def test_successful_question_request(client, scripted_llm):
    client.post(
        "/documents",
        files={"file": ("employee_handbook.md", b"Employees receive 18 days of annual leave.", "text/markdown")},
    )
    response = client.post("/ask", json={"question": "How many annual leave days do employees receive?"})
    assert response.status_code == 200
    body = response.json()
    assert body["question"].startswith("How many annual leave")
    assert body["found_in_documents"] is True
    assert body["sources"]
    assert body["sources"][0]["document"] == "employee_handbook.md"
    assert "18 days" in body["sources"][0]["excerpt"]
    assert scripted_llm.calls


def test_empty_question_validation(client):
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 422
    assert response.json()["error_code"] == "validation_error"


def test_question_against_empty_knowledge_base(client, scripted_llm):
    response = client.post("/ask", json={"question": "How many annual leave days do employees receive?"})
    assert response.status_code == 200
    body = response.json()
    assert body["found_in_documents"] is False
    assert body["sources"] == []
    assert "unavailable" in body["answer"].lower()
    assert scripted_llm.calls == []


def test_response_schema_fields(client):
    client.post("/documents", files={"file": ("a.txt", b"Core hours are 11:00 to 16:00 IST.", "text/plain")})
    body = client.post("/ask", json={"question": "What are the core hours?"}).json()
    assert set(body.keys()) == {"question", "answer", "found_in_documents", "sources"}
    if body["sources"]:
        assert set(body["sources"][0].keys()) >= {"document", "excerpt", "chunk_id"}
