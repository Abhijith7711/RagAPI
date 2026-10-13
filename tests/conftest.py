from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.factory import create_app
from app.schemas import LLMStructuredResponse, RetrievedChunk
from app.services.embeddings import FakeEmbeddings


class ScriptedLLM:
    def __init__(self, response=None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls: list[tuple[str, list[RetrievedChunk]]] = []

    def complete(self, question: str, evidence: list[RetrievedChunk]) -> LLMStructuredResponse:
        self.calls.append((question, evidence))
        if self.error is not None:
            raise self.error
        if self.response is not None:
            return self.response
        if not evidence:
            return LLMStructuredResponse(
                answer="The information is unavailable in the provided documents.",
                found_in_documents=False,
                supporting_chunk_ids=[],
            )
        return LLMStructuredResponse(
            answer=evidence[0].text,
            found_in_documents=True,
            supporting_chunk_ids=[evidence[0].chunk_id],
        )


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        chroma_persist_directory=tmp_path / "chroma",
        chroma_collection_name="test_documents",
        embedding_model="fake",
        groq_api_key="",
        llm_provider="passthrough",
        chunk_size=400,
        chunk_overlap=50,
        top_k=5,
        # FakeEmbeddings uses hash-based bag-of-words vectors whose cosine
        # distances are much higher than real embeddings. Use a permissive
        # threshold (same as evaluation/run_evaluation.py) so chunks actually
        # reach the QA service in tests.
        relevance_threshold=1.05,
        max_upload_size_mb=1,
    )


@pytest.fixture
def scripted_llm() -> ScriptedLLM:
    return ScriptedLLM()


@pytest.fixture
def app(settings, scripted_llm):
    return create_app(settings=settings, embedder=FakeEmbeddings(), llm=scripted_llm)


@pytest.fixture
def client(app) -> TestClient:
    return TestClient(app)
