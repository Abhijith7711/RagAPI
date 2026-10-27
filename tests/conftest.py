from __future__ import annotations

import hashlib
import math

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings

from app.config import Settings
from app.factory import create_app
from app.schemas import LLMStructuredResponse


class FakeEmbeddings(Embeddings):
    """Deterministic, dependency-free embeddings for unit tests.

    Tokens are hashed into a fixed-size bag-of-words vector so overlapping
    wording produces similar vectors without downloading a model.
    """
    model_name: str = "fake-hash-embeddings"

    def __init__(self, dimensions: int = 32):
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in text.lower().split():
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:2], "little") % self.dimensions
            vector[index] += 1.0
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
        return vector


class ScriptedLLM:
    """Test double that mimics the LLMClient protocol used by QAService."""

    def __init__(self, response=None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls: list[tuple[str, list[dict]]] = []

    def invoke(self, question: str, evidence: list[dict]) -> LLMStructuredResponse:
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
            answer=evidence[0]["text"],
            found_in_documents=True,
            supporting_chunk_ids=[evidence[0]["chunk_id"]],
        )


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        chroma_persist_directory=tmp_path / "chroma",
        chroma_collection_name="test_documents",
        embedding_model="fake",
        groq_api_key="",
        chunk_size=400,
        chunk_overlap=50,
        top_k=5,
        # FakeEmbeddings uses hash-based bag-of-words vectors. With the
        # similarity_search_with_relevance_scores API (scores in [0, 1],
        # higher = more relevant), use a permissive floor of 0.0 so all
        # retrieved chunks reach the QA service during tests.
        relevance_threshold=0.0,
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
