"""Request, response, and internal data models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Source(BaseModel):
    document: str
    excerpt: str
    chunk_id: str
    page: int | None = None


class IngestResponse(BaseModel):
    status: Literal["ingested", "duplicate"]
    filename: str
    document_id: str
    chunk_count: int
    is_duplicate: bool


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    relevance_threshold: float | None = Field(default=None, ge=0.0, le=1.0)

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Question must not be empty.")
        return cleaned


class AskResponse(BaseModel):
    question: str
    answer: str
    found_in_documents: bool
    sources: list[Source] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"


class ReadyResponse(BaseModel):
    status: Literal["ready", "not_ready"]
    chroma_persist_directory: bool
    embedding_backend: str


class ErrorResponse(BaseModel):
    detail: str
    error_code: str


class LLMStructuredResponse(BaseModel):
    """Schema for structured LLM output — used by ChatGroq.with_structured_output()."""
    answer: str
    found_in_documents: bool
    supporting_chunk_ids: list[str] = Field(default_factory=list)
