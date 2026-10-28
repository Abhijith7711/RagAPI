"""Environment-driven application settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys & LLM Configuration
    groq_api_key: str = ""
    llm_model: str = "openai/gpt-oss-120b"
    llm_temperature: float = 0.1

    # Embeddings & Vector Store
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_persist_directory: Path = Path("./data/chroma")
    chroma_collection_name: str = "documents"

    # Chunking & Retrieval Parameters
    chunk_size: int = Field(default=1000, ge=100)
    chunk_overlap: int = Field(default=150, ge=0)
    top_k: int = Field(default=5, ge=1, le=20)
    relevance_threshold: float = Field(default=0.20, ge=0.0, le=1.0)
    max_upload_size_mb: int = Field(default=10, ge=1)
    max_question_length: int = Field(default=2000, ge=1)
    log_level: str = "INFO"
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
