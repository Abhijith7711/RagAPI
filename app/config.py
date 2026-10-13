"""Environment-driven application settings."""

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

    llm_provider: str = "groq"
    groq_api_key: str = ""
    llm_model: str = "llama-3.1-8b-instant"
    llm_temperature: float = 0.1
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chroma_persist_directory: Path = Path("./data/chroma")
    chroma_collection_name: str = "documents"
    chunk_size: int = Field(default=2000, ge=100)
    chunk_overlap: int = Field(default=300, ge=0)
    top_k: int = Field(default=5, ge=1, le=20)
    relevance_threshold: float = Field(default=0.55, ge=0.0, le=2.0)
    max_upload_size_mb: int = Field(default=10, ge=1)
    max_question_length: int = Field(default=2000, ge=1)
    log_level: str = "INFO"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
