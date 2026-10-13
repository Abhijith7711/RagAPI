"""FastAPI application factory."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.routes import documents, health, questions
from app.config import Settings, get_settings
from app.exceptions import AppError
from app.schemas import ErrorResponse
from app.services.embeddings import FakeEmbeddings, SentenceTransformerEmbeddings
from app.services.ingestion import IngestionService
from app.services.llm import GroqLLM, LLMClient, PassthroughLLM
from app.services.qa import QAService
from app.services.retrieval import RetrievalService
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def build_embedder(settings: Settings, embedder=None):
    if embedder is not None:
        return embedder
    if settings.embedding_model in {"fake", "fake-hash-embeddings"}:
        return FakeEmbeddings()
    return SentenceTransformerEmbeddings(settings.embedding_model)


def build_llm(settings: Settings, llm: LLMClient | None = None) -> LLMClient:
    if llm is not None:
        return llm
    if settings.llm_provider == "passthrough" or not settings.groq_api_key:
        logger.warning("Using passthrough LLM; configure GROQ_API_KEY for hosted generation.")
        return PassthroughLLM()
    return GroqLLM(
        api_key=settings.groq_api_key,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
    )


def create_app(
    settings: Settings | None = None,
    embedder=None,
    llm: LLMClient | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    embedder = build_embedder(settings, embedder)
    vector_store = VectorStore(
        persist_directory=settings.chroma_persist_directory,
        collection_name=settings.chroma_collection_name,
        embedding_model=embedder.model_name,
    )
    llm_client = build_llm(settings, llm)
    ingestion = IngestionService(
        embedder=embedder,
        vector_store=vector_store,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    qa = QAService(retrieval=RetrievalService(embedder, vector_store), llm=llm_client)

    app = FastAPI(
        title="Reliable RAG API",
        description="Document-grounded question answering with evidence and abstention.",
        version="1.0.0",
    )
    app.state.settings = settings
    app.state.embedder = embedder
    app.state.vector_store = vector_store
    app.state.ingestion_service = ingestion
    app.state.qa_service = qa

    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(questions.router)

    @app.exception_handler(AppError)
    async def handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(detail=exc.detail, error_code=exc.error_code).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        messages = []
        for error in exc.errors():
            location = ".".join(str(part) for part in error.get("loc", []) if part != "body")
            messages.append(f"{location}: {error.get('msg')}" if location else str(error.get("msg")))
        detail = messages[0] if messages else "Request validation failed."
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(detail=detail, error_code="validation_error").model_dump(),
        )

    return app
