from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.routes import documents, health, questions, ui
from app.config import Settings, get_settings
from app.exceptions import AppError
from app.schemas import ErrorResponse
from app.services.ingestion import IngestionService
from app.services.qa import LLMClient, QAService

if TYPE_CHECKING:
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore

logger = logging.getLogger(__name__)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def build_vectorstore(settings: Settings, embedding_function: Embeddings | None = None) -> VectorStore:
    """Create a persistent LangChain Chroma vector store with cosine similarity."""
    from langchain_chroma import Chroma

    if embedding_function is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        embedding_function = HuggingFaceEmbeddings(model_name=settings.embedding_model)

    persist_dir = settings.chroma_persist_directory
    persist_dir.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=embedding_function,
        persist_directory=str(persist_dir),
        collection_metadata={"hnsw:space": "cosine"},
    )


def build_llm(settings: Settings, llm: LLMClient | None = None) -> LLMClient:
    """Return a LangChain LLM adapter, or the injected test double."""
    if llm is not None:
        return llm  # type: ignore[return-value]  # test double satisfies protocol

    from langchain_groq import ChatGroq
    from app.services.qa import LangChainLLM

    if not settings.groq_api_key:
        from app.exceptions import LLMUnavailableError
        raise LLMUnavailableError(
            "GROQ_API_KEY is not configured. Please set your GROQ_API_KEY in the .env file."
        )

    chat_model = ChatGroq(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        api_key=settings.groq_api_key,
    )
    return LangChainLLM(chat_model)


def create_app(
    settings: Settings | None = None,
    embedder: Embeddings | None = None,
    llm: LLMClient | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    vectorstore = build_vectorstore(settings, embedding_function=embedder)
    llm_client = build_llm(settings, llm)

    ingestion = IngestionService(
        vectorstore=vectorstore,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    qa = QAService(vectorstore=vectorstore, llm=llm_client)

    app = FastAPI(
        title="RAG API",
        description="Document-grounded question answering with evidence and abstention.",
        version="1.0.0",
    )
    app.state.settings = settings
    app.state.vectorstore = vectorstore
    app.state.ingestion_service = ingestion
    app.state.qa_service = qa

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(questions.router)
    app.include_router(ui.router)
    app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

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

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(detail=str(exc.detail), error_code="http_error").model_dump(),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled server error: %s", exc)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                detail="An unexpected internal error occurred.",
                error_code="internal_error",
            ).model_dump(),
        )

    return app