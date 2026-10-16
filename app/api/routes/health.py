from fastapi import APIRouter, Request

from app.schemas import HealthResponse, ReadyResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@router.get("/ready", response_model=ReadyResponse)
def ready(request: Request) -> ReadyResponse:
    persist_dir = request.app.state.settings.chroma_persist_directory
    persist_ok = persist_dir.exists()
    vectorstore = request.app.state.vectorstore
    embed_fn = getattr(vectorstore, "_embedding_function", None)
    model_name = getattr(embed_fn, "model_name", "unknown")
    return ReadyResponse(
        status="ready" if persist_ok else "not_ready",
        chroma_persist_directory=persist_ok,
        embedding_backend=model_name,
    )
