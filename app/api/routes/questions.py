from fastapi import APIRouter, Depends

from app.config import Settings
from app.dependencies import get_qa_service, get_settings
from app.schemas import AskRequest, AskResponse
from app.services.qa import QAService

router = APIRouter(tags=["questions"])


@router.post("/ask", response_model=AskResponse)
def ask_question(
    payload: AskRequest,
    settings: Settings = Depends(get_settings),
    qa: QAService = Depends(get_qa_service),
) -> AskResponse:
    return qa.ask(
        payload.question,
        top_k=payload.top_k or settings.top_k,
        relevance_threshold=(
            payload.relevance_threshold
            if payload.relevance_threshold is not None
            else settings.relevance_threshold
        ),
    )
