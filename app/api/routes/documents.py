"""Document ingestion endpoint — validates, loads, and indexes uploaded files."""

from fastapi import APIRouter, Depends, File, UploadFile

from app.config import Settings
from app.dependencies import get_ingestion_service, get_settings
from app.exceptions import PayloadTooLargeError
from app.schemas import IngestResponse
from app.services.document_loader import validate_filename
from app.services.ingestion import IngestionService

router = APIRouter(tags=["documents"])


@router.post("/documents", response_model=IngestResponse)
async def upload_document(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
    ingestion: IngestionService = Depends(get_ingestion_service),
) -> IngestResponse:
    filename = validate_filename(file.filename)
    content = await file.read()
    if len(content) > settings.max_upload_bytes:
        raise PayloadTooLargeError(settings.max_upload_size_mb)
    return ingestion.ingest(filename, content)
