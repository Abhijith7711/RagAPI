"""Shared FastAPI dependencies bound on application startup."""

from fastapi import Request

from app.config import Settings
from app.services.ingestion import IngestionService
from app.services.qa import QAService


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_ingestion_service(request: Request) -> IngestionService:
    return request.app.state.ingestion_service


def get_qa_service(request: Request) -> QAService:
    return request.app.state.qa_service
