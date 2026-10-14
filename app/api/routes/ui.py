"""UI route — serves the web interface."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["ui"])

_TEMPLATE_PATH = Path(__file__).resolve().parents[2] / "templates" / "index.html"


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index() -> HTMLResponse:
    """Serve the main UI page."""
    html = _TEMPLATE_PATH.read_text(encoding="utf-8")
    return HTMLResponse(content=html)
