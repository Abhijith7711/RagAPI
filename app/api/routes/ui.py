"""UI route — serves the web interface."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter(tags=["ui"])

_TEMPLATE_PATH = Path(__file__).resolve().parents[2] / "templates" / "index.html"


@lru_cache(maxsize=1)
def _get_index_html() -> str:
    return _TEMPLATE_PATH.read_text(encoding="utf-8")


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index() -> HTMLResponse:
    """Serve the main UI page."""
    return HTMLResponse(content=_get_index_html())
