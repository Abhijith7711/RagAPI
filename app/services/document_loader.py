"""Extract text from PDF, TXT, and Markdown uploads."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import fitz

from app.exceptions import EmptyDocumentError, UnreadableDocumentError, UnsupportedFileTypeError
from app.schemas import ExtractedDocument, ExtractedPage

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def validate_filename(filename: str | None) -> str:
    if not filename or not filename.strip():
        raise UnsupportedFileTypeError("unnamed")
    safe_name = Path(filename).name
    suffix = Path(safe_name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(safe_name)
    return safe_name


def load_document(filename: str, content: bytes) -> ExtractedDocument:
    if content is None or len(content) == 0:
        raise EmptyDocumentError()

    safe_name = validate_filename(filename)
    suffix = Path(safe_name).suffix.lower()

    if suffix == ".pdf":
        pages = _extract_pdf(content)
    else:
        pages = _extract_text_file(content)

    combined = normalize_whitespace("\n\n".join(page.text for page in pages if page.text.strip()))
    if not combined:
        raise EmptyDocumentError("No extractable text was found in the document.")

    normalized_pages = [
        ExtractedPage(page_number=page.page_number, text=normalize_whitespace(page.text))
        for page in pages
        if page.text.strip()
    ]
    return ExtractedDocument(filename=safe_name, text=combined, pages=normalized_pages)


def _extract_text_file(content: bytes) -> list[ExtractedPage]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = content.decode("latin-1")
        except UnicodeDecodeError as exc:
            raise UnreadableDocumentError("The text file is not readable as UTF-8 or Latin-1.") from exc
    return [ExtractedPage(page_number=None, text=text)]


def _extract_pdf(content: bytes) -> list[ExtractedPage]:
    try:
        document = fitz.open(stream=content, filetype="pdf")
    except Exception as exc:
        raise UnreadableDocumentError("The PDF is malformed or unreadable.") from exc

    try:
        pages: list[ExtractedPage] = []
        for index, page in enumerate(document, start=1):
            try:
                text = page.get_text() or ""
            except Exception as exc:
                raise UnreadableDocumentError(f"Failed to extract text from PDF page {index}.") from exc
            pages.append(ExtractedPage(page_number=index, text=text))
        if not pages:
            raise EmptyDocumentError("The PDF contains no pages.")
        return pages
    finally:
        document.close()
