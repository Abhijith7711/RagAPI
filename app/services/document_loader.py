"""Extract text from PDF, TXT, and Markdown uploads → LangChain Documents."""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

import fitz
from langchain_core.documents import Document

from app.exceptions import EmptyDocumentError, UnreadableDocumentError, UnsupportedFileTypeError

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}


def content_hash(text: str) -> str:
    """SHA-256 fingerprint used for duplicate detection."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def validate_filename(filename: str | None) -> str:
    if not filename or not filename.strip():
        raise UnsupportedFileTypeError("unnamed")
    safe_name = Path(filename.replace("\\", "/")).name
    if Path(safe_name).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(safe_name)
    return safe_name


def load_document(filename: str, content: bytes) -> list[Document]:
    """Parse an uploaded file into a list of LangChain Documents (one per page/section).

    Each Document carries metadata: filename, page_number, document_hash.
    """
    if not content:
        raise EmptyDocumentError()

    safe_name = validate_filename(filename)
    suffix = Path(safe_name).suffix.lower()

    if suffix == ".pdf":
        pages = _extract_pdf(content)
    else:
        pages = _extract_text(content)

    # Build LangChain Documents with metadata
    docs: list[Document] = []
    all_text_parts: list[str] = []
    for page_text, page_number in pages:
        cleaned = _normalize(page_text)
        if cleaned:
            all_text_parts.append(cleaned)
            docs.append(Document(
                page_content=cleaned,
                metadata={"filename": safe_name, "page_number": page_number},
            ))

    if not docs:
        raise EmptyDocumentError("No extractable text was found in the document.")

    # Attach document_hash to every document (computed from combined text)
    doc_hash = content_hash("\n\n".join(all_text_parts))
    for doc in docs:
        doc.metadata["document_hash"] = doc_hash

    return docs


def _extract_text(content: bytes) -> list[tuple[str, int | None]]:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = content.decode("latin-1")
        except UnicodeDecodeError as exc:
            raise UnreadableDocumentError("The text file is not readable as UTF-8 or Latin-1.") from exc
    return [(text, None)]


def _extract_pdf(content: bytes) -> list[tuple[str, int]]:
    try:
        document = fitz.open(stream=content, filetype="pdf")
    except Exception as exc:
        raise UnreadableDocumentError("The PDF is malformed or unreadable.") from exc
    try:
        pages: list[tuple[str, int]] = []
        for index, page in enumerate(document, start=1):
            try:
                text = page.get_text() or ""
            except Exception as exc:
                raise UnreadableDocumentError(f"Failed to extract text from PDF page {index}.") from exc
            pages.append((text, index))
        if not pages:
            raise EmptyDocumentError("The PDF contains no pages.")
        return pages
    finally:
        document.close()
