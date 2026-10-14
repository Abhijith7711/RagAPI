"""Split extracted documents into overlapping character chunks."""

from __future__ import annotations

import hashlib

from app.schemas import Chunk, ExtractedDocument

# ~4 characters per token: 2000 chars ≈ 500 tokens, 300 chars ≈ 75 tokens.
DEFAULT_CHUNK_SIZE = 2000
DEFAULT_CHUNK_OVERLAP = 300


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def chunk_document(
    document: ExtractedDocument,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    document_hash = content_hash(document.text)
    document_id = document_hash
    page_spans = _page_spans(document)
    windows = _split_windows(document.text, chunk_size, chunk_overlap)

    chunks: list[Chunk] = []
    for index, (start, end, text) in enumerate(windows):
        chunks.append(
            Chunk(
                chunk_id=f"{document_id}:{index}",
                document_id=document_id,
                filename=document.filename,
                chunk_index=index,
                text=text,
                page_number=_page_for_span(page_spans, start, end),
                document_hash=document_hash,
            )
        )
    return chunks


def _page_spans(document: ExtractedDocument) -> list[tuple[int, int, int | None]]:
    """Map character offsets in the combined text to original page numbers."""
    spans: list[tuple[int, int, int | None]] = []
    cursor = 0
    remaining = document.text
    for page in document.pages:
        page_text = page.text
        if not page_text:
            continue
        found_at = remaining.find(page_text)
        if found_at == -1:
            continue
        start = cursor + found_at
        end = start + len(page_text)
        spans.append((start, end, page.page_number))
        cursor = end
        remaining = document.text[cursor:]
    if not spans:
        spans.append((0, len(document.text), None))
    return spans


def _page_for_span(spans: list[tuple[int, int, int | None]], start: int, end: int) -> int | None:
    best_page = spans[0][2]
    best_overlap = -1
    for span_start, span_end, page in spans:
        overlap = min(end, span_end) - max(start, span_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_page = page
    return best_page


def _split_windows(text: str, chunk_size: int, overlap: int) -> list[tuple[int, int, str]]:
    windows: list[tuple[int, int, str]] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        if end < length:
            window = text[start:end]
            break_at = _best_break(window, chunk_size)
            if break_at is not None:
                end = start + break_at
        piece = text[start:end].strip()
        if piece:
            # Recompute offsets for stripped piece so metadata stays aligned.
            stripped_start = text.find(piece, start, end + 1)
            if stripped_start == -1:
                stripped_start = start
            windows.append((stripped_start, stripped_start + len(piece), piece))
        if end >= length:
            break
        raw_start = max(end - overlap, start + 1)
        # Snap raw_start forward to avoid slicing mid-word
        while raw_start < end and raw_start > 0 and text[raw_start - 1] not in " \n\r\t":
            raw_start += 1
        start = raw_start if raw_start < end else max(end - overlap, start + 1)
    return windows


def _best_break(window: str, chunk_size: int) -> int | None:
    min_break = int(chunk_size * 0.5)
    for separator in ("\n\n", "\n", " "):
        idx = window.rfind(separator)
        if idx >= min_break:
            return idx
    return None
