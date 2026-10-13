from pathlib import Path

import fitz
import pytest

from app.exceptions import EmptyDocumentError, UnreadableDocumentError, UnsupportedFileTypeError
from app.services.document_loader import load_document


def _pdf_with_text(text: str, pages: int = 1) -> bytes:
    document = fitz.open()
    for index in range(pages):
        page = document.new_page()
        page.insert_text((72, 72), f"{text} page {index + 1}")
    payload = document.tobytes()
    document.close()
    return payload


def test_load_markdown():
    extracted = load_document("notes.md", b"# Title\n\nEmployees receive 18 days of annual leave.")
    assert extracted.filename == "notes.md"
    assert "18 days" in extracted.text


def test_load_txt():
    extracted = load_document("policy.txt", b"Remote work is limited to 2 days each week.")
    assert extracted.filename == "policy.txt"
    assert "2 days" in extracted.text


def test_load_pdf_preserves_page_numbers():
    extracted = load_document("handbook.pdf", _pdf_with_text("Leave policy", pages=2))
    assert extracted.filename == "handbook.pdf"
    assert len(extracted.pages) == 2
    assert extracted.pages[0].page_number == 1
    assert extracted.pages[1].page_number == 2
    assert "Leave policy page 1" in extracted.text


def test_unsupported_extension():
    with pytest.raises(UnsupportedFileTypeError):
        load_document("photo.png", b"not-a-document")


def test_empty_file():
    with pytest.raises(EmptyDocumentError):
        load_document("empty.txt", b"")


def test_malformed_pdf():
    with pytest.raises(UnreadableDocumentError):
        load_document("broken.pdf", b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog")


def test_pdf_without_extractable_text():
    document = fitz.open()
    document.new_page()
    payload = document.tobytes()
    document.close()
    with pytest.raises(EmptyDocumentError):
        load_document("blank.pdf", payload)


def test_filename_uses_basename_only():
    extracted = load_document(r"..\secrets\notes.md", b"Visible policy text.")
    assert extracted.filename == "notes.md"
    assert Path(extracted.filename).name == extracted.filename
