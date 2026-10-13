from app.schemas import ExtractedDocument, ExtractedPage
from app.services.chunking import chunk_document, content_hash


def test_chunk_metadata_and_ids():
    document = ExtractedDocument(
        filename="handbook.md",
        text="Alpha paragraph.\n\n" + ("Beta sentence. " * 80),
        pages=[ExtractedPage(page_number=1, text="Alpha paragraph.\n\n" + ("Beta sentence. " * 80))],
    )
    chunks = chunk_document(document, chunk_size=200, chunk_overlap=40)
    assert len(chunks) >= 2
    assert chunks[0].filename == "handbook.md"
    assert chunks[0].chunk_index == 0
    assert chunks[0].document_id == content_hash(document.text)
    assert chunks[0].chunk_id == f"{chunks[0].document_id}:0"
    assert chunks[0].page_number == 1
    assert chunks[1].chunk_id != chunks[0].chunk_id


def test_duplicate_hash_is_stable():
    text = "Employees receive 18 days of annual leave."
    assert content_hash(text) == content_hash(text)


def test_pdf_page_assigned_to_chunk():
    page1 = "Page one policy. " * 20
    page2 = "Page two policy. " * 20
    document = ExtractedDocument(
        filename="split.pdf",
        text=f"{page1}\n\n{page2}",
        pages=[
            ExtractedPage(page_number=1, text=page1.strip()),
            ExtractedPage(page_number=2, text=page2.strip()),
        ],
    )
    chunks = chunk_document(document, chunk_size=120, chunk_overlap=20)
    pages = {chunk.page_number for chunk in chunks}
    assert 1 in pages
    assert 2 in pages
