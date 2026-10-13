"""Document ingestion: extract, chunk, embed, and persist."""

from __future__ import annotations

import logging

from app.schemas import IngestResponse
from app.services.chunking import chunk_document, content_hash
from app.services.document_loader import load_document
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


class IngestionService:
    def __init__(
        self,
        embedder: EmbeddingService,
        vector_store: VectorStore,
        chunk_size: int,
        chunk_overlap: int,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def ingest(self, filename: str, content: bytes) -> IngestResponse:
        extracted = load_document(filename, content)
        document_hash = content_hash(extracted.text)
        existing = self.vector_store.find_by_document_hash(document_hash)
        if existing:
            logger.info("Duplicate document skipped: %s", extracted.filename)
            return IngestResponse(
                status="duplicate",
                filename=extracted.filename,
                document_id=document_hash,
                chunk_count=len(existing),
                is_duplicate=True,
            )

        chunks = chunk_document(
            extracted,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        embeddings = self.embedder.embed_documents([chunk.text for chunk in chunks])
        self.vector_store.add_chunks(chunks, embeddings)
        return IngestResponse(
            status="ingested",
            filename=extracted.filename,
            document_id=document_hash,
            chunk_count=len(chunks),
            is_duplicate=False,
        )
