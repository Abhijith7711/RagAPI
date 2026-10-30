"""Document ingestion: load, split, embed, and persist using LangChain."""

from __future__ import annotations

import logging

from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.schemas import IngestResponse
from app.services.document_loader import load_document

logger = logging.getLogger(__name__)

_BATCH_SIZE = 64  # Max documents per vectorstore.add_documents() call


class IngestionService:
    """Handles document ingestion: loading, splitting, deduplication, and indexing."""

    def __init__(self, vectorstore: VectorStore, chunk_size: int, chunk_overlap: int) -> None:
        self.vectorstore = vectorstore
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def _find_existing_by_hash(self, document_hash: str) -> dict:
        """Query ChromaDB for chunks that share the given document hash.

        NOTE: Uses the private ``_collection`` attribute on the Chroma
        vectorstore. If langchain-chroma exposes a public metadata-query helper
        in a future version, replace this with that call.
        """
        return self.vectorstore._collection.get(  # noqa: SLF001
            where={"document_hash": document_hash},
            include=[],
        )

    def ingest(self, filename: str, content: bytes) -> IngestResponse:
        docs = load_document(filename, content)
        document_hash = docs[0].metadata["document_hash"]

        # Duplicate detection via ChromaDB metadata query
        existing = self._find_existing_by_hash(document_hash)
        if existing and existing.get("ids"):
            logger.info("Duplicate document skipped: %s", filename)
            return IngestResponse(
                status="duplicate",
                filename=filename,
                document_id=document_hash,
                chunk_count=len(existing["ids"]),
                is_duplicate=True,
            )

        # Split into chunks and attach indexing metadata
        chunks = self.splitter.split_documents(docs)
        for index, chunk in enumerate(chunks):
            chunk_id = f"{document_hash}:{index}"
            chunk.metadata["chunk_index"] = index
            chunk.metadata["document_id"] = document_hash
            chunk.metadata["chunk_id"] = chunk_id
            chunk.id = chunk_id

        for i in range(0, len(chunks), _BATCH_SIZE):
            self.vectorstore.add_documents(chunks[i : i + _BATCH_SIZE])
        logger.info("Indexed %s chunks for %s", len(chunks), filename)
        return IngestResponse(
            status="ingested",
            filename=filename,
            document_id=document_hash,
            chunk_count=len(chunks),
            is_duplicate=False,
        )
