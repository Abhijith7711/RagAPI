"""Persistent ChromaDB adapter."""

from __future__ import annotations

import logging
from pathlib import Path

import chromadb

from app.exceptions import EmbeddingModelMismatchError
from app.schemas import Chunk, RetrievedChunk

logger = logging.getLogger(__name__)

DISTANCE_METRIC = "cosine"


class VectorStore:
    def __init__(
        self,
        persist_directory: str | Path,
        collection_name: str,
        embedding_model: str,
    ):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self._client = chromadb.PersistentClient(path=str(self.persist_directory))
        self._collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": DISTANCE_METRIC,
                "embedding_model": self.embedding_model,
            },
        )
        stored_model = (collection.metadata or {}).get("embedding_model")
        if stored_model and stored_model != self.embedding_model:
            raise EmbeddingModelMismatchError(stored=str(stored_model), requested=self.embedding_model)
        return collection

    def count(self) -> int:
        return self._collection.count()

    def find_by_document_hash(self, document_hash: str) -> list[Chunk]:
        result = self._collection.get(
            where={"document_hash": document_hash},
            include=["documents", "metadatas"],
        )
        return self._to_chunks(result)

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        ids = [chunk.chunk_id for chunk in chunks]
        existing = self._collection.get(ids=ids)
        if existing.get("ids"):
            self._collection.delete(ids=existing["ids"])
        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=[chunk.text for chunk in chunks],
            metadatas=[_chunk_metadata(chunk) for chunk in chunks],
        )
        logger.info("Indexed %s chunks for document %s", len(chunks), chunks[0].filename)

    def query(self, embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        if self.count() == 0:
            return []
        n_results = min(top_k, self.count())
        result = self._collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        ids = (result.get("ids") or [[]])[0]
        retrieved: list[RetrievedChunk] = []
        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            metadata = metadata or {}
            retrieved.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    document_id=str(metadata.get("document_id", "")),
                    filename=str(metadata.get("filename", "")),
                    chunk_index=int(metadata.get("chunk_index", 0)),
                    text=text,
                    page_number=_optional_page(metadata.get("page_number")),
                    document_hash=str(metadata.get("document_hash", "")),
                    distance=float(distance),
                )
            )
        return retrieved

    def _to_chunks(self, result: dict) -> list[Chunk]:
        chunks: list[Chunk] = []
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        for chunk_id, text, metadata in zip(ids, documents, metadatas):
            metadata = metadata or {}
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=str(metadata.get("document_id", "")),
                    filename=str(metadata.get("filename", "")),
                    chunk_index=int(metadata.get("chunk_index", 0)),
                    text=text or "",
                    page_number=_optional_page(metadata.get("page_number")),
                    document_hash=str(metadata.get("document_hash", "")),
                )
            )
        return chunks


def _chunk_metadata(chunk: Chunk) -> dict:
    return {
        "document_id": chunk.document_id,
        "filename": chunk.filename,
        "chunk_index": chunk.chunk_index,
        "document_hash": chunk.document_hash,
        "page_number": chunk.page_number if chunk.page_number is not None else -1,
    }


def _optional_page(value) -> int | None:
    if value is None or value == -1 or value == "-1":
        return None
    return int(value)
