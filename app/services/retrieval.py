"""Query embedding, similarity search, and relevance filtering."""

from __future__ import annotations

import logging

from app.schemas import RetrievedChunk
from app.services.embeddings import EmbeddingService
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RetrievalService:
    def __init__(self, embedder: EmbeddingService, vector_store: VectorStore):
        self.embedder = embedder
        self.vector_store = vector_store

    def search(
        self,
        question: str,
        *,
        top_k: int,
        relevance_threshold: float,
    ) -> list[RetrievedChunk]:
        if self.vector_store.count() == 0:
            return []
        query_embedding = self.embedder.embed_query(question)
        candidates = self.vector_store.query(query_embedding, top_k=top_k)
        filtered = [chunk for chunk in candidates if chunk.distance <= relevance_threshold]
        logger.info(
            "Retrieved %s candidates, %s passed relevance_threshold=%.3f",
            len(candidates),
            len(filtered),
            relevance_threshold,
        )
        return filtered
