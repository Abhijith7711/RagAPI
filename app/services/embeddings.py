"""Embedding backends used for indexing and query-time retrieval."""

from __future__ import annotations

import hashlib
import math
from typing import Protocol


class EmbeddingService(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


class FakeEmbeddings:
    """Deterministic, dependency-free embedder for unit tests.

    Tokens are hashed into a fixed-size bag-of-words vector so overlapping
    wording produces similar vectors without downloading a model.
    """

    def __init__(self, dimensions: int = 32, model_name: str = "fake-hash-embeddings"):
        self.dimensions = dimensions
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in text.lower().split():
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:2], "little") % self.dimensions
            vector[index] += 1.0
        return _l2_normalize(vector)


class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.dimensions = 384
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self.dimensions = int(self._model.get_sentence_embedding_dimension())
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
