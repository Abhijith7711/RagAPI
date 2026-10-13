"""Grounded question answering with citation validation and abstention."""

from __future__ import annotations

import logging

from app.exceptions import LLMUnavailableError
from app.schemas import AskResponse, RetrievedChunk, Source
from app.services.llm import LLMClient
from app.services.retrieval import RetrievalService

logger = logging.getLogger(__name__)

ABSTENTION = "The information is unavailable in the provided documents."


class QAService:
    def __init__(self, retrieval: RetrievalService, llm: LLMClient):
        self.retrieval = retrieval
        self.llm = llm

    def ask(
        self,
        question: str,
        *,
        top_k: int,
        relevance_threshold: float,
    ) -> AskResponse:
        evidence = self.retrieval.search(
            question,
            top_k=top_k,
            relevance_threshold=relevance_threshold,
        )
        if not evidence:
            return AskResponse(
                question=question,
                answer=ABSTENTION,
                found_in_documents=False,
                sources=[],
            )

        try:
            llm_result = self.llm.complete(question, evidence)
        except LLMUnavailableError:
            raise
        except ValueError:
            logger.warning("Malformed LLM output; abstaining")
            return AskResponse(
                question=question,
                answer=ABSTENTION,
                found_in_documents=False,
                sources=[],
            )

        return self._ground_response(question, llm_result, evidence)

    def _ground_response(self, question: str, llm_result, evidence: list[RetrievedChunk]) -> AskResponse:
        evidence_by_id = {item.chunk_id: item for item in evidence}
        valid_ids = [chunk_id for chunk_id in llm_result.supporting_chunk_ids if chunk_id in evidence_by_id]

        if not llm_result.found_in_documents or not valid_ids:
            logger.info("Abstaining after citation validation")
            return AskResponse(
                question=question,
                answer=ABSTENTION,
                found_in_documents=False,
                sources=[],
            )

        sources = [_to_source(evidence_by_id[chunk_id]) for chunk_id in valid_ids]
        return AskResponse(
            question=question,
            answer=llm_result.answer,
            found_in_documents=True,
            sources=sources,
        )


def _to_source(chunk: RetrievedChunk) -> Source:
    excerpt = chunk.text if len(chunk.text) <= 500 else chunk.text[:497] + "..."
    return Source(
        document=chunk.filename,
        excerpt=excerpt,
        chunk_id=chunk.chunk_id,
        page=chunk.page_number,
    )
