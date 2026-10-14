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
        evidence_by_doc = {}
        for item in evidence:
            evidence_by_doc[item.document_hash] = item
            evidence_by_doc[item.document_id] = item
            evidence_by_doc[item.filename] = item

        resolved_chunks = []
        for cited in llm_result.supporting_chunk_ids:
            if cited in evidence_by_id:
                resolved_chunks.append(evidence_by_id[cited])
            elif cited in evidence_by_doc:
                resolved_chunks.append(evidence_by_doc[cited])
            else:
                matching = [item for item in evidence if item.chunk_id.startswith(f"{cited}:")]
                if matching:
                    resolved_chunks.extend(matching)

        if not llm_result.found_in_documents or not resolved_chunks:
            logger.info("Abstaining after citation validation")
            return AskResponse(
                question=question,
                answer=ABSTENTION,
                found_in_documents=False,
                sources=[],
            )

        unique_sources = []
        seen = set()
        for item in resolved_chunks:
            if item.chunk_id not in seen:
                seen.add(item.chunk_id)
                unique_sources.append(_to_source(item))

        return AskResponse(
            question=question,
            answer=llm_result.answer,
            found_in_documents=True,
            sources=unique_sources,
        )


def _to_source(chunk: RetrievedChunk) -> Source:
    excerpt = chunk.text if len(chunk.text) <= 500 else chunk.text[:497] + "..."
    return Source(
        document=chunk.filename,
        excerpt=excerpt,
        chunk_id=chunk.chunk_id,
        page=chunk.page_number,
    )
