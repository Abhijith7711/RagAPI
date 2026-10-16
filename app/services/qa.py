"""Grounded question answering with citation validation and abstention."""

from __future__ import annotations

import logging
from typing import Protocol

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.exceptions import LLMUnavailableError
from app.prompts import SYSTEM_PROMPT, build_user_prompt
from app.schemas import AskResponse, LLMStructuredResponse, Source

logger = logging.getLogger(__name__)

ABSTENTION = "The information is unavailable in the provided documents."


class LLMClient(Protocol):
    """Protocol so tests can inject a scripted/fake LLM."""
    def invoke(self, question: str, evidence: list[dict]) -> LLMStructuredResponse: ...


class LangChainLLM:
    """Production LLM adapter using any LangChain ChatModel."""

    def __init__(self, chat_model: BaseChatModel):
        self.structured = chat_model.with_structured_output(LLMStructuredResponse)

    def invoke(self, question: str, evidence: list[dict]) -> LLMStructuredResponse:
        user_prompt = build_user_prompt(question, evidence)
        try:
            result = self.structured.invoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ])
        except Exception as exc:
            logger.exception("LLM request failed")
            raise LLMUnavailableError("The language model provider request failed.") from exc
        if result is None:
            raise ValueError("The model did not return valid structured output.")
        return result


class QAService:
    def __init__(self, vectorstore: Chroma, llm: LLMClient):
        self.vectorstore = vectorstore
        self.llm = llm

    def ask(self, question: str, *, top_k: int, relevance_threshold: float) -> AskResponse:
        # Retrieve relevant chunks
        if self.vectorstore._collection.count() == 0:
            return self._abstain(question)

        results = self.vectorstore.similarity_search_with_score(question, k=top_k)

        # Filter by relevance threshold (cosine distance: lower = more similar)
        evidence: list[tuple[Document, float]] = [
            (doc, score) for doc, score in results if score <= relevance_threshold
        ]
        if not evidence:
            return self._abstain(question)

        # Build evidence dicts for prompt
        evidence_dicts = []
        for idx, (doc, score) in enumerate(evidence):
            chunk_id = (
                doc.metadata.get("chunk_id")
                or (doc.metadata.get("document_id", "") + f":{doc.metadata.get('chunk_index', idx)}")
            )
            evidence_dicts.append({
                "chunk_id": chunk_id,
                "filename": doc.metadata.get("filename", ""),
                "page_number": doc.metadata.get("page_number"),
                "text": doc.page_content,
            })

        try:
            llm_result = self.llm.invoke(question, evidence_dicts)
        except LLMUnavailableError:
            raise
        except (ValueError, Exception):
            logger.warning("LLM output error; abstaining")
            return self._abstain(question)

        return self._ground_response(question, llm_result, evidence_dicts)

    def _ground_response(
        self, question: str, llm_result: LLMStructuredResponse, evidence: list[dict]
    ) -> AskResponse:
        """Validate cited chunk IDs against actual retrieved evidence."""
        evidence_by_id = {e["chunk_id"]: e for e in evidence}
        # Also index by document_hash and filename for flexible matching
        evidence_by_doc: dict[str, dict] = {}
        for e in evidence:
            doc_hash = e["chunk_id"].split(":")[0] if ":" in e["chunk_id"] else ""
            evidence_by_doc[doc_hash] = e
            evidence_by_doc[e["filename"]] = e

        resolved = []
        for cited in llm_result.supporting_chunk_ids:
            if cited in evidence_by_id:
                resolved.append(evidence_by_id[cited])
            elif cited in evidence_by_doc:
                resolved.append(evidence_by_doc[cited])
            else:
                # Try prefix matching (e.g., LLM cites "abc123" but actual is "abc123:0")
                matching = [e for e in evidence if e["chunk_id"].startswith(f"{cited}:")]
                resolved.extend(matching)

        if not llm_result.found_in_documents or not resolved:
            logger.info("Abstaining after citation validation")
            return self._abstain(question)

        # Deduplicate sources
        seen: set[str] = set()
        sources: list[Source] = []
        for e in resolved:
            if e["chunk_id"] not in seen:
                seen.add(e["chunk_id"])
                excerpt = e["text"] if len(e["text"]) <= 500 else e["text"][:497] + "..."
                sources.append(Source(
                    document=e["filename"],
                    excerpt=excerpt,
                    chunk_id=e["chunk_id"],
                    page=e.get("page_number"),
                ))

        return AskResponse(
            question=question,
            answer=llm_result.answer,
            found_in_documents=True,
            sources=sources,
        )

    @staticmethod
    def _abstain(question: str) -> AskResponse:
        return AskResponse(
            question=question,
            answer=ABSTENTION,
            found_in_documents=False,
            sources=[],
        )
