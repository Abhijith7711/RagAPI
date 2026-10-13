from app.exceptions import LLMUnavailableError
from app.schemas import LLMStructuredResponse, RetrievedChunk
from app.services.qa import ABSTENTION, QAService
from tests.conftest import ScriptedLLM


class StaticRetrieval:
    def __init__(self, chunks: list[RetrievedChunk]):
        self.chunks = chunks

    def search(self, question: str, *, top_k: int, relevance_threshold: float):
        return self.chunks


def _chunk(chunk_id: str = "doc:0", text: str = "Employees receive 18 days of annual leave.") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document_id="doc",
        filename="employee_handbook.md",
        chunk_index=0,
        text=text,
        page_number=1,
        document_hash="abc",
        distance=0.1,
    )


def test_grounded_answer_uses_retrieved_excerpt():
    evidence = [_chunk()]
    llm = ScriptedLLM(
        LLMStructuredResponse(
            answer="Employees receive 18 days of annual leave.",
            found_in_documents=True,
            supporting_chunk_ids=["doc:0"],
        )
    )
    result = QAService(StaticRetrieval(evidence), llm).ask("How much leave?", top_k=5, relevance_threshold=0.5)
    assert result.found_in_documents is True
    assert result.sources[0].document == "employee_handbook.md"
    assert result.sources[0].excerpt == evidence[0].text
    assert result.sources[0].chunk_id == "doc:0"
    assert result.sources[0].page == 1


def test_unanswerable_question_abstains_without_llm_when_no_evidence():
    llm = ScriptedLLM()
    result = QAService(StaticRetrieval([]), llm).ask("What is the 401k match?", top_k=5, relevance_threshold=0.5)
    assert result.found_in_documents is False
    assert result.sources == []
    assert result.answer == ABSTENTION
    assert llm.calls == []


def test_multi_evidence_keeps_all_valid_citations():
    evidence = [
        _chunk("a:0", "Employees receive 18 days of annual leave."),
        _chunk("a:1", "Employees receive 10 days of paid sick leave."),
    ]
    evidence[1].filename = "employee_handbook.md"
    llm = ScriptedLLM(
        LLMStructuredResponse(
            answer="Employees receive 18 days of annual leave and 10 days of sick leave.",
            found_in_documents=True,
            supporting_chunk_ids=["a:0", "a:1"],
        )
    )
    result = QAService(StaticRetrieval(evidence), llm).ask("leave entitlements", top_k=5, relevance_threshold=0.5)
    assert {source.chunk_id for source in result.sources} == {"a:0", "a:1"}


def test_unknown_chunk_ids_are_rejected():
    llm = ScriptedLLM(
        LLMStructuredResponse(
            answer="Invented answer",
            found_in_documents=True,
            supporting_chunk_ids=["not-a-real-id"],
        )
    )
    result = QAService(StaticRetrieval([_chunk()]), llm).ask("leave", top_k=5, relevance_threshold=0.5)
    assert result.found_in_documents is False
    assert result.sources == []
    assert result.answer == ABSTENTION


def test_malformed_llm_response_abstains():
    llm = ScriptedLLM(error=ValueError("bad json"))
    result = QAService(StaticRetrieval([_chunk()]), llm).ask("leave", top_k=5, relevance_threshold=0.5)
    assert result.found_in_documents is False
    assert result.answer == ABSTENTION


def test_llm_api_error_does_not_invent_answer():
    llm = ScriptedLLM(error=LLMUnavailableError())
    try:
        QAService(StaticRetrieval([_chunk()]), llm).ask("leave", top_k=5, relevance_threshold=0.5)
        assert False, "expected LLMUnavailableError"
    except LLMUnavailableError:
        pass
