import asyncio
from typing import cast

from app.rag.llm import LLM, MockLLM
from app.rag.pipeline import RagPipeline, RagResult
from app.rag.prompts import NO_ANSWER
from app.retrieval.eval_set import Doc
from app.retrieval.pipeline import HybridRetriever

DOCS = [
    Doc("d1", "Refunds for annual plans are issued within 30 days of purchase."),
    Doc("d2", "Support is available by email on weekdays."),
]


class StubRetriever:
    """Returns a fixed ranking for every query."""

    def __init__(self, ranking: list[str]) -> None:
        self._ranking = ranking

    async def hybrid_rerank(self, query: str, k: int = 5) -> list[str]:
        return self._ranking[:k]


class FakeLLM:
    """Returns a canned reply and counts how often it is called."""

    def __init__(self, reply: str) -> None:
        self.name = "fake"
        self.reply = reply
        self.calls = 0

    async def complete(self, system: str, user: str) -> str:
        self.calls += 1
        return self.reply


def _pipeline(llm: object, **kwargs: object) -> RagPipeline:
    retriever = cast(HybridRetriever, StubRetriever(["d1", "d2"]))
    return RagPipeline(retriever, DOCS, cast(LLM, llm), **kwargs)  # type: ignore[arg-type]


def _ask(pipeline: RagPipeline, question: str) -> RagResult:
    return asyncio.run(pipeline.ask(question))


def test_weak_support_falls_back_without_calling_llm() -> None:
    llm = FakeLLM("should never be used")
    pipeline = _pipeline(llm, use_expansion=False, use_hyde=False)
    result = _ask(pipeline, "warranty coverage laptops")
    assert result.status == "fallback"
    assert result.grounded
    assert llm.calls == 0


def test_supported_question_gets_cited_answer() -> None:
    pipeline = _pipeline(MockLLM(), use_expansion=False, use_hyde=False)
    result = _ask(pipeline, "refunds annual plans")
    assert result.status == "answered"
    assert result.grounded
    assert [c.chunk_id for c in result.citations] == ["d1"]
    assert result.unsupported_citations == []


def test_invented_citation_is_flagged_as_ungrounded() -> None:
    llm = FakeLLM("Refunds are issued within 30 days. [ghost]")
    pipeline = _pipeline(llm, use_expansion=False, use_hyde=False)
    result = _ask(pipeline, "refunds annual plans")
    assert result.status == "answered"
    assert result.unsupported_citations == ["ghost"]
    assert not result.grounded


def test_model_refusal_becomes_fallback() -> None:
    llm = FakeLLM(NO_ANSWER)
    pipeline = _pipeline(llm, use_expansion=False, use_hyde=False)
    result = _ask(pipeline, "refunds annual plans")
    assert result.status == "fallback"
    assert result.grounded


def test_query_transforms_add_extra_queries() -> None:
    pipeline = _pipeline(MockLLM())
    result = _ask(pipeline, "refunds annual plans")
    assert result.queries[0] == "refunds annual plans"
    assert len(result.queries) > 1
    assert result.hyde
