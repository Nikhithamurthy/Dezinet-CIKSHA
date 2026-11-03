import asyncio
from pathlib import Path
from typing import Any

import pytest

from app.evals.core import Document, RagOutput, support
from app.evals.dataset import (
    QAPair,
    TemplateWriter,
    build_corpus,
    generate_golden,
    load_golden,
    save_golden,
)
from app.evals.judge import judge_pair
from app.evals.metrics import answer_relevance, context_precision, context_recall, faithfulness
from app.evals.pipeline import ExtractiveAnswerer, LexicalRetriever, RagPipeline
from app.evals.runner import evaluate
from app.safety.guard import Guard

run = asyncio.run
GT = "The refund window of the Growth plan is 30 days."


def _golden() -> Any:
    return run(generate_golden(build_corpus(), TemplateWriter(), n=50))


# ----------------------------------------------------------------------------- dataset


def test_golden_set_has_50_unique_grounded_pairs() -> None:
    docs = build_corpus()
    golden = _golden()
    assert len(golden) == 50
    assert len({g.question for g in golden}) == 50
    texts = {d.title: d.text for d in docs}
    assert all(support(g.ground_truth, texts[g.source]) == 1.0 for g in golden)


def test_golden_set_round_trips_through_jsonl(tmp_path: Path) -> None:
    golden = _golden()
    path = tmp_path / "golden.jsonl"
    save_golden(golden, path)
    assert load_golden(path) == golden


def test_hallucinated_pairs_are_filtered_out() -> None:
    class BadWriter:
        async def write(self, document: Document) -> list[QAPair]:
            return [QAPair(question="What is the moon made of?", ground_truth="Green cheese.")]

    assert run(generate_golden(build_corpus(), BadWriter(), n=50)) == []


# ----------------------------------------------------------------------------- metrics


def test_faithfulness_flags_unsupported_claims() -> None:
    assert faithfulness(GT, [GT]) == 1.0
    assert faithfulness("The refund window is 90 days for everyone.", [GT]) == 0.0
    assert faithfulness(GT + " Customers love it.", [GT]) == 0.5


def test_answer_relevance() -> None:
    question = "What is the refund window of the Growth plan?"
    assert answer_relevance(question, GT) == 1.0
    assert answer_relevance(question, "The weather is sunny.") == 0.0
    assert answer_relevance(question, "I do not know.") == 0.0


def test_context_precision_rewards_ranking() -> None:
    other = "The price of the Starter plan is $12 per month."
    assert context_precision([GT, other], GT) == 1.0
    assert context_precision([other, GT], GT) == 0.5
    assert context_precision([other], GT) == 0.0


def test_context_recall() -> None:
    truth = GT + " Refunds are paid by card."
    assert context_recall(truth, [GT]) == 0.5
    assert context_recall(truth, [GT, "Refunds are paid by card."]) == 1.0


# ------------------------------------------------------------------------- CI gate


class FixedAnswer:
    """Right retrieval, but the generator hallucinates."""

    def __init__(self, inner: RagPipeline) -> None:
        self._inner = inner

    async def run(self, question: str) -> RagOutput:
        out = await self._inner.run(question)
        return RagOutput("The refund window is 90 days for everyone.", out.contexts)


class WrongContexts:
    """Retrieval ignores the question."""

    def __init__(self, inner: RagPipeline) -> None:
        self._inner = inner

    async def run(self, question: str) -> RagOutput:
        return await self._inner.run("What is the price of the Starter plan?")


@pytest.mark.parametrize("guarded", [False, True])
def test_baseline_passes_the_gate(guarded: bool) -> None:
    pipeline = RagPipeline(
        LexicalRetriever(build_corpus()),
        ExtractiveAnswerer(),
        guard=Guard() if guarded else None,
    )
    report = run(evaluate(pipeline, _golden()))
    assert report.passed, report.format()


def test_gate_blocks_a_hallucinating_generator() -> None:
    inner = RagPipeline(LexicalRetriever(build_corpus()), ExtractiveAnswerer())
    report = run(evaluate(FixedAnswer(inner), _golden()))
    assert not report.passed
    assert "faithfulness" in report.failures
    assert "FAIL" in report.format()


def test_gate_blocks_broken_retrieval() -> None:
    inner = RagPipeline(LexicalRetriever(build_corpus()), ExtractiveAnswerer())
    report = run(evaluate(WrongContexts(inner), _golden()))
    assert not report.passed
    assert "context_recall" in report.failures


# ----------------------------------------------------------------------------- judge


class _Judge:
    def __init__(self, replies: list[str]) -> None:
        self._replies = replies

    async def complete(self, system: str, prompt: str) -> str:
        return self._replies.pop(0)


def test_pairwise_judge_cancels_position_bias() -> None:
    always_first: Any = _Judge(['{"winner": "first"}'] * 2)
    assert run(judge_pair(always_first, "q", "a", "b", "truth")) == "tie"


def test_pairwise_judge_keeps_a_consistent_verdict() -> None:
    consistent: Any = _Judge(['{"winner": "first"}', '{"winner": "second"}'])
    assert run(judge_pair(consistent, "q", "a", "b", "truth")) == "a"
