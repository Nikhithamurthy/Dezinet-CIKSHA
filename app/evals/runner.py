"""Run a RAG system over the golden set and gate it on thresholds (CI regression test)."""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean

from app.evals.core import GoldenItem, RagOutput, RagSystem
from app.evals.metrics import Scores, score_item

METRICS = ("faithfulness", "answer_relevance", "context_precision", "context_recall")
_HINTS = {
    "faithfulness": "generation is adding claims the context does not support (hallucination)",
    "answer_relevance": "answers drift from the question: check the prompt or the model",
    "context_precision": "retrieval ranks irrelevant chunks high: check embeddings and reranking",
    "context_recall": "retrieval misses needed facts: check chunking, top-k and the index",
}


@dataclass(frozen=True)
class Thresholds:
    faithfulness: float = 0.90
    answer_relevance: float = 0.85
    context_precision: float = 0.80
    context_recall: float = 0.80


@dataclass(frozen=True)
class EvalReport:
    n: int
    means: dict[str, float]
    thresholds: Thresholds
    worst: dict[str, tuple[str, float]]  # metric -> (question, score) of the weakest item

    @property
    def failures(self) -> list[str]:
        return [m for m in METRICS if self.means[m] < getattr(self.thresholds, m)]

    @property
    def passed(self) -> bool:
        return not self.failures

    def format(self) -> str:
        lines = [f"{self.n} golden questions"]
        for metric in METRICS:
            value = self.means[metric]
            limit = getattr(self.thresholds, metric)
            mark = "ok" if value >= limit else "FAIL"
            lines.append(f"{metric:<18} {value:.3f}  (min {limit:.2f})  {mark}")
            if mark == "FAIL":
                question, score = self.worst[metric]
                lines.append(f"  weakest: {question!r} -> {score:.2f}")
                lines.append(f"  likely cause: {_HINTS[metric]}")
        lines.append("RESULT: " + ("PASS" if self.passed else "FAIL"))
        return "\n".join(lines)


def _values(scores: Sequence[Scores], metric: str) -> list[float]:
    return [float(getattr(score, metric)) for score in scores]


async def evaluate(
    system: RagSystem,
    items: Sequence[GoldenItem],
    thresholds: Thresholds | None = None,
    *,
    concurrency: int = 8,
) -> EvalReport:
    if not items:
        raise ValueError("golden set is empty")
    gate = asyncio.Semaphore(concurrency)

    async def answer(item: GoldenItem) -> RagOutput:
        async with gate:
            return await system.run(item.question)

    outputs = await asyncio.gather(*(answer(item) for item in items))
    scores = [score_item(i, o) for i, o in zip(items, outputs, strict=True)]
    means: dict[str, float] = {}
    worst: dict[str, tuple[str, float]] = {}
    for metric in METRICS:
        values = _values(scores, metric)
        means[metric] = fmean(values)
        index = min(range(len(values)), key=values.__getitem__)
        worst[metric] = (items[index].question, values[index])
    return EvalReport(len(items), means, thresholds or Thresholds(), worst)
