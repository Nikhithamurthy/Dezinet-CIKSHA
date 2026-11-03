"""The four RAGAS-style metrics, computed offline with lexical support (no LLM needed).

faithfulness       claims in the answer supported by the retrieved context
answer_relevance   does the answer address the question (refusals score 0)
context_precision  average precision: are the useful chunks ranked first
context_recall     is every fact in the ground truth present in the retrieved context"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.evals.core import GoldenItem, RagOutput, sentences, support, token_set

SUPPORT_THRESHOLD = 0.8
_REFUSAL = re.compile(
    r"\bi (?:do not|don't|cannot|can't) (?:know|answer|help)\b|\bnot enough information\b",
    re.IGNORECASE,
)


def faithfulness(answer: str, contexts: Sequence[str]) -> float:
    claims = sentences(answer)
    if not claims or not contexts:
        return 0.0
    joined = " ".join(contexts)
    supported = sum(1 for claim in claims if support(claim, joined) >= SUPPORT_THRESHOLD)
    return supported / len(claims)


def answer_relevance(question: str, answer: str) -> float:
    if _REFUSAL.search(answer):
        return 0.0
    wanted = token_set(question)
    if not wanted:
        return 0.0
    return len(wanted & token_set(answer)) / len(wanted)


def context_precision(contexts: Sequence[str], ground_truth: str) -> float:
    relevant = [support(ground_truth, chunk) >= SUPPORT_THRESHOLD for chunk in contexts]
    total = sum(relevant)
    if total == 0:
        return 0.0
    hits = 0
    average = 0.0
    for rank, is_relevant in enumerate(relevant, start=1):
        if is_relevant:
            hits += 1
            average += hits / rank
    return average / total


def context_recall(ground_truth: str, contexts: Sequence[str]) -> float:
    claims = sentences(ground_truth)
    if not claims or not contexts:
        return 0.0
    joined = " ".join(contexts)
    found = sum(1 for claim in claims if support(claim, joined) >= SUPPORT_THRESHOLD)
    return found / len(claims)


@dataclass(frozen=True)
class Scores:
    faithfulness: float
    answer_relevance: float
    context_precision: float
    context_recall: float


def score_item(item: GoldenItem, output: RagOutput) -> Scores:
    return Scores(
        faithfulness=faithfulness(output.answer, output.contexts),
        answer_relevance=answer_relevance(item.question, output.answer),
        context_precision=context_precision(output.contexts, item.ground_truth),
        context_recall=context_recall(item.ground_truth, output.contexts),
    )
