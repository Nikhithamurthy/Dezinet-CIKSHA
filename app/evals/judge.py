"""LLM-as-a-judge with a rubric, plus a pairwise judge that cancels position bias."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.rag.llm import LLM
from app.tools.structured import extract_structured

RUBRIC = (
    "Score 1-5 against the ground truth. 5: correct, complete, and nothing unsupported. "
    "3: partly correct or missing a detail. 1: wrong or unrelated. "
    "Do not reward length; a short correct answer beats a long one with extras."
)


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score: int = Field(ge=1, le=5)
    reason: str = Field(min_length=1, max_length=300)


class Preference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    winner: Literal["first", "second", "tie"]


async def judge_answer(llm: LLM, question: str, answer: str, ground_truth: str) -> Verdict:
    prompt = (
        f"{RUBRIC}\n\nQuestion: {question}\nGround truth: {ground_truth}\nAnswer: {answer}\n"
        'Return {"score": <1-5>, "reason": "<one sentence>"}.'
    )
    return await extract_structured(llm, Verdict, prompt)


async def _prefer(
    llm: LLM, question: str, first: str, second: str, truth: str
) -> Literal["first", "second", "tie"]:
    prompt = (
        "Which answer is closer to the ground truth? Ignore length and style.\n"
        f"Question: {question}\nGround truth: {truth}\n"
        f"First: {first}\nSecond: {second}\n"
        'Return {"winner": "first" | "second" | "tie"}.'
    )
    return (await extract_structured(llm, Preference, prompt)).winner


async def judge_pair(
    llm: LLM, question: str, a: str, b: str, ground_truth: str
) -> Literal["a", "b", "tie"]:
    """Ask twice with the order swapped. Only a verdict that survives the swap counts, so a
    judge that always prefers the first position produces a tie, not a winner."""
    forward = await _prefer(llm, question, a, b, ground_truth)
    backward = await _prefer(llm, question, b, a, ground_truth)
    if forward == "first" and backward == "second":
        return "a"
    if forward == "second" and backward == "first":
        return "b"
    return "tie"
