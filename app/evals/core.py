"""Shared types and text helpers for the evaluation suite."""

import re
from dataclasses import dataclass
from typing import Protocol

_TOKEN = re.compile(r"[a-z0-9]+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
STOPWORDS = frozenset(
    "a an the of is are was were to for and or in on at by with that this it its be as from "
    "what which who whom when where how does do did can could should would will i you your my "
    "we our me please".split()
)


@dataclass(frozen=True)
class Document:
    title: str
    text: str


@dataclass(frozen=True)
class GoldenItem:
    question: str
    ground_truth: str
    source: str


@dataclass(frozen=True)
class RagOutput:
    answer: str
    contexts: tuple[str, ...]
    flags: tuple[str, ...] = ()


class RagSystem(Protocol):
    async def run(self, question: str) -> RagOutput: ...


def token_set(text: str) -> frozenset[str]:
    words: list[str] = _TOKEN.findall(text.lower())
    return frozenset(w for w in words if w not in STOPWORDS)


def support(claim: str, context: str) -> float:
    """Share of the claim's content words that appear in the context (0..1)."""
    claim_tokens = token_set(claim)
    if not claim_tokens:
        return 1.0
    return len(claim_tokens & token_set(context)) / len(claim_tokens)


def sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_END.split(text.strip()) if part.strip()]
