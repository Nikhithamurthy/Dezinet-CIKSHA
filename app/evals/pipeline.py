"""A small RAG pipeline with optional guardrails. Swap in your real retriever and LLM."""

from collections.abc import Sequence
from typing import Protocol

from app.evals.core import Document, RagOutput, sentences, token_set
from app.safety.guard import BLOCKED_MESSAGE, Guard
from app.safety.pii import Redaction


class Retriever(Protocol):
    async def search(self, question: str, k: int) -> list[str]: ...


class Answerer(Protocol):
    async def answer(self, question: str, contexts: Sequence[str]) -> str: ...


class LexicalRetriever:
    """Sentence-level chunks ranked by content-word overlap with the question."""

    def __init__(self, docs: Sequence[Document]) -> None:
        chunks = [sentence for doc in docs for sentence in sentences(doc.text)]
        self._chunks = [(chunk, token_set(chunk)) for chunk in chunks]

    async def search(self, question: str, k: int) -> list[str]:
        wanted = token_set(question)
        scored = [(-len(wanted & tokens), i) for i, (_, tokens) in enumerate(self._chunks)]
        return [self._chunks[i][0] for score, i in sorted(scored)[:k] if score < 0]


class ExtractiveAnswerer:
    """Offline stand-in for the LLM: answers with the best-ranked chunk."""

    async def answer(self, question: str, contexts: Sequence[str]) -> str:
        return contexts[0] if contexts else "I do not know."


class RagPipeline:
    def __init__(
        self,
        retriever: Retriever,
        answerer: Answerer,
        *,
        k: int = 3,
        guard: Guard | None = None,
    ) -> None:
        self._retriever = retriever
        self._answerer = answerer
        self._k = k
        self._guard = guard

    async def run(self, question: str) -> RagOutput:
        guard = self._guard
        flags: list[str] = []
        redaction: Redaction | None = None
        query = question
        if guard is not None:
            decision = guard.check_input(question)
            if not decision.allowed:
                return RagOutput(BLOCKED_MESSAGE, (), decision.reasons)
            query, redaction = decision.text, decision.redaction
            flags.extend(decision.reasons)
        contexts = await self._retriever.search(query, self._k)
        if guard is not None:
            contexts, dropped = guard.filter_contexts(contexts)
            flags.extend(dropped)
        answer = await self._answerer.answer(query, contexts)
        if guard is not None:
            checked = guard.check_output(answer, redaction)
            answer = checked.text
            flags.extend(checked.reasons)
        return RagOutput(answer, tuple(contexts), tuple(flags))
