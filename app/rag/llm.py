import re
from typing import Protocol

from app.rag.context import parse_context
from app.rag.grounding import STOPWORDS, support_score
from app.rag.prompts import NO_ANSWER
from app.retrieval.bm25 import tokenize

# Tiny hand-written table used by the mock to vary queries. A real LLM does this itself.
SYNONYMS = {
    "cost": "price",
    "price": "cost",
    "reset": "change",
    "leave": "vacation",
    "discount": "reduction",
    "support": "help",
}
MOCK_MIN_SENTENCE_SUPPORT = 0.5


class LLM(Protocol):
    name: str

    async def complete(self, system: str, user: str) -> str: ...


class MockLLM:
    """Offline stand-in. It does NOT understand language: it expands queries from a
    small synonym table and answers by copying the best-matching context sentences."""

    def __init__(self) -> None:
        self.name = "mock-extractive"

    async def complete(self, system: str, user: str) -> str:
        if system.startswith("TASK: EXPAND"):
            return self._expand(system, user)
        if system.startswith("TASK: HYDE"):
            return self._hyde(user)
        if system.startswith("TASK: ANSWER"):
            return self._answer(user)
        raise ValueError("MockLLM does not understand this prompt")

    def _keywords(self, text: str) -> list[str]:
        return [t for t in tokenize(text) if t not in STOPWORDS]

    def _expand(self, system: str, user: str) -> str:
        match = re.search(r"Write (\d+) alternative", system)
        n = int(match.group(1)) if match else 3
        keywords = self._keywords(user)
        if not keywords:
            return user
        swapped = [SYNONYMS.get(t, t) for t in keywords]
        candidates = [
            " ".join(keywords),
            " ".join(swapped),
            "details about " + " ".join(keywords),
        ]
        return "\n".join(list(dict.fromkeys(candidates))[:n])

    def _hyde(self, user: str) -> str:
        keywords = " ".join(self._keywords(user))
        if not keywords:
            return user
        return f"{keywords}. This document explains {keywords}."

    def _answer(self, user: str) -> str:
        chunks = parse_context(user)
        questions = re.findall(r"^Question: (.*)$", user, re.MULTILINE)
        question = questions[-1] if questions else ""
        scored: list[tuple[float, str, str]] = []
        for chunk in chunks:
            for sentence in re.split(r"(?<=[.!?])\s+", chunk.text):
                text = sentence.strip()
                if text:
                    scored.append((support_score(question, text), chunk.chunk_id, text))
        scored.sort(key=lambda item: -item[0])
        best = [item for item in scored[:2] if item[0] >= MOCK_MIN_SENTENCE_SUPPORT]
        if not best:
            return NO_ANSWER
        return " ".join(f"{text} [{chunk_id}]" for _, chunk_id, text in best)


class OpenAILLM:
    """Real model via the OpenAI API. Needs OPENAI_API_KEY."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        from openai import AsyncOpenAI

        self.name = model
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key)

    async def complete(self, system: str, user: str) -> str:
        response = await self._client.chat.completions.create(
            model=self._model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return str(response.choices[0].message.content or "")


def get_llm(kind: str = "mock") -> LLM:
    if kind == "mock":
        return MockLLM()
    if kind == "openai":
        return OpenAILLM()
    raise ValueError(f"Unknown LLM '{kind}'. Use mock or openai")
