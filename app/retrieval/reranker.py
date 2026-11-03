import asyncio
from collections.abc import Sequence
from typing import Any, Protocol

from app.retrieval.bm25 import tokenize

_STOPWORDS = {
    "a", "an", "and", "are", "can", "do", "does", "for", "how", "i", "in", "is", "it",
    "my", "of", "on", "or", "the", "to", "what", "when", "who", "you", "your",
}  # fmt: skip


class Reranker(Protocol):
    name: str

    async def rerank(
        self, query: str, docs: Sequence[tuple[str, str]], top_k: int
    ) -> list[tuple[str, float]]: ...


class LexicalReranker:
    """Offline stand-in for a cross-encoder. It scores each (query, document) PAIR
    jointly: how many query terms the document contains, plus a bonus for query
    word pairs that appear next to each other. It is NOT a learned model."""

    def __init__(self) -> None:
        self.name = "lexical-overlap"

    def _score(self, query: str, text: str) -> float:
        terms = [t for t in tokenize(query) if t not in _STOPWORDS]
        if not terms:
            return 0.0
        doc_tokens = tokenize(text)
        doc_set = set(doc_tokens)
        coverage = sum(1 for t in terms if t in doc_set) / len(terms)
        query_pairs = set(zip(terms, terms[1:], strict=False))
        doc_pairs = set(zip(doc_tokens, doc_tokens[1:], strict=False))
        pair_score = len(query_pairs & doc_pairs) / len(query_pairs) if query_pairs else 0.0
        return coverage + 0.5 * pair_score

    async def rerank(
        self, query: str, docs: Sequence[tuple[str, str]], top_k: int
    ) -> list[tuple[str, float]]:
        scored = [(doc_id, self._score(query, text)) for doc_id, text in docs]
        scored.sort(key=lambda item: -item[1])
        return scored[:top_k]


class CrossEncoderReranker:
    """Real cross-encoder (bge-reranker-base). Needs: uv add sentence-transformers."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        self.name = model_name
        self._model: Any = None

    def _score(self, query: str, docs: list[tuple[str, str]]) -> list[float]:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.name)
        scores = self._model.predict([(query, text) for _, text in docs])
        return [float(s) for s in scores]

    async def rerank(
        self, query: str, docs: Sequence[tuple[str, str]], top_k: int
    ) -> list[tuple[str, float]]:
        items = list(docs)
        scores = await asyncio.to_thread(self._score, query, items)
        ranked = [(doc_id, score) for (doc_id, _), score in zip(items, scores, strict=True)]
        ranked.sort(key=lambda item: -item[1])
        return ranked[:top_k]
