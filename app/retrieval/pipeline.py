from collections.abc import Sequence
from typing import Self

from app.embeddings.embedder import Embedder
from app.retrieval.bm25 import BM25Index
from app.retrieval.dense import DenseIndex
from app.retrieval.eval_set import Doc
from app.retrieval.reranker import Reranker
from app.retrieval.rrf import reciprocal_rank_fusion


class HybridRetriever:
    """Stage 1: dense + BM25 candidates fused with RRF. Stage 2: rerank to top-k."""

    def __init__(
        self,
        docs: Sequence[Doc],
        vectors: Sequence[Sequence[float]],
        embedder: Embedder,
        reranker: Reranker | None = None,
        *,
        rrf_k: int = 60,
        candidates: int = 50,
    ) -> None:
        self._docs = {doc.id: doc for doc in docs}
        self._dense = DenseIndex([doc.id for doc in docs], vectors)
        self._bm25 = BM25Index([(doc.id, doc.text) for doc in docs])
        self._embedder = embedder
        self._reranker = reranker
        self._rrf_k = rrf_k
        self._candidates = candidates

    @classmethod
    async def build(
        cls,
        docs: Sequence[Doc],
        embedder: Embedder,
        reranker: Reranker | None = None,
        *,
        rrf_k: int = 60,
        candidates: int = 50,
    ) -> Self:
        vectors = await embedder.embed([doc.text for doc in docs])
        return cls(docs, vectors, embedder, reranker, rrf_k=rrf_k, candidates=candidates)

    async def vector_search(self, query: str, k: int = 5) -> list[str]:
        return [doc_id for doc_id, _ in await self._dense_ranking(query, k)]

    async def bm25_search(self, query: str, k: int = 5) -> list[str]:
        return [doc_id for doc_id, _ in self._bm25.search(query, k)]

    async def hybrid(self, query: str, k: int = 5) -> list[str]:
        return [doc_id for doc_id, _ in await self._fused(query)][:k]

    async def hybrid_rerank(self, query: str, k: int = 5) -> list[str]:
        fused = await self._fused(query)
        if self._reranker is None:
            return [doc_id for doc_id, _ in fused][:k]
        pairs = [(doc_id, self._docs[doc_id].text) for doc_id, _ in fused]
        ranked = await self._reranker.rerank(query, pairs, k)
        return [doc_id for doc_id, _ in ranked]

    async def _dense_ranking(self, query: str, k: int) -> list[tuple[str, float]]:
        vector = (await self._embedder.embed([query]))[0]
        return self._dense.search(vector, k)

    async def _fused(self, query: str) -> list[tuple[str, float]]:
        dense = [d for d, _ in await self._dense_ranking(query, self._candidates)]
        sparse = [d for d, _ in self._bm25.search(query, self._candidates)]
        return reciprocal_rank_fusion([dense, sparse], k=self._rrf_k)[: self._candidates]
