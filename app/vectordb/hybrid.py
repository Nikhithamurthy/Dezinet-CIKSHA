from collections.abc import Sequence
from typing import Any

from app.vectordb.db import keyword_search, search

RRF_K = 60


def rrf_fuse(
    ranked_lists: Sequence[Sequence[dict[str, Any]]],
    k: int = RRF_K,
    top_n: int | None = None,
    key: str = "id",
) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion: score = sum over lists of 1 / (k + rank), rank starting at 1."""
    scores: dict[Any, float] = {}
    items: dict[Any, dict[str, Any]] = {}

    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            ident = item[key]
            scores[ident] = scores.get(ident, 0.0) + 1.0 / (k + rank)
            items.setdefault(ident, item)

    fused = [
        {**items[ident], "rrf_score": score}
        for ident, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    ]
    return fused[:top_n] if top_n is not None else fused


async def hybrid_candidates(
    conn: Any,
    tenant_id: str,
    query: str,
    query_embedding: list[float],
    pool: int = 50,
) -> list[dict[str, Any]]:
    dense = await search(conn, tenant_id, query_embedding, k=pool)
    sparse = await keyword_search(conn, tenant_id, query, k=pool)
    return rrf_fuse([dense, sparse], top_n=pool)
