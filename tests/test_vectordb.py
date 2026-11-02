import asyncio
from typing import Any

import asyncpg
import pytest

from app.embeddings.embedder import MockEmbedder
from app.vectordb.db import (
    DIMENSIONS,
    ChunkRow,
    add_chunks,
    add_document,
    connect,
    delete_tenant,
    ensure_tenant,
    get_dsn,
    init_schema,
    search,
    vector_literal,
)


def _db_available() -> bool:
    async def probe() -> bool:
        try:
            conn = await asyncpg.connect(get_dsn(), timeout=3)
        except Exception:
            return False
        await conn.close()
        return True

    return asyncio.run(probe())


needs_db = pytest.mark.skipif(
    not _db_available(), reason="Postgres not running (docker start ciksha-pg)"
)


async def _seed(conn: Any, tenant: str, texts: list[str]) -> None:
    vectors = await MockEmbedder(DIMENSIONS).embed(texts)
    await ensure_tenant(conn, tenant)
    doc_id = await add_document(conn, tenant, f"{tenant}.md")
    rows = [
        ChunkRow(i, text, {"source": f"{tenant}.md"}, vec)
        for i, (text, vec) in enumerate(zip(texts, vectors, strict=True))
    ]
    await add_chunks(conn, tenant, doc_id, rows)


async def _clean(conn: Any, tenants: list[str]) -> None:
    for tenant in tenants:
        await delete_tenant(conn, tenant)


def test_vector_literal_checks_dimensions() -> None:
    with pytest.raises(ValueError):
        vector_literal([0.1, 0.2])
    assert vector_literal([0.5] * DIMENSIONS).startswith("[0.5,0.5")


@needs_db
def test_schema_has_hnsw_index_with_tuned_params() -> None:
    async def scenario() -> str:
        conn = await connect()
        try:
            await init_schema(conn)
            return str(
                await conn.fetchval(
                    "SELECT indexdef FROM pg_indexes WHERE indexname = 'chunks_embedding_hnsw'"
                )
            )
        finally:
            await conn.close()

    definition = asyncio.run(scenario())
    assert "hnsw" in definition
    assert "ef_construction" in definition


@needs_db
def test_tenant_isolation() -> None:
    async def scenario() -> tuple[list[str], list[str]]:
        conn = await connect()
        try:
            await init_schema(conn)
            await _clean(conn, ["test_a", "test_b"])
            await _seed(conn, "test_a", ["refund policy for annual plans"])
            await _seed(conn, "test_b", ["refund policy for annual plans", "secret of tenant b"])
            query = (await MockEmbedder(DIMENSIONS).embed(["refund policy"]))[0]
            hits_a = [h["content"] for h in await search(conn, "test_a", query, k=5)]
            hits_b = [h["content"] for h in await search(conn, "test_b", query, k=5)]
            return hits_a, hits_b
        finally:
            await _clean(conn, ["test_a", "test_b"])
            await conn.close()

    hits_a, hits_b = asyncio.run(scenario())
    assert hits_a == ["refund policy for annual plans"]
    assert "secret of tenant b" not in hits_a
    assert len(hits_b) == 2


@needs_db
def test_cosine_search_ranks_closest_first() -> None:
    async def scenario() -> list[dict[str, Any]]:
        conn = await connect()
        try:
            await init_schema(conn)
            await _clean(conn, ["test_c"])
            await _seed(
                conn, "test_c", ["refund policy for annual plans", "weather forecast hyderabad"]
            )
            query = (await MockEmbedder(DIMENSIONS).embed(["annual plans refund policy"]))[0]
            return await search(conn, "test_c", query, k=2)
        finally:
            await _clean(conn, ["test_c"])
            await conn.close()

    hits = asyncio.run(scenario())
    assert hits[0]["content"] == "refund policy for annual plans"
    assert hits[0]["distance"] < hits[1]["distance"]


@needs_db
def test_all_three_distance_operators() -> None:
    async def scenario() -> dict[str, float]:
        conn = await connect()
        try:
            await init_schema(conn)
            await _clean(conn, ["test_d"])
            await _seed(conn, "test_d", ["refund policy for annual plans"])
            query = (await MockEmbedder(DIMENSIONS).embed(["refund policy for annual plans"]))[0]
            out: dict[str, float] = {}
            for metric in ("cosine", "inner_product", "l2"):
                out[metric] = (await search(conn, "test_d", query, k=1, metric=metric))[0][
                    "distance"
                ]
            return out
        finally:
            await _clean(conn, ["test_d"])
            await conn.close()

    result = asyncio.run(scenario())
    assert result["cosine"] == pytest.approx(0.0, abs=1e-4)
    assert result["l2"] == pytest.approx(0.0, abs=1e-3)
    assert result["inner_product"] == pytest.approx(-1.0, abs=1e-4)  # <#> is negative dot
