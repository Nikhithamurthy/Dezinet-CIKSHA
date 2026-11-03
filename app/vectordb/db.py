import json
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import asyncpg

DEFAULT_DSN = "postgresql://ciksha:ciksha@localhost:5432/ciksha"
DIMENSIONS = 1536

# <=> cosine distance, <#> NEGATIVE inner product, <-> L2 distance
OPERATORS = {"cosine": "<=>", "inner_product": "<#>", "l2": "<->"}

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS tenants (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES tenants (id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id BIGSERIAL PRIMARY KEY,
    tenant_id TEXT NOT NULL REFERENCES tenants (id) ON DELETE CASCADE,
    document_id BIGINT NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding vector(1536) NOT NULL,
    content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

CREATE INDEX IF NOT EXISTS chunks_tenant_idx ON chunks (tenant_id);
CREATE INDEX IF NOT EXISTS chunks_content_fts ON chunks USING gin (content_tsv);
CREATE INDEX IF NOT EXISTS chunks_metadata_gin ON chunks USING gin (metadata);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw ON chunks
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
"""

INSERT_CHUNK_SQL = """
INSERT INTO chunks (tenant_id, document_id, chunk_index, content, metadata, embedding)
VALUES ($1, $2, $3, $4, $5::text::jsonb, $6::text::vector)
"""


@dataclass
class ChunkRow:
    chunk_index: int
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: Sequence[float] = field(default_factory=list)


def get_dsn() -> str:
    return os.getenv("DATABASE_URL", DEFAULT_DSN)


async def connect(dsn: str | None = None) -> Any:
    return await asyncpg.connect(dsn or get_dsn())


def vector_literal(vector: Sequence[float]) -> str:
    """pgvector text format: [0.1,0.2,...]. Rejects wrong dimensions early."""
    if len(vector) != DIMENSIONS:
        raise ValueError(f"expected {DIMENSIONS} dimensions, got {len(vector)}")
    return "[" + ",".join(f"{x:.6g}" for x in vector) + "]"


async def init_schema(conn: Any) -> None:
    await conn.execute(SCHEMA_SQL)


async def ensure_tenant(conn: Any, tenant_id: str, name: str | None = None) -> None:
    await conn.execute(
        "INSERT INTO tenants (id, name) VALUES ($1, $2) ON CONFLICT (id) DO NOTHING",
        tenant_id,
        name or tenant_id,
    )


async def delete_tenant(conn: Any, tenant_id: str) -> None:
    await conn.execute("DELETE FROM tenants WHERE id = $1", tenant_id)  # cascades


async def add_document(conn: Any, tenant_id: str, filename: str) -> int:
    new_id = await conn.fetchval(
        "INSERT INTO documents (tenant_id, filename) VALUES ($1, $2) RETURNING id",
        tenant_id,
        filename,
    )
    return int(new_id)


async def add_chunks(conn: Any, tenant_id: str, document_id: int, rows: Sequence[ChunkRow]) -> None:
    await conn.executemany(
        INSERT_CHUNK_SQL,
        [
            (
                tenant_id,
                document_id,
                row.chunk_index,
                row.content,
                json.dumps(row.metadata),
                vector_literal(row.embedding),
            )
            for row in rows
        ],
    )


async def search(
    conn: Any,
    tenant_id: str,
    query: Sequence[float],
    k: int = 5,
    metric: str = "cosine",
) -> list[dict[str, Any]]:
    """Tenant-filtered nearest-neighbour search. tenant_id is mandatory, so a
    caller cannot accidentally search across tenants."""
    if not tenant_id:
        raise ValueError("tenant_id is required")
    if metric not in OPERATORS:
        raise ValueError(f"metric must be one of {sorted(OPERATORS)}")
    if not 1 <= k <= 100:
        raise ValueError("k must be between 1 and 100")
    op = OPERATORS[metric]  # whitelisted above, so safe to put in the SQL text
    sql = (
        f"SELECT id, document_id, content, metadata, "
        f"embedding {op} $1::text::vector AS distance "
        f"FROM chunks WHERE tenant_id = $2 "
        f"ORDER BY embedding {op} $1::text::vector LIMIT $3"
    )
    rows = await conn.fetch(sql, vector_literal(query), tenant_id, k)
    return [
        {
            "id": r["id"],
            "document_id": r["document_id"],
            "content": r["content"],
            "metadata": json.loads(r["metadata"]),
            "distance": float(r["distance"]),
        }
        for r in rows
    ]


async def keyword_search(
    conn: Any,
    tenant_id: str,
    query: str,
    k: int = 5,
) -> list[dict[str, Any]]:
    """Full-text keyword search, tenant-filtered. Uses websearch_to_tsquery so
    user input such as ERR-5034 or quoted phrases cannot break the query."""
    if not tenant_id:
        raise ValueError("tenant_id is required")
    if not 1 <= k <= 100:
        raise ValueError("k must be between 1 and 100")
    sql = (
        "SELECT id, document_id, content, metadata, "
        "ts_rank_cd(content_tsv, websearch_to_tsquery('english', $1)) AS score "
        "FROM chunks "
        "WHERE tenant_id = $2 AND content_tsv @@ websearch_to_tsquery('english', $1) "
        "ORDER BY score DESC, id LIMIT $3"
    )
    rows = await conn.fetch(sql, query, tenant_id, k)
    return [
        {
            "id": r["id"],
            "document_id": r["document_id"],
            "content": r["content"],
            "metadata": json.loads(r["metadata"]),
            "score": float(r["score"]),
        }
        for r in rows
    ]
