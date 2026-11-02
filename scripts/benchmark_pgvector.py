import asyncio
import time
from typing import Any

import numpy as np

from app.vectordb.db import DIMENSIONS, connect, vector_literal

N = 10_000
TINY_ROWS = 100
BIG_TENANTS = ["acme", "globex", "initech", "umbrella"]
QUERIES = 50
K = 10
SEARCH = "SELECT id FROM bench_vectors ORDER BY embedding <=> $1::text::vector LIMIT 10"


def make_data(rng: np.random.Generator) -> Any:
    centers = rng.standard_normal((40, DIMENSIONS), dtype=np.float32)
    labels = rng.integers(0, 40, size=N)
    noise = rng.standard_normal((N, DIMENSIONS), dtype=np.float32)
    data = centers[labels] + 0.7 * noise
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    return data


async def measure(conn: Any, sql: str, qlits: list[str], truth: Any) -> tuple[float, float, float]:
    await conn.fetch(sql, qlits[0])  # warm up
    found: list[list[int]] = []
    start = time.perf_counter()
    for lit in qlits:
        rows = await conn.fetch(sql, lit)
        found.append([int(r["id"]) for r in rows])
    elapsed = time.perf_counter() - start
    hits = sum(len(set(f) & set(map(int, t))) for f, t in zip(found, truth, strict=True))
    recall = hits / (len(qlits) * K)
    return elapsed / len(qlits) * 1000, len(qlits) / elapsed, recall


def show(label: str, result: tuple[float, float, float]) -> None:
    ms, qps, recall = result
    print(f"| {label} | {ms:.1f} | {qps:.0f} | {recall:.0%} |")


def filtered_sql(tenant: str) -> str:
    # tenant comes from the constants above, never from user input
    return (
        f"SELECT id FROM bench_vectors WHERE tenant_id = '{tenant}' "
        "ORDER BY embedding <=> $1::text::vector LIMIT 10"
    )


async def filtered(conn: Any, tenant: str, qlits: list[str]) -> tuple[float, float]:
    counts: list[int] = []
    start = time.perf_counter()
    for lit in qlits:
        counts.append(len(await conn.fetch(filtered_sql(tenant), lit)))
    ms = (time.perf_counter() - start) / len(qlits) * 1000
    return sum(counts) / len(counts), ms


async def plan(conn: Any, tenant: str, lit: str) -> str:
    rows = await conn.fetch("EXPLAIN " + filtered_sql(tenant), lit)
    lines = [str(r[0]).strip() for r in rows]
    scans = [line for line in lines if "Scan" in line]
    return scans[0] if scans else lines[0]


async def main() -> None:
    rng = np.random.default_rng(42)
    data = make_data(rng)
    tenants = ["tiny"] * TINY_ROWS
    tenants += [BIG_TENANTS[i] for i in rng.integers(0, 4, size=N - TINY_ROWS)]
    picks = rng.choice(np.arange(TINY_ROWS, N), size=QUERIES, replace=False)
    noise = rng.standard_normal((QUERIES, DIMENSIONS), dtype=np.float32) / np.sqrt(DIMENSIONS)
    queries = data[picks] + 0.5 * noise
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    truth = np.argsort(-(queries @ data.T), axis=1)[:, :K]
    qlits = [vector_literal(q) for q in queries]

    conn = await connect()
    version = await conn.fetchval("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
    print(f"pgvector {version} | {N} synthetic clustered vectors x {DIMENSIONS} dims\n")
    await conn.execute("DROP TABLE IF EXISTS bench_vectors")
    await conn.execute(
        "CREATE TABLE bench_vectors (id INT PRIMARY KEY, tenant_id TEXT NOT NULL, "
        "embedding vector(1536) NOT NULL)"
    )
    insert = "INSERT INTO bench_vectors VALUES ($1, $2, $3::text::vector)"
    for start in range(0, N, 500):
        batch = [
            (i, tenants[i], vector_literal(data[i])) for i in range(start, min(start + 500, N))
        ]
        await conn.executemany(insert, batch)
    await conn.execute("ANALYZE bench_vectors")
    await conn.execute("SET maintenance_work_mem = '512MB'")
    table_mb = await conn.fetchval("SELECT pg_relation_size('bench_vectors')") / 1e6
    print(f"Table size: {table_mb:.0f} MB (vectors are stored out of line, so this is small)\n")

    header = "| Setting | Avg latency (ms) | QPS | Recall@10 |\n|---|---|---|---|"
    print("## 1. Index comparison (unfiltered cosine search, 50 queries)")
    print(header)
    show("exact scan (no ANN index)", await measure(conn, SEARCH, qlits, truth))

    t0 = time.perf_counter()
    await conn.execute(
        "CREATE INDEX bench_hnsw ON bench_vectors USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    )
    hnsw_build = time.perf_counter() - t0
    hnsw_mb = await conn.fetchval("SELECT pg_relation_size('bench_hnsw')") / 1e6
    for ef in (10, 40, 100):
        await conn.execute(f"SET hnsw.ef_search = {ef}")
        show(f"HNSW ef_search={ef}", await measure(conn, SEARCH, qlits, truth))

    print("\n## 2. Metadata filtering with HNSW (ef_search=40, LIMIT 10)")
    print("| Mode | Tenant (share of rows) | Avg rows returned (of 10) | Avg latency (ms) |")
    print("|---|---|---|---|")
    await conn.execute("SET hnsw.ef_search = 40")
    for tenant, share in (("tiny", "1%"), ("acme", "~25%")):
        rows_avg, ms = await filtered(conn, tenant, qlits)
        print(f"| post-filter (HNSW only) | {tenant} ({share}) | {rows_avg:.1f} | {ms:.1f} |")
    print(f"\nPlan for tiny: {await plan(conn, 'tiny', qlits[0])}\n")

    try:
        await conn.execute("SET hnsw.iterative_scan = relaxed_order")
        for tenant, share in (("tiny", "1%"), ("acme", "~25%")):
            rows_avg, ms = await filtered(conn, tenant, qlits)
            print(f"| iterative scan | {tenant} ({share}) | {rows_avg:.1f} | {ms:.1f} |")
        await conn.execute("SET hnsw.iterative_scan = off")
    except Exception as exc:
        print(f"| iterative scan | not supported by this pgvector version ({exc}) | - | - |")

    await conn.execute("CREATE INDEX bench_tenant_idx ON bench_vectors (tenant_id)")
    await conn.execute("ANALYZE bench_vectors")
    for tenant, share in (("tiny", "1%"), ("acme", "~25%")):
        rows_avg, ms = await filtered(conn, tenant, qlits)
        print(f"| B-tree on tenant_id | {tenant} ({share}) | {rows_avg:.1f} | {ms:.1f} |")
    print(f"\nPlan for tiny after the B-tree: {await plan(conn, 'tiny', qlits[0])}")

    await conn.execute("DROP INDEX bench_hnsw")
    await conn.execute("DROP INDEX bench_tenant_idx")
    t0 = time.perf_counter()
    await conn.execute(
        "CREATE INDEX bench_ivf ON bench_vectors USING ivfflat (embedding vector_cosine_ops) "
        "WITH (lists = 50)"
    )
    ivf_build = time.perf_counter() - t0
    ivf_mb = await conn.fetchval("SELECT pg_relation_size('bench_ivf')") / 1e6
    print("\n## 3. IVFFlat (lists = 50)")
    print(header)
    for probes in (1, 5, 20):
        await conn.execute(f"SET ivfflat.probes = {probes}")
        show(f"IVFFlat probes={probes}", await measure(conn, SEARCH, qlits, truth))

    print("\n## 4. Build cost")
    print("| Index | Build time (s) | Index size (MB) |")
    print("|---|---|---|")
    print(f"| HNSW (m=16, ef_construction=64) | {hnsw_build:.1f} | {hnsw_mb:.0f} |")
    print(f"| IVFFlat (lists=50) | {ivf_build:.1f} | {ivf_mb:.0f} |")
    raw_gb = 1_000_000 * DIMENSIONS * 4 / 1e9
    print(f"\n## 5. RAM sizing for 1M x {DIMENSIONS} float32 vectors")
    low, high = raw_gb * 1.5, raw_gb * 2
    print(f"Raw vectors: {raw_gb:.2f} GB. HNSW at 1.5x to 2x: {low:.1f} to {high:.1f} GB.")
    await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
