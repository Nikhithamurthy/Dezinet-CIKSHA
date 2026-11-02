# pgvector Report

pgvector 0.8.7 on PostgreSQL 16 (Docker). 10,000 synthetic clustered vectors, 1536 dimensions, cosine distance, 50 queries, Recall@10 against exact brute force.

## 1. Index comparison
| Setting | Avg latency (ms) | QPS | Recall@10 |
|---|---|---|---|
| exact scan (no ANN index) | 65.2 | 15 | 100% |
| HNSW ef_search=10 | 47.8 | 21 | 96% |
| HNSW ef_search=40 | 46.5 | 21 | 100% |
| HNSW ef_search=100 | 45.8 | 22 | 100% |
| IVFFlat probes=1 | 45.8 | 22 | 98% |
| IVFFlat probes=5 | 48.5 | 21 | 100% |
| IVFFlat probes=20 | 56.7 | 18 | 100% |

## 2. Tenant filtering (HNSW ef_search=40, LIMIT 10)
| Mode | Tenant (share) | Avg rows returned | Avg latency (ms) |
|---|---|---|---|
| post-filter (HNSW only) | tiny (1%) | 10.0 | 70.0 |
| post-filter (HNSW only) | acme (~25%) | 8.8 | 49.3 |
| iterative scan | tiny (1%) | 10.0 | 68.8 |
| iterative scan | acme (~25%) | 10.0 | 49.6 |
| B-tree on tenant_id | tiny (1%) | 10.0 | 69.0 |
| B-tree on tenant_id | acme (~25%) | 10.0 | 404.2 |

## 3. Build cost
| Index | Build time (s) | Index size (MB) |
|---|---|---|
| HNSW (m=16, ef_construction=64) | 7.8 | 82 |
| IVFFlat (lists=50) | 1.4 | 82 |

## Findings
- HNSW recall rose from 96% to 100% as ef_search went from 10 to 40, with no visible latency cost at this size. IVFFlat recall rose from 98% to 100% between probes=1 and probes=5, and latency grew from 45.8 ms to 56.7 ms at probes=20.
- HNSW took 7.8 s to build against 1.4 s for IVFFlat. Both indexes were 82 MB, about 1.3x the 61 MB of raw vectors, below the 1.5x to 2x rule of thumb (9.2 to 12.3 GB for 1M vectors).
- Post-filtering returned 8.8 of 10 rows for the 25% tenant, showing the failure from naive post-filtering. Iterative scan restored 10 rows at the same latency. The B-tree also returned 10 rows but took 404 ms for that tenant.
- For the 1% tenant the planner chose a sequential scan, so the HNSW failure did not appear there. The 10 rows came from an exact scan.

## Limits
Synthetic data, only 10,000 rows, a single run with no repeats, latency dominated by per-query overhead at this scale, and the acme query plans were not inspected. Row-Level Security and ChromaDB are not covered.