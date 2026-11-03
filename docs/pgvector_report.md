# pgvector Benchmark

pgvector 0.8.7 on PostgreSQL 16 (Docker). 10,000 synthetic clustered vectors, 1536 dimensions, 50 queries, cosine distance, top 10.

## 1. Index comparison
| Setting | Avg latency (ms) | QPS | Recall@10 |
|---|---|---|---|
| exact scan (no ANN index) | 59.2 | 17 | 100% |
| HNSW ef_search=10 | 47.5 | 21 | 92% |
| HNSW ef_search=40 | 48.0 | 21 | 98% |
| HNSW ef_search=100 | 46.6 | 21 | 100% |
| IVFFlat probes=1 | 47.7 | 21 | 91% |
| IVFFlat probes=5 | 50.5 | 20 | 100% |
| IVFFlat probes=20 | 57.1 | 18 | 100% |

## 2. Metadata filtering (HNSW, ef_search=40, LIMIT 10)
| Mode | Tenant (share of rows) | Avg rows returned (of 10) | Avg latency (ms) |
|---|---|---|---|
| post-filter (HNSW only) | tiny (1%) | 10.0 | 70.1 |
| post-filter (HNSW only) | acme (~25%) | 8.7 | 49.2 |
| iterative scan | tiny (1%) | 10.0 | 68.5 |
| iterative scan | acme (~25%) | 10.0 | 50.9 |
| B-tree on tenant_id | tiny (1%) | 10.0 | 70.8 |
| B-tree on tenant_id | acme (~25%) | 10.0 | 406.2 |

## 3. Build cost
| Index | Build time (s) | Index size (MB) |
|---|---|---|
| HNSW (m=16, ef_construction=64) | 7.6 | 82 |
| IVFFlat (lists=50) | 1.2 | 82 |

## Findings
- Recall rises with the search effort setting for both indexes: HNSW 92% to 100% (ef_search 10 to 100), IVFFlat 91% to 100% (probes 1 to 5).
- Latency barely changed (about 47 ms) and the exact scan was only 59 ms. At 10,000 rows the time is dominated by client and Docker overhead, so this run does not show an HNSW speed advantage. Only the recall differences are reliable.
- HNSW took 7.6 s to build against 1.2 s for IVFFlat.
- Post-filtering returned only 8.7 of 10 rows for the 25% tenant, which shows the failure in interview question 4. Iterative scan and a B-tree both restored 10 rows, but the B-tree raised latency from about 50 ms to 406 ms.
- For the 1% tenant, PostgreSQL chose a sequential scan with the filter applied first, so it returned 10 rows and the failure did not appear.
- RAM sizing for 1M x 1536 float32 vectors is 6.14 GB raw, or 9.2 to 12.3 GB with HNSW at 1.5x to 2x (a calculation, not measured).

## Limits
Synthetic data, only 10,000 rows, no Row-Level Security yet, and ChromaDB is not covered.