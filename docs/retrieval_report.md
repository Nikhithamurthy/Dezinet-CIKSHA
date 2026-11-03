# Retrieval Benchmark

Embedder: mock-bow-256 (hashed bag-of-words). Reranker: lexical-overlap (a stand-in, not a learned cross-encoder). 30 documents, 21 queries, K=5.

## Overall
| Pipeline | MRR@5 | Hit rate@5 | Precision@5 | Recall@5 | Avg latency (ms) |
|---|---|---|---|---|---|
| vector only | 0.805 | 90% | 0.181 | 90% | 0.45 |
| BM25 only | 0.952 | 95% | 0.190 | 95% | 0.02 |
| hybrid (RRF) | 0.905 | 95% | 0.190 | 95% | 0.66 |
| hybrid + rerank | 0.952 | 95% | 0.190 | 95% | 0.97 |

## MRR@5 by query type
| Query type | vector only | BM25 only | hybrid (RRF) | hybrid + rerank |
|---|---|---|---|---|
| exact | 1.000 | 1.000 | 1.000 | 1.000 |
| keyword | 0.838 | 1.000 | 1.000 | 1.000 |
| paraphrase | 0.300 | 0.750 | 0.500 | 0.750 |

Not found by hybrid + rerank: "can I get my money back".

## Findings
- All pipelines scored 1.000 on exact IDs. The mock embedder hashes tokens, so it also matches rare identifiers; this run therefore does not demonstrate vector search failing on exact codes.
- BM25 beat vector-only on keyword queries (1.000 vs 0.838).
- On this small corpus hybrid RRF (0.905) did not beat BM25 alone (0.952); the reranker raised hybrid from 0.905 to 0.952.
- Paraphrase scores are low and driven by accidental word overlap, because the mock embedder has no semantic understanding.
- Latency is under 1 ms because everything is in memory with 30 documents; it does not test the brief's latency targets.

## Limits
Tiny corpus, mock embedder, lexical reranker. A real run needs sentence-transformers (embedder and bge-reranker-base). The Postgres full-text plus pgvector version is in app/vectordb/hybrid.py.