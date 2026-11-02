# Embeddings Report

Embedder: mock-bow-256 (hashed bag-of-words, offline). Cache: Redis.

## Findings
- On unit vectors, cosine and dot product are identical (0.548), and L2 distance is a function of them.
- Scaling a vector by 3 leaves cosine unchanged (0.548) but changes dot (1.643) and L2 (2.591), so cosine is safer when magnitude should not matter.
- The embedding cache cut embedder calls from 3 to 0 on the second run (hit rate 50% over both runs).
- Limit: the mock embedder only matches shared words, so "Teams" vs "Team" scored -0.167. A real model (BGE or OpenAI) would capture this.
- Storage: 1M float32 vectors need 6.14 GB at 1536 dimensions and 2.05 GB at 512, a 66.7% saving.