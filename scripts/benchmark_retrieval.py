import asyncio
import sys
import time
from collections import defaultdict
from collections.abc import Awaitable, Callable

from app.embeddings.embedder import get_embedder
from app.retrieval.eval_set import load_docs, load_queries
from app.retrieval.metrics import hit_at_k, mean, precision_at_k, recall_at_k, reciprocal_rank
from app.retrieval.pipeline import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker, LexicalReranker, Reranker

K = 5
Search = Callable[[str, int], Awaitable[list[str]]]


async def main() -> None:
    provider = sys.argv[1] if len(sys.argv) > 1 else "mock"
    rerank_kind = sys.argv[2] if len(sys.argv) > 2 else "lexical"
    reranker: Reranker = CrossEncoderReranker() if rerank_kind == "cross" else LexicalReranker()
    docs, queries = load_docs(), load_queries()
    embedder = get_embedder(provider)
    retriever = await HybridRetriever.build(docs, embedder, reranker)
    await retriever.hybrid_rerank("warm up", K)
    print(f"Embedder: {embedder.name} | Reranker: {reranker.name}")
    print(f"{len(docs)} documents, {len(queries)} queries, K={K}\n")

    strategies: dict[str, Search] = {
        "vector only": retriever.vector_search,
        "BM25 only": retriever.bm25_search,
        "hybrid (RRF)": retriever.hybrid,
        "hybrid + rerank": retriever.hybrid_rerank,
    }
    print("## 1. Overall")
    print("| Pipeline | MRR@5 | Hit rate@5 | Precision@5 | Recall@5 | Avg latency (ms) |")
    print("|---|---|---|---|---|---|")
    by_category: dict[str, dict[str, list[float]]] = {}
    misses: list[str] = []
    for name, search in strategies.items():
        rr, hits, prec, rec = [], [], [], []
        categories: dict[str, list[float]] = defaultdict(list)
        start = time.perf_counter()
        for q in queries:
            ranked = await search(q.query, K)
            relevant = set(q.relevant)
            value = reciprocal_rank(ranked, relevant, K)
            rr.append(value)
            hits.append(hit_at_k(ranked, relevant, K))
            prec.append(precision_at_k(ranked, relevant, K))
            rec.append(recall_at_k(ranked, relevant, K))
            categories[q.category].append(value)
            if name == "hybrid + rerank" and value == 0.0:
                misses.append(f"{q.category}: {q.query}")
        ms = (time.perf_counter() - start) / len(queries) * 1000
        by_category[name] = categories
        print(
            f"| {name} | {mean(rr):.3f} | {mean(hits):.0%} | {mean(prec):.3f} "
            f"| {mean(rec):.0%} | {ms:.2f} |"
        )

    print("\n## 2. MRR@5 by query type")
    names = list(strategies)
    print("| Query type | " + " | ".join(names) + " |")
    print("|---|" + "---|" * len(names))
    for category in ("exact", "keyword", "paraphrase"):
        row = " | ".join(f"{mean(by_category[n][category]):.3f}" for n in names)
        print(f"| {category} | {row} |")

    print("\n## 3. Not found in the top 5 by hybrid + rerank")
    for item in misses or ["(none)"]:
        print(f"- {item}")


if __name__ == "__main__":
    asyncio.run(main())
