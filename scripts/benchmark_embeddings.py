import asyncio
import sys
import time

from app.embeddings.cache import CacheBackend, InMemoryCache, RedisCache
from app.embeddings.embedder import get_embedder
from app.embeddings.metrics import (
    cosine_similarity,
    dot_product,
    euclidean_distance,
    l2_normalize,
)
from app.embeddings.service import EmbeddingService

PAIRS = [
    ("How do I reset my password?", "I forgot my login password"),
    ("How do I reset my password?", "What is the refund policy for annual plans?"),
    ("Dezinet Teams subscription price", "How much does the Team plan cost?"),
    ("Dezinet Teams subscription price", "Office hours of the support team"),
]


async def main() -> None:
    provider = sys.argv[1] if len(sys.argv) > 1 else "mock"
    embedder = get_embedder(provider)
    await embedder.embed(["warm up"])  # load the model before timing anything
    print(f"Embedder: {embedder.name} ({embedder.dimensions} dimensions)\n")

    print("## 1. Similarity metrics on unit-normalized vectors")
    print("| Text A | Text B | Cosine | Dot | L2 distance |")
    print("|---|---|---|---|---|")
    for text_a, text_b in PAIRS:
        va, vb = await embedder.embed([text_a, text_b])
        va, vb = l2_normalize(va), l2_normalize(vb)
        print(
            f"| {text_a[:32]} | {text_b[:32]} | {cosine_similarity(va, vb):.3f} "
            f"| {dot_product(va, vb):.3f} | {euclidean_distance(va, vb):.3f} |"
        )

    va, vb = await embedder.embed([PAIRS[0][0], PAIRS[0][1]])
    scaled = [3 * x for x in va]
    print("\n## 2. Scaling vector A by 3 (first pair)")
    print(f"- cosine: {cosine_similarity(va, vb):.3f} -> {cosine_similarity(scaled, vb):.3f}")
    print(f"- dot:    {dot_product(va, vb):.3f} -> {dot_product(scaled, vb):.3f}")
    print(f"- L2:     {euclidean_distance(va, vb):.3f} -> {euclidean_distance(scaled, vb):.3f}")

    redis_cache = RedisCache()
    backend: CacheBackend
    if await redis_cache.ping():
        backend, label = redis_cache, "Redis"
    else:
        backend, label = InMemoryCache(), "in-memory (Redis not reachable)"
    service = EmbeddingService(embedder, backend, batch_size=16)
    stamp = int(time.time())
    texts = [f"benchmark document number {i} run {stamp}" for i in range(40)]

    start = time.perf_counter()
    await service.embed(texts)
    cold_ms = (time.perf_counter() - start) * 1000
    cold_calls = service.stats.api_calls
    start = time.perf_counter()
    await service.embed(texts)
    warm_ms = (time.perf_counter() - start) * 1000

    print(f"\n## 3. Embedding cache ({label}), 40 texts, batch size 16")
    print("| Run | Embedder calls | Time (ms) |")
    print("|---|---|---|")
    print(f"| cold | {cold_calls} | {cold_ms:.1f} |")
    print(f"| warm | {service.stats.api_calls - cold_calls} | {warm_ms:.1f} |")
    print(f"Overall hit rate: {service.stats.hit_rate:.0%}")
    await redis_cache.close()

    print("\n## 4. Storage per 1 million float32 vectors")
    print("| Dimensions | GB |")
    print("|---|---|")
    for dims in (384, 512, 768, 1536, 3072):
        print(f"| {dims} | {dims * 4 * 1_000_000 / 1e9:.2f} |")
    print("Truncating 1536 -> 512 saves 66.7% of storage.")


if __name__ == "__main__":
    asyncio.run(main())
