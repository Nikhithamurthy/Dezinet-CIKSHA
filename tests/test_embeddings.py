import asyncio
import math
from collections.abc import Sequence

import pytest

from app.embeddings.cache import InMemoryCache, cache_key
from app.embeddings.embedder import MockEmbedder
from app.embeddings.metrics import (
    cosine_similarity,
    dot_product,
    euclidean_distance,
    l2_norm,
    l2_normalize,
    truncate_and_normalize,
)
from app.embeddings.retry import retry_with_backoff
from app.embeddings.service import EmbeddingService


class CountingEmbedder:
    name = "counting"
    dimensions = 4

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(t)), 1.0, 0.0, 0.0] for t in texts]


def test_normalize_gives_unit_length() -> None:
    assert math.isclose(l2_norm(l2_normalize([3.0, 4.0, 12.0])), 1.0)


def test_dot_equals_cosine_on_unit_vectors() -> None:
    a = l2_normalize([1.0, 2.0, 3.0])
    b = l2_normalize([4.0, -1.0, 2.0])
    assert math.isclose(dot_product(a, b), cosine_similarity(a, b))
    # For unit vectors: L2 distance squared = 2 - 2 * cosine
    assert math.isclose(euclidean_distance(a, b) ** 2, 2 - 2 * cosine_similarity(a, b))


def test_cosine_ignores_magnitude_but_dot_and_l2_do_not() -> None:
    a = [1.0, 2.0, 3.0]
    b = [2.0, 4.0, 6.0]
    assert math.isclose(cosine_similarity(a, b), 1.0)
    assert dot_product(a, b) == 28.0
    assert euclidean_distance(a, b) > 0


def test_euclidean_distance() -> None:
    assert euclidean_distance([0.0, 0.0], [3.0, 4.0]) == 5.0


def test_matryoshka_truncation_renormalizes() -> None:
    assert truncate_and_normalize([3.0, 4.0, 12.0], 2) == pytest.approx([0.6, 0.8])
    with pytest.raises(ValueError):
        truncate_and_normalize([1.0, 2.0], 5)


def test_cache_key_normalizes_text_and_includes_model() -> None:
    assert cache_key("  Hello ", "m1") == cache_key("hello", "m1")
    assert cache_key("hello", "m1") != cache_key("hello", "m2")
    assert cache_key("hello", "m1").startswith("emb:m1:sha256:")


def test_mock_embedder_ranks_similar_text_higher() -> None:
    texts = [
        "refund policy for subscriptions",
        "subscription refund policy",
        "weather forecast hyderabad",
    ]
    query, close, far = asyncio.run(MockEmbedder().embed(texts))
    assert cosine_similarity(query, close) > cosine_similarity(query, far)


def test_second_call_is_served_from_cache() -> None:
    embedder = CountingEmbedder()
    service = EmbeddingService(embedder, InMemoryCache())
    first = asyncio.run(service.embed(["alpha", "beta", "gamma"]))
    second = asyncio.run(service.embed(["alpha", "beta", "gamma"]))
    assert first == second
    assert len(embedder.calls) == 1
    assert service.stats.cache_hits == 3
    assert service.stats.cache_misses == 3


def test_duplicates_and_case_variants_are_embedded_once() -> None:
    embedder = CountingEmbedder()
    service = EmbeddingService(embedder, InMemoryCache())
    vectors = asyncio.run(service.embed(["Hello", "hello ", "Hello"]))
    assert embedder.calls == [["Hello"]]
    assert vectors[0] == vectors[1] == vectors[2]


def test_batching_splits_requests() -> None:
    embedder = CountingEmbedder()
    service = EmbeddingService(embedder, None, batch_size=2)
    asyncio.run(service.embed(["a", "b", "c", "d", "e"]))
    assert [len(call) for call in embedder.calls] == [2, 2, 1]
    assert service.stats.api_calls == 3


def test_retry_backs_off_exponentially_then_succeeds() -> None:
    delays: list[float] = []
    attempts = 0

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    async def flaky() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("rate limited")
        return "ok"

    result = asyncio.run(
        retry_with_backoff(flaky, retries=4, base_delay=0.5, jitter=False, sleep=fake_sleep)
    )
    assert result == "ok"
    assert delays == [0.5, 1.0]


def test_retry_gives_up_after_max_retries() -> None:
    delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    async def always_fails() -> str:
        raise RuntimeError("down")

    with pytest.raises(RuntimeError):
        asyncio.run(retry_with_backoff(always_fails, retries=2, jitter=False, sleep=fake_sleep))
    assert delays == [0.5, 1.0]
