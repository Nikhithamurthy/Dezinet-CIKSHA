import json
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial

from app.embeddings.cache import CacheBackend, cache_key
from app.embeddings.embedder import Embedder
from app.embeddings.retry import SleepFn, batched, default_sleep, retry_with_backoff


@dataclass
class EmbeddingStats:
    cache_hits: int = 0
    cache_misses: int = 0
    api_calls: int = 0
    texts_embedded: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total else 0.0


class EmbeddingService:
    """Cache lookup -> dedupe -> batch -> retry with backoff -> cache write."""

    def __init__(
        self,
        embedder: Embedder,
        cache: CacheBackend | None = None,
        *,
        batch_size: int = 64,
        retries: int = 4,
        base_delay: float = 0.5,
        sleep: SleepFn = default_sleep,
    ) -> None:
        self._embedder = embedder
        self._cache = cache
        self._batch_size = batch_size
        self._retries = retries
        self._base_delay = base_delay
        self._sleep = sleep
        self.stats = EmbeddingStats()

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        keys = [cache_key(text, self._embedder.name) for text in texts]
        if self._cache is not None:
            cached = await self._cache.get_many(keys)
        else:
            cached = [None] * len(keys)

        vectors: dict[str, list[float]] = {}
        missing: dict[str, str] = {}
        for key, text, raw in zip(keys, texts, cached, strict=True):
            if raw is not None:
                vectors[key] = json.loads(raw)
                self.stats.cache_hits += 1
            else:
                self.stats.cache_misses += 1
                missing.setdefault(key, text)  # duplicates are embedded once

        new_items: dict[str, str] = {}
        for batch_keys in batched(list(missing), self._batch_size):
            batch_texts = [missing[key] for key in batch_keys]
            result = await retry_with_backoff(
                partial(self._embedder.embed, batch_texts),
                retries=self._retries,
                base_delay=self._base_delay,
                sleep=self._sleep,
            )
            self.stats.api_calls += 1
            self.stats.texts_embedded += len(batch_texts)
            for key, vec in zip(batch_keys, result, strict=True):
                vectors[key] = vec
                new_items[key] = json.dumps(vec)

        if self._cache is not None and new_items:
            await self._cache.set_many(new_items)
        return [vectors[key] for key in keys]
