import hashlib
import logging
import os
from typing import Protocol

import redis.asyncio as aioredis

logger = logging.getLogger("ciksha")
DEFAULT_REDIS_URL = "redis://localhost:6379/0"


def normalize_text(text: str) -> str:
    return text.strip().lower()


def cache_key(text: str, model: str) -> str:
    """emb:<model>:sha256:<hash of trimmed, lowercased text>.
    The model is part of the key because different models give different vectors."""
    digest = hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()
    return f"emb:{model}:sha256:{digest}"


class CacheBackend(Protocol):
    async def get_many(self, keys: list[str]) -> list[str | None]: ...

    async def set_many(self, items: dict[str, str]) -> None: ...


class InMemoryCache:
    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    async def get_many(self, keys: list[str]) -> list[str | None]:
        return [self._data.get(key) for key in keys]

    async def set_many(self, items: dict[str, str]) -> None:
        self._data.update(items)


class RedisCache:
    """Fails open: if Redis is down, behave like an empty cache instead of crashing."""

    def __init__(self, url: str | None = None, ttl_seconds: int = 7 * 24 * 3600) -> None:
        resolved = url or os.getenv("REDIS_URL", DEFAULT_REDIS_URL)
        self._client = aioredis.Redis.from_url(resolved, decode_responses=True)
        self._ttl = ttl_seconds

    async def ping(self) -> bool:
        try:
            return bool(await self._client.ping())
        except Exception:
            return False

    async def get_many(self, keys: list[str]) -> list[str | None]:
        if not keys:
            return []
        try:
            values = await self._client.mget(keys)
        except Exception:
            logger.warning("Redis unavailable, treating as cache miss")
            return [None] * len(keys)
        return [value if isinstance(value, str) else None for value in values]

    async def set_many(self, items: dict[str, str]) -> None:
        if not items:
            return
        try:
            pipe = self._client.pipeline()
            for key, value in items.items():
                pipe.set(key, value, ex=self._ttl)
            await pipe.execute()
        except Exception:
            logger.warning("Redis unavailable, skipping cache write")

    async def close(self) -> None:
        await self._client.aclose()
