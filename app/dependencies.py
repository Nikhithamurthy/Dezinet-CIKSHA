import secrets
import time
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request

from app.config import Settings, get_settings
from app.db import DatabasePool, DBSession


async def verify_api_key(
    settings: Annotated[Settings, Depends(get_settings)],
    x_api_key: Annotated[str | None, Header()] = None,
) -> str:
    valid = [k.strip() for k in settings.api_keys.split(",") if k.strip()]
    if x_api_key is None or not any(secrets.compare_digest(x_api_key, k) for k in valid):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return x_api_key


class RateLimiter:
    """Sliding-window limiter. In-memory for now; Task 08 moves it to Redis."""

    def __init__(self, window_seconds: float = 60.0) -> None:
        self._window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str, limit: int) -> bool:
        now = time.monotonic()
        hits = self._hits[key]
        while hits and now - hits[0] >= self._window:
            hits.popleft()
        if len(hits) >= limit:
            return False
        hits.append(now)
        return True

    def reset(self) -> None:
        self._hits.clear()


limiter = RateLimiter()


async def rate_limit(
    api_key: Annotated[str, Depends(verify_api_key)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> None:
    if not limiter.allow(api_key, settings.rate_limit_per_minute):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Try again in a minute.",
            headers={"Retry-After": "60"},
        )


async def get_db_session(request: Request) -> AsyncIterator[DBSession]:
    pool: DatabasePool = request.app.state.db_pool
    async with pool.acquire() as session:
        yield session  # automatically released after the request
