import asyncio
import random
from collections.abc import Awaitable, Callable, Iterator, Sequence
from typing import TypeVar

T = TypeVar("T")
SleepFn = Callable[[float], Awaitable[None]]


async def default_sleep(delay: float) -> None:
    await asyncio.sleep(delay)


def batched(items: Sequence[T], size: int) -> Iterator[Sequence[T]]:
    if size < 1:
        raise ValueError("size must be >= 1")
    for start in range(0, len(items), size):
        yield items[start : start + size]


async def retry_with_backoff(
    func: Callable[[], Awaitable[T]],
    *,
    retries: int = 4,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    jitter: bool = True,
    sleep: SleepFn = default_sleep,
) -> T:
    """Retry with exponential backoff: 0.5s, 1s, 2s, 4s... capped, with optional jitter."""
    attempt = 0
    while True:
        try:
            return await func()
        except Exception:
            if attempt >= retries:
                raise
            delay = min(max_delay, base_delay * (2**attempt))
            if jitter:
                delay *= 0.5 + random.random() / 2
            await sleep(delay)
            attempt += 1
