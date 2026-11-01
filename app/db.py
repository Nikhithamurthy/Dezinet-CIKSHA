import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


class DBSession:
    async def ping(self) -> str:
        await asyncio.sleep(0)
        return "ok"


class DatabasePool:
    def __init__(self, size: int = 5) -> None:
        self.size = size
        self._semaphore = asyncio.Semaphore(size)  # at most `size` sessions at once

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[DBSession]:
        async with self._semaphore:
            yield DBSession()

    async def close(self) -> None:
        return None
