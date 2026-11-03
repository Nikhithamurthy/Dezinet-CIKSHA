import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, Self


@dataclass(frozen=True)
class Message:
    role: str  # "system", "user" or "assistant"
    content: str

    def to_json(self) -> str:
        return json.dumps({"role": self.role, "content": self.content})

    @classmethod
    def from_json(cls, raw: str | bytes) -> Self:
        data = json.loads(raw)
        return cls(role=data["role"], content=data["content"])


class ChatStore(Protocol):
    async def append(self, session_id: str, messages: Sequence[Message]) -> None: ...
    async def history(self, session_id: str) -> list[Message]: ...
    async def get_summary(self, session_id: str) -> tuple[str, int]: ...
    async def set_summary(self, session_id: str, summary: str, covered: int) -> None: ...
    async def clear(self, session_id: str) -> None: ...


class InMemoryStore:
    """Same behaviour as RedisChatStore (sliding TTL, message cap) for tests and demos."""

    def __init__(
        self,
        ttl_seconds: int = 3600,
        max_messages: int = 200,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._ttl = ttl_seconds
        self._max = max_messages
        self._clock = clock
        self._messages: dict[str, list[Message]] = {}
        self._summaries: dict[str, tuple[str, int]] = {}
        self._expires: dict[str, float] = {}

    def _alive(self, session_id: str) -> bool:
        deadline = self._expires.get(session_id)
        if deadline is None:
            return False
        if self._clock() >= deadline:
            self._drop(session_id)
            return False
        return True

    def _touch(self, session_id: str) -> None:
        self._expires[session_id] = self._clock() + self._ttl

    def _drop(self, session_id: str) -> None:
        self._messages.pop(session_id, None)
        self._summaries.pop(session_id, None)
        self._expires.pop(session_id, None)

    async def append(self, session_id: str, messages: Sequence[Message]) -> None:
        self._alive(session_id)  # purges the session if it already expired
        bucket = self._messages.setdefault(session_id, [])
        bucket.extend(messages)
        del bucket[: -self._max]
        self._touch(session_id)

    async def history(self, session_id: str) -> list[Message]:
        if not self._alive(session_id):
            return []
        self._touch(session_id)
        return list(self._messages.get(session_id, []))

    async def get_summary(self, session_id: str) -> tuple[str, int]:
        if not self._alive(session_id):
            return "", 0
        return self._summaries.get(session_id, ("", 0))

    async def set_summary(self, session_id: str, summary: str, covered: int) -> None:
        self._summaries[session_id] = (summary, covered)
        self._touch(session_id)

    async def clear(self, session_id: str) -> None:
        self._drop(session_id)


class RedisChatStore:
    """History in a Redis list, summary in a Redis hash. Every key gets a TTL that
    resets on activity. Create the client with decode_responses=True."""

    def __init__(
        self,
        client: Any,
        ttl_seconds: int = 3600,
        max_messages: int = 200,
        prefix: str = "chat",
    ) -> None:
        self._client = client
        self._ttl = ttl_seconds
        self._max = max_messages
        self._prefix = prefix

    def _key(self, session_id: str, part: str) -> str:
        return f"{self._prefix}:{session_id}:{part}"

    async def append(self, session_id: str, messages: Sequence[Message]) -> None:
        if not messages:
            return
        key = self._key(session_id, "msgs")
        pipe = self._client.pipeline()
        pipe.rpush(key, *[m.to_json() for m in messages])
        pipe.ltrim(key, -self._max, -1)
        pipe.expire(key, self._ttl)
        pipe.expire(self._key(session_id, "summary"), self._ttl)
        await pipe.execute()

    async def history(self, session_id: str) -> list[Message]:
        key = self._key(session_id, "msgs")
        pipe = self._client.pipeline()
        pipe.lrange(key, 0, -1)
        pipe.expire(key, self._ttl)
        raw, _ = await pipe.execute()
        return [Message.from_json(item) for item in raw]

    async def get_summary(self, session_id: str) -> tuple[str, int]:
        data = await self._client.hgetall(self._key(session_id, "summary"))
        if not data:
            return "", 0
        return str(data["text"]), int(data["covered"])

    async def set_summary(self, session_id: str, summary: str, covered: int) -> None:
        key = self._key(session_id, "summary")
        pipe = self._client.pipeline()
        pipe.hset(key, mapping={"text": summary, "covered": covered})
        pipe.expire(key, self._ttl)
        await pipe.execute()

    async def clear(self, session_id: str) -> None:
        await self._client.delete(self._key(session_id, "msgs"), self._key(session_id, "summary"))
