from functools import lru_cache
from typing import Any

from app.memory.store import Message

MESSAGE_OVERHEAD = 4  # rough per-message framing cost in chat formats


@lru_cache(maxsize=1)
def _encoder() -> Any:
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # missing package, or no network for the first vocab download
        return None


def count_tokens(text: str) -> int:
    encoder = _encoder()
    if encoder is None:
        return max(1, len(text) // 4)
    return len(encoder.encode(text))


def message_tokens(message: Message) -> int:
    return count_tokens(message.content) + MESSAGE_OVERHEAD
