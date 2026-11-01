from functools import lru_cache
from typing import Any

import tiktoken


@lru_cache
def _encoding() -> Any:
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None  # offline on first run: fall back to ~4 chars per token


def count_tokens(text: str) -> int:
    encoding = _encoding()
    if encoding is None:
        return max(1, len(text) // 4)
    return len(encoding.encode(text))
