import asyncio
import hashlib
import math
from collections.abc import AsyncIterator


async def stream_tokens(prompt: str, max_tokens: int) -> AsyncIterator[str]:
    """Simulates an LLM emitting one token at a time (non-blocking)."""
    text = f"You asked: {prompt}. This is a simulated streaming answer from CIKSHA."
    for word in text.split()[:max_tokens]:
        await asyncio.sleep(0.05)  # stands in for network/model latency
        yield word + " "


def _embed_one(text: str) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()  # 32 bytes -> 32 dims
    raw = [byte / 255 for byte in digest]
    norm = math.sqrt(sum(x * x for x in raw)) or 1.0
    return [x / norm for x in raw]  # unit vector, ready for Task 04


async def embed_texts(texts: list[str]) -> list[list[float]]:
    # CPU-bound work goes to a thread so the event loop is never blocked.
    return await asyncio.to_thread(lambda: [_embed_one(t) for t in texts])
