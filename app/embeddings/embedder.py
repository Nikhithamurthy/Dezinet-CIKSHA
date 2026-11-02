import asyncio
import hashlib
import os
import re
from collections.abc import Sequence
from typing import Any, Protocol

from app.embeddings.metrics import l2_normalize


class Embedder(Protocol):
    name: str
    dimensions: int

    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class MockEmbedder:
    """Offline hashed bag-of-words. Texts that share words get similar vectors."""

    def __init__(self, dimensions: int = 256) -> None:
        self.name = f"mock-bow-{dimensions}"
        self.dimensions = dimensions

    def _one(self, text: str) -> list[float]:
        vec = [0.0] * self.dimensions
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vec[index] += 1.0 if digest[4] % 2 == 0 else -1.0
        return l2_normalize(vec)

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._one(text) for text in texts]


class LocalEmbedder:
    """Open-source BGE model via sentence-transformers (384 dimensions)."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", dimensions: int = 384) -> None:
        self.name = model_name
        self.dimensions = dimensions
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.name)
        return self._model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._load().encode(texts, normalize_embeddings=True)
        return [[float(x) for x in row] for row in vectors]

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        # Model inference is CPU-heavy: keep it off the event loop.
        return await asyncio.to_thread(self._encode, list(texts))


class OpenAIEmbedder:
    """OpenAI text-embedding-3-small (1536d) or -large (3072d). Needs OPENAI_API_KEY."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dimensions: int | None = None,
        api_key: str | None = None,
    ) -> None:
        from openai import AsyncOpenAI

        self._model = model
        self._requested = dimensions
        self.name = model if dimensions is None else f"{model}-{dimensions}"
        self.dimensions = dimensions or (3072 if "large" in model else 1536)
        self._client = AsyncOpenAI(api_key=api_key)

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        kwargs: dict[str, Any] = {}
        if self._requested:
            kwargs["dimensions"] = self._requested  # Matryoshka shortening
        response = await self._client.embeddings.create(
            model=self._model, input=list(texts), **kwargs
        )
        return [[float(x) for x in item.embedding] for item in response.data]


def get_embedder(kind: str = "mock", dimensions: int | None = None) -> Embedder:
    if kind == "mock":
        return MockEmbedder()
    if kind == "local":
        return LocalEmbedder()
    if kind == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("Set OPENAI_API_KEY to use the OpenAI embedder")
        return OpenAIEmbedder(dimensions=dimensions)
    raise ValueError(f"Unknown embedder '{kind}'. Use mock, local or openai")
