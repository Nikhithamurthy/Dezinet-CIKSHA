"""Streaming client for any OpenAI-compatible server (Ollama, vLLM)."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType
from typing import Any

import httpx


@dataclass(frozen=True)
class CompletionResult:
    ok: bool
    ttft_s: float
    total_s: float
    completion_tokens: int
    error: str | None = None


class OpenAICompatClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "not-needed",
        timeout: float = 300.0,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._model = model
        self._clock = clock
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            timeout=timeout,
            transport=transport,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    async def __aenter__(self) -> OpenAICompatClient:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self._http.aclose()

    async def stream_chat(
        self, prompt: str, max_tokens: int = 128, temperature: float = 0.0
    ) -> CompletionResult:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        start = self._clock()
        ttft = 0.0
        chunks = 0
        usage_tokens: int | None = None
        try:
            async with self._http.stream("POST", "chat/completions", json=payload) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode(errors="replace")[:200]
                    err = f"HTTP {resp.status_code}: {body}"
                    return CompletionResult(False, 0.0, self._clock() - start, 0, err)
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    event = json.loads(data)
                    usage = event.get("usage")
                    if usage and usage.get("completion_tokens") is not None:
                        usage_tokens = int(usage["completion_tokens"])
                    for choice in event.get("choices", []):
                        if choice.get("delta", {}).get("content"):
                            if chunks == 0:
                                ttft = self._clock() - start
                            chunks += 1
        except (httpx.HTTPError, json.JSONDecodeError) as exc:
            return CompletionResult(False, 0.0, self._clock() - start, 0, repr(exc))
        tokens = usage_tokens if usage_tokens is not None else chunks
        return CompletionResult(True, ttft, self._clock() - start, tokens)
