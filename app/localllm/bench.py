"""Concurrency sweep: aggregate tokens/s and continuous-batching efficiency."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from app.localllm.client import CompletionResult


class StreamClient(Protocol):
    async def stream_chat(
        self, prompt: str, max_tokens: int = 128, temperature: float = 0.0
    ) -> CompletionResult: ...


@dataclass(frozen=True)
class BenchPoint:
    concurrency: int
    requests: int
    failures: int
    wall_s: float
    total_tokens: int
    aggregate_tps: float
    mean_stream_tps: float
    ttft_p50_ms: float
    ttft_p95_ms: float
    latency_p95_s: float
    speedup: float = 1.0  # aggregate_tps vs the first (lowest) concurrency level
    efficiency: float = 1.0  # speedup / ideal linear speedup


def _pct(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, math.ceil(p / 100 * len(ordered)) - 1))
    return ordered[rank]


def summarize(concurrency: int, results: Sequence[CompletionResult], wall_s: float) -> BenchPoint:
    good = [r for r in results if r.ok]
    tokens = sum(r.completion_tokens for r in good)
    stream_rates = [r.completion_tokens / r.total_s for r in good if r.total_s > 0]
    return BenchPoint(
        concurrency=concurrency,
        requests=len(results),
        failures=len(results) - len(good),
        wall_s=wall_s,
        total_tokens=tokens,
        aggregate_tps=tokens / wall_s if wall_s > 0 else 0.0,
        mean_stream_tps=sum(stream_rates) / len(stream_rates) if stream_rates else 0.0,
        ttft_p50_ms=_pct([r.ttft_s * 1000 for r in good], 50),
        ttft_p95_ms=_pct([r.ttft_s * 1000 for r in good], 95),
        latency_p95_s=_pct([r.total_s for r in good], 95),
    )


async def run_level(
    client: StreamClient,
    prompts: Sequence[str],
    concurrency: int,
    requests: int,
    max_tokens: int,
) -> BenchPoint:
    sem = asyncio.Semaphore(concurrency)

    async def one(i: int) -> CompletionResult:
        async with sem:
            return await client.stream_chat(prompts[i % len(prompts)], max_tokens=max_tokens)

    start = time.perf_counter()
    results = await asyncio.gather(*(one(i) for i in range(requests)))
    return summarize(concurrency, results, time.perf_counter() - start)


async def sweep(
    client: StreamClient,
    prompts: Sequence[str],
    levels: Sequence[int],
    requests_per_level: int = 32,
    max_tokens: int = 128,
    warmup: bool = True,
) -> list[BenchPoint]:
    if warmup:  # first call makes Ollama load the model; don't time that
        await client.stream_chat(prompts[0], max_tokens=8)
    points = [
        await run_level(client, prompts, c, max(requests_per_level, c), max_tokens) for c in levels
    ]
    base = points[0]
    out: list[BenchPoint] = []
    for p in points:
        speedup = p.aggregate_tps / base.aggregate_tps if base.aggregate_tps else 0.0
        ideal = p.concurrency / base.concurrency
        out.append(replace(p, speedup=speedup, efficiency=speedup / ideal))
    return out
