import asyncio

import httpx
import pytest

from app.localllm.bench import sweep
from app.localllm.client import CompletionResult, OpenAICompatClient
from app.localllm.simulator import continuous_batching, static_batching
from app.localllm.vram import (
    MODELS,
    QUANTS,
    decode_ceiling_tps,
    estimate,
    kv_bytes_per_token,
    max_concurrent_sequences,
)


def test_kv_cache_per_token_matches_architecture() -> None:
    assert kv_bytes_per_token(MODELS["llama3-8b"]) == 131072  # 128 KiB
    assert kv_bytes_per_token(MODELS["qwen2.5-7b"]) == 57344  # GQA: only 4 KV heads


def test_quantization_shrinks_weights_and_4bit_8b_fits_consumer_gpu() -> None:
    spec = MODELS["llama3-8b"]
    fp16 = estimate(spec, QUANTS["fp16"], 4096, 1)
    awq = estimate(spec, QUANTS["awq-int4"], 4096, 1)
    assert fp16.weights_gib == pytest.approx(14.96, abs=0.05)
    assert awq.weights_gib < fp16.weights_gib / 3
    assert awq.total_gib < 8.0  # fits an 8 GB card at 4k context, 1 sequence


def test_quantization_buys_concurrency() -> None:
    spec = MODELS["llama3-8b"]
    fp16 = max_concurrent_sequences(spec, QUANTS["fp16"], 24, 8192)
    awq = max_concurrent_sequences(spec, QUANTS["awq-int4"], 24, 8192)
    assert awq > fp16 >= 1
    assert max_concurrent_sequences(spec, QUANTS["fp16"], 8, 8192) == 0


def test_decode_is_bandwidth_bound_and_batching_amortises_weights() -> None:
    spec = MODELS["llama3-8b"]
    fp16 = decode_ceiling_tps(spec, QUANTS["fp16"], 1008)
    awq = decode_ceiling_tps(spec, QUANTS["awq-int4"], 1008)
    assert fp16 == pytest.approx(62, rel=0.03)
    assert awq > 3 * fp16
    assert decode_ceiling_tps(spec, QUANTS["fp16"], 1008, batch=16) > 8 * fp16


def test_continuous_batching_beats_static() -> None:
    lengths = [2, 10, 2, 10]
    static = static_batching(lengths, slots=2)
    cont = continuous_batching(lengths, slots=2)
    assert static.steps == 20 and cont.steps == 14
    assert cont.utilization > static.utilization


def _sse(*events: str) -> bytes:
    return ("".join(f"data: {e}\n\n" for e in events) + "data: [DONE]\n\n").encode()


def test_client_parses_sse_stream_ttft_and_usage() -> None:
    body = _sse(
        '{"choices":[{"delta":{"role":"assistant"}}]}',
        '{"choices":[{"delta":{"content":"Hel"}}]}',
        '{"choices":[{"delta":{"content":"lo"}}]}',
        '{"choices":[],"usage":{"completion_tokens":2}}',
    )

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(200, content=body)

    ticks = iter([0.0, 0.1, 0.3])

    async def run() -> CompletionResult:
        async with OpenAICompatClient(
            "http://x/v1", "m", transport=httpx.MockTransport(handler), clock=lambda: next(ticks)
        ) as client:
            return await client.stream_chat("hi")

    r = asyncio.run(run())
    assert r.ok and r.completion_tokens == 2
    assert r.ttft_s == pytest.approx(0.1) and r.total_s == pytest.approx(0.3)


def test_client_reports_http_errors() -> None:
    transport = httpx.MockTransport(lambda req: httpx.Response(503, text="overloaded"))

    async def run() -> CompletionResult:
        async with OpenAICompatClient("http://x/v1", "m", transport=transport) as client:
            return await client.stream_chat("hi")

    r = asyncio.run(run())
    assert not r.ok and r.error is not None and "503" in r.error


class _BatchingFake:
    """Pretends the server batches perfectly: latency independent of load."""

    async def stream_chat(
        self, prompt: str, max_tokens: int = 128, temperature: float = 0.0
    ) -> CompletionResult:
        await asyncio.sleep(0.03)
        return CompletionResult(True, 0.005, 0.03, max_tokens)


def test_sweep_reports_speedup_against_baseline() -> None:
    points = asyncio.run(sweep(_BatchingFake(), ["p"], [1, 4], requests_per_level=8, max_tokens=10))
    assert points[0].speedup == pytest.approx(1.0)
    assert points[1].speedup > 2.0
    assert points[1].failures == 0
