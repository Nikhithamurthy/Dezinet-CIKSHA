"""Usage:
uv run python -m app.localllm.cli vram --model llama3-8b --gpu-gib 24
uv run python -m app.localllm.cli bench --base-url http://localhost:11434/v1 --model llama3
"""

from __future__ import annotations

import argparse
import asyncio

from app.localllm.bench import BenchPoint, sweep
from app.localllm.client import OpenAICompatClient
from app.localllm.vram import (
    MODELS,
    QUANTS,
    decode_ceiling_tps,
    estimate,
    max_concurrent_sequences,
)

PROMPTS = [
    "Explain PagedAttention in simple terms.",
    "Write a Python function that merges two sorted lists.",
    "Summarise the trade-offs between AWQ and GPTQ quantization.",
    "Why is LLM decoding memory-bandwidth bound?",
]


def _vram(args: argparse.Namespace) -> None:
    spec = MODELS[args.model]
    print(
        f"{spec.name} | ctx={args.context} | concurrency={args.concurrency} "
        f"| GPU={args.gpu_gib} GiB"
    )
    print(
        f"{'format':<12}{'weights':>9}{'kv':>8}{'total':>8}"
        f"{'fits':>6}{'max seqs':>10}{'tok/s cap':>11}"
    )
    for q in QUANTS.values():
        e = estimate(spec, q, args.context, args.concurrency)
        fits = "yes" if e.total_gib <= args.gpu_gib * 0.9 else "NO"
        seqs = max_concurrent_sequences(spec, q, args.gpu_gib, args.context)
        cap = decode_ceiling_tps(spec, q, args.bandwidth)
        print(
            f"{q.name:<12}{e.weights_gib:>8.1f}G{e.kv_gib:>7.1f}G{e.total_gib:>7.1f}G"
            f"{fits:>6}{seqs:>10}{cap:>11.0f}"
        )


def _print_bench(points: list[BenchPoint]) -> None:
    print(
        f"{'conc':>5}{'agg tok/s':>11}{'per-req':>9}{'ttft p50':>10}{'ttft p95':>10}"
        f"{'speedup':>9}{'effic.':>8}{'fail':>6}"
    )
    for p in points:
        print(
            f"{p.concurrency:>5}{p.aggregate_tps:>11.1f}{p.mean_stream_tps:>9.1f}"
            f"{p.ttft_p50_ms:>9.0f}m{p.ttft_p95_ms:>9.0f}m{p.speedup:>8.2f}x"
            f"{p.efficiency:>8.2f}{p.failures:>6}"
        )


async def _bench(args: argparse.Namespace) -> None:
    levels = [int(x) for x in args.levels.split(",")]
    async with OpenAICompatClient(args.base_url, args.model) as client:
        _print_bench(await sweep(client, PROMPTS, levels, args.requests, args.max_tokens))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="app.localllm.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("vram")
    v.add_argument("--model", choices=list(MODELS), default="llama3-8b")
    v.add_argument("--gpu-gib", type=float, default=24.0)
    v.add_argument("--context", type=int, default=8192)
    v.add_argument("--concurrency", type=int, default=8)
    v.add_argument("--bandwidth", type=float, default=1008.0, help="GB/s (RTX 4090 = 1008)")
    b = sub.add_parser("bench")
    b.add_argument("--base-url", default="http://localhost:8000/v1")
    b.add_argument("--model", required=True)
    b.add_argument("--levels", default="1,2,4,8,16")
    b.add_argument("--requests", type=int, default=32)
    b.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args(argv)
    if args.cmd == "vram":
        _vram(args)
    else:
        asyncio.run(_bench(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
