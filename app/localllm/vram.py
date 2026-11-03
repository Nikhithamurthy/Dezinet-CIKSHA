"""GPU memory planning for locally served LLMs (weights + KV cache)."""

from __future__ import annotations

from dataclasses import dataclass

GIB = 1024**3
DEFAULT_OVERHEAD_GIB = 1.5  # CUDA context + activation buffers (rule of thumb)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    params_b: float
    layers: int
    kv_heads: int
    head_dim: int


@dataclass(frozen=True)
class QuantFormat:
    name: str
    bits_per_weight: float  # effective, including scales/zero-points
    runtime: str


MODELS: dict[str, ModelSpec] = {
    "llama3-8b": ModelSpec("llama3-8b", 8.03, 32, 8, 128),
    "mistral-7b": ModelSpec("mistral-7b", 7.25, 32, 8, 128),
    "qwen2.5-7b": ModelSpec("qwen2.5-7b", 7.61, 28, 4, 128),
}

QUANTS: dict[str, QuantFormat] = {
    "fp16": QuantFormat("fp16", 16.0, "vLLM / HF"),
    "int8": QuantFormat("int8", 8.0, "vLLM (W8A8) / bitsandbytes"),
    "gptq-int4": QuantFormat("gptq-int4", 4.25, "vLLM"),
    "awq-int4": QuantFormat("awq-int4", 4.25, "vLLM"),
    "gguf-q8_0": QuantFormat("gguf-q8_0", 8.5, "Ollama / llama.cpp"),
    "gguf-q5_k_m": QuantFormat("gguf-q5_k_m", 5.7, "Ollama / llama.cpp"),
    "gguf-q4_k_m": QuantFormat("gguf-q4_k_m", 4.85, "Ollama / llama.cpp"),
}


@dataclass(frozen=True)
class VramEstimate:
    weights_gib: float
    kv_gib: float
    overhead_gib: float

    @property
    def total_gib(self) -> float:
        return self.weights_gib + self.kv_gib + self.overhead_gib


def weight_bytes(spec: ModelSpec, quant: QuantFormat) -> float:
    return spec.params_b * 1e9 * quant.bits_per_weight / 8


def kv_bytes_per_token(spec: ModelSpec, kv_elem_bytes: int = 2) -> int:
    """K and V, for every layer: 2 * layers * kv_heads * head_dim * bytes."""
    return 2 * spec.layers * spec.kv_heads * spec.head_dim * kv_elem_bytes


def estimate(
    spec: ModelSpec,
    quant: QuantFormat,
    context: int,
    concurrency: int,
    kv_elem_bytes: int = 2,
    overhead_gib: float = DEFAULT_OVERHEAD_GIB,
) -> VramEstimate:
    kv = kv_bytes_per_token(spec, kv_elem_bytes) * context * concurrency
    return VramEstimate(weight_bytes(spec, quant) / GIB, kv / GIB, overhead_gib)


def max_concurrent_sequences(
    spec: ModelSpec,
    quant: QuantFormat,
    gpu_gib: float,
    context: int,
    utilization: float = 0.90,
    kv_elem_bytes: int = 2,
    overhead_gib: float = DEFAULT_OVERHEAD_GIB,
) -> int:
    """How many full-context sequences fit (what vLLM's block allocator is sizing)."""
    budget = gpu_gib * utilization - weight_bytes(spec, quant) / GIB - overhead_gib
    if budget <= 0:
        return 0
    per_seq = kv_bytes_per_token(spec, kv_elem_bytes) * context / GIB
    return int(budget // per_seq)


def decode_ceiling_tps(
    spec: ModelSpec,
    quant: QuantFormat,
    bandwidth_gb_s: float,
    batch: int = 1,
    context: int = 1024,
) -> float:
    """Upper bound on aggregate tokens/s if decoding is purely memory-bound.

    Each step streams the weights once (shared by the whole batch) plus every
    sequence's KV cache. This is why batching raises throughput: the weight
    traffic is amortised across the batch.
    """
    step_bytes = weight_bytes(spec, quant) + batch * context * kv_bytes_per_token(spec)
    return batch / (step_bytes / (bandwidth_gb_s * 1e9))
