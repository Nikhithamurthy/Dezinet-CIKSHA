import math
import time
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass


def percentile(values: Sequence[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(p / 100 * len(ordered)) - 1))
    return ordered[index]


@dataclass(frozen=True)
class StreamMetrics:
    ttft_ms: float  # request start -> first token
    itl_mean_ms: float  # mean gap between tokens
    itl_p95_ms: float
    total_ms: float  # request start -> last token
    tokens: int
    tokens_per_s: float


class StreamMeter:
    """Call on_token() as each token is sent. TTFT is measured from `started`
    (the request start), which is what the user actually waits."""

    def __init__(
        self,
        started: float | None = None,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._clock = clock
        self._start = clock() if started is None else started
        self._stamps: list[float] = []

    def on_token(self) -> None:
        self._stamps.append(self._clock())

    def summary(self) -> StreamMetrics:
        if not self._stamps:
            return StreamMetrics(0.0, 0.0, 0.0, 0.0, 0, 0.0)
        ttft = (self._stamps[0] - self._start) * 1000
        gaps = [(b - a) * 1000 for a, b in zip(self._stamps, self._stamps[1:], strict=False)]
        total = (self._stamps[-1] - self._start) * 1000
        span_s = self._stamps[-1] - self._stamps[0]
        tokens = len(self._stamps)
        rate = (tokens - 1) / span_s if tokens > 1 and span_s > 0 else 0.0
        mean_gap = sum(gaps) / len(gaps) if gaps else 0.0
        return StreamMetrics(ttft, mean_gap, percentile(gaps, 95), total, tokens, rate)


async def measure_stream(tokens: AsyncIterator[str]) -> tuple[str, StreamMetrics]:
    meter = StreamMeter()
    parts: list[str] = []
    async for token in tokens:
        meter.on_token()
        parts.append(token)
    return "".join(parts), meter.summary()
