"""Static vs continuous batching, counted in decode steps."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class BatchSim:
    steps: int
    tokens: int
    slots: int

    @property
    def utilization(self) -> float:
        return self.tokens / (self.steps * self.slots) if self.steps else 0.0


def static_batching(lengths: Sequence[int], slots: int) -> BatchSim:
    """Each batch runs until its longest sequence finishes."""
    work = [n for n in lengths if n > 0]
    steps = sum(max(work[i : i + slots]) for i in range(0, len(work), slots))
    return BatchSim(steps, sum(work), slots)


def continuous_batching(lengths: Sequence[int], slots: int) -> BatchSim:
    """A finished sequence's slot is refilled on the very next step."""
    pending = deque(n for n in lengths if n > 0)
    tokens = sum(pending)
    active: list[int] = []
    steps = 0
    while pending or active:
        while pending and len(active) < slots:
            active.append(pending.popleft())
        steps += 1
        active = [r - 1 for r in active if r - 1 > 0]
    return BatchSim(steps, tokens, slots)
