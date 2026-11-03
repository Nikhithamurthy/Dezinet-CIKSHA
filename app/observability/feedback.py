import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from app.observability.tracing import TraceStore

RATINGS = ("up", "down", "correction")


class UnknownTrace(Exception):
    pass


@dataclass(frozen=True)
class FeedbackRecord:
    feedback_id: str
    trace_id: str
    tenant_id: str
    user_id: str
    rating: str
    comment: str | None
    correction: str | None
    created_at: str


class FeedbackStore:
    """User feedback is always linked to a trace, so a thumbs-down can be opened
    next to the exact retrieval, prompt stages and answer that produced it."""

    def __init__(self, traces: TraceStore) -> None:
        self._traces = traces
        self._records: list[FeedbackRecord] = []

    def add(
        self,
        trace_id: str,
        tenant_id: str,
        user_id: str,
        rating: str,
        comment: str | None = None,
        correction: str | None = None,
    ) -> FeedbackRecord:
        trace = self._traces.get(trace_id)
        if trace is None or trace.tenant_id != tenant_id:
            raise UnknownTrace(trace_id)
        if rating not in RATINGS:
            raise ValueError(f"rating must be one of {RATINGS}")
        if rating == "correction" and not (correction and correction.strip()):
            raise ValueError("a correction rating needs the corrected answer")
        record = FeedbackRecord(
            uuid.uuid4().hex[:12],
            trace_id,
            tenant_id,
            user_id,
            rating,
            comment,
            correction,
            datetime.now(UTC).isoformat(),
        )
        self._records.append(record)
        return record

    def for_trace(self, trace_id: str) -> list[FeedbackRecord]:
        return [r for r in self._records if r.trace_id == trace_id]

    def summary(self) -> dict[str, int]:
        out = {rating: 0 for rating in RATINGS}
        for record in self._records:
            out[record.rating] += 1
        return out

    def regression_cases(self) -> list[dict[str, Any]]:
        """Thumbs-down and corrections become test cases for the golden set."""
        cases: list[dict[str, Any]] = []
        for record in self._records:
            trace = self._traces.get(record.trace_id)
            if record.rating == "up" or trace is None:
                continue
            cases.append(
                {
                    "trace_id": record.trace_id,
                    "rating": record.rating,
                    "question": trace.attributes.get("question"),
                    "bad_answer": trace.attributes.get("answer"),
                    "expected_answer": record.correction,
                    "comment": record.comment,
                }
            )
        return cases
