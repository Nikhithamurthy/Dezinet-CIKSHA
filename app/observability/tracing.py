import contextvars
import json
import time
import uuid
from collections import OrderedDict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class Span:
    span_id: str
    name: str
    parent_id: str | None
    start_ms: float
    duration_ms: float = 0.0
    attributes: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class Trace:
    trace_id: str
    name: str
    tenant_id: str
    user_id: str
    endpoint: str
    started_at: str
    t0: float = field(default_factory=time.perf_counter)
    duration_ms: float = 0.0
    finished: bool = False
    spans: list[Span] = field(default_factory=list)
    attributes: dict[str, Any] = field(default_factory=dict)

    def add_span(
        self,
        name: str,
        start_ms: float,
        duration_ms: float,
        parent_id: str | None = None,
        **attributes: Any,
    ) -> Span:
        span_obj = Span(
            uuid.uuid4().hex[:8], name, parent_id, start_ms, duration_ms, dict(attributes)
        )
        self.spans.append(span_obj)
        return span_obj

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("t0")
        data["spans"] = sorted(data["spans"], key=lambda s: s["start_ms"])
        return data


_current_trace: contextvars.ContextVar[Trace | None] = contextvars.ContextVar(
    "current_trace", default=None
)
_current_span: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_span", default=None
)


def current_trace() -> Trace | None:
    return _current_trace.get()


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Span]:
    """Child span of whatever span is active. Does nothing when no trace is active,
    so instrumented code costs almost nothing outside a traced request."""
    trace = _current_trace.get()
    if trace is None:
        yield Span("none", name, None, 0.0, attributes=dict(attributes))
        return
    began = time.perf_counter()
    current = Span(
        uuid.uuid4().hex[:8],
        name,
        _current_span.get(),
        (began - trace.t0) * 1000,
        attributes=dict(attributes),
    )
    token = _current_span.set(current.span_id)
    try:
        yield current
    except BaseException as exc:
        current.error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        current.duration_ms = (time.perf_counter() - began) * 1000
        _current_span.reset(token)
        trace.spans.append(current)


class TraceStore:
    """In-memory store (newest 1000 traces) with an optional JSON-lines file."""

    def __init__(self, max_traces: int = 1000, path: Path | None = None) -> None:
        self._traces: OrderedDict[str, Trace] = OrderedDict()
        self._max = max_traces
        self._path = path

    def add(self, trace: Trace) -> None:
        self._traces[trace.trace_id] = trace
        while len(self._traces) > self._max:
            self._traces.popitem(last=False)

    def persist(self, trace: Trace) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trace.to_dict()) + "\n")

    def get(self, trace_id: str) -> Trace | None:
        return self._traces.get(trace_id)

    def recent(self, n: int = 20) -> list[Trace]:
        return list(self._traces.values())[-n:]


class Tracer:
    def __init__(self, store: TraceStore) -> None:
        self.store = store

    def start(
        self,
        name: str,
        *,
        tenant_id: str = "default",
        user_id: str = "anonymous",
        endpoint: str = "",
    ) -> Trace:
        trace = Trace(
            trace_id=uuid.uuid4().hex,
            name=name,
            tenant_id=tenant_id,
            user_id=user_id,
            endpoint=endpoint,
            started_at=datetime.now(UTC).isoformat(),
        )
        self.store.add(trace)
        return trace

    @contextmanager
    def activate(self, trace: Trace) -> Iterator[Trace]:
        trace_token = _current_trace.set(trace)
        span_token = _current_span.set(None)
        try:
            yield trace
        finally:
            _current_span.reset(span_token)
            _current_trace.reset(trace_token)

    def finish(self, trace: Trace) -> None:
        if trace.finished:
            return
        trace.finished = True
        trace.duration_ms = (time.perf_counter() - trace.t0) * 1000
        self.store.persist(trace)

    @contextmanager
    def trace(
        self,
        name: str,
        *,
        tenant_id: str = "default",
        user_id: str = "anonymous",
        endpoint: str = "",
    ) -> Iterator[Trace]:
        trace = self.start(name, tenant_id=tenant_id, user_id=user_id, endpoint=endpoint)
        try:
            with self.activate(trace):
                yield trace
        except BaseException as exc:
            trace.attributes["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.finish(trace)


def render_tree(trace: Trace) -> str:
    children: dict[str | None, list[Span]] = {}
    for item in sorted(trace.spans, key=lambda s: s.start_ms):
        children.setdefault(item.parent_id, []).append(item)
    lines = [f"{trace.name} [{trace.trace_id[:8]}] {trace.duration_ms:.1f} ms"]
    shown = ("model", "prompt_tokens", "completion_tokens", "cost_usd", "returned")

    def walk(parent: str | None, depth: int) -> None:
        for item in children.get(parent, []):
            extra = " ".join(f"{k}={item.attributes[k]}" for k in shown if k in item.attributes)
            failure = f" ERROR {item.error}" if item.error else ""
            lines.append(f"{'  ' * depth}- {item.name} {item.duration_ms:.2f} ms {extra}{failure}")
            walk(item.span_id, depth + 1)

    walk(None, 1)
    return "\n".join(lines)
