"""A small state-graph controller: typed state, conditional edges, checkpoints, interrupts.

Same model as LangGraph (nodes, edges, checkpointer, interrupt_before), kept dependency-free so
it runs offline and every control-flow decision is visible and testable."""

import json
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Generic, Literal, Protocol, TypeVar, cast

END = "__end__"
HALT_KEY = "halt_reason"

S = TypeVar("S", bound=Mapping[str, Any])
NodeFn = Callable[[S], Awaitable[Mapping[str, Any]]]
Router = Callable[[S], str]
Status = Literal["running", "interrupted", "completed"]


class GraphError(Exception):
    """The graph is mis-wired or was used incorrectly."""


class GraphRecursionError(GraphError):
    """Step limit hit or an action repeated too often, and no fallback node exists."""


def _clone(state: Mapping[str, Any]) -> dict[str, Any]:
    """JSON round-trip: proves the state is serialisable and isolates snapshots from mutation."""
    return cast(dict[str, Any], json.loads(json.dumps(dict(state))))


@dataclass(frozen=True)
class Checkpoint:
    thread_id: str
    step: int
    node: str  # node that produced this snapshot ("" before the first node)
    next_node: str  # node that runs next (END when finished)
    status: Status
    state: dict[str, Any]


class Checkpointer(Protocol):
    def put(self, checkpoint: Checkpoint) -> None: ...
    def latest(self, thread_id: str) -> Checkpoint | None: ...
    def history(self, thread_id: str) -> list[Checkpoint]: ...
    def fork(
        self,
        thread_id: str,
        step: int,
        new_thread_id: str,
        updates: Mapping[str, Any] | None = None,
    ) -> None: ...


class InMemoryCheckpointer:
    """Swap for Postgres or Redis in production: the contract is just put/latest/history/fork."""

    def __init__(self) -> None:
        self._threads: dict[str, list[Checkpoint]] = {}

    def put(self, checkpoint: Checkpoint) -> None:
        stored = replace(checkpoint, state=_clone(checkpoint.state))
        items = self._threads.setdefault(checkpoint.thread_id, [])
        if items and items[-1].step == checkpoint.step:
            items[-1] = stored  # upsert: marking a step "interrupted" does not add a new step
        else:
            items.append(stored)

    def latest(self, thread_id: str) -> Checkpoint | None:
        items = self._threads.get(thread_id)
        return items[-1] if items else None

    def history(self, thread_id: str) -> list[Checkpoint]:
        return list(self._threads.get(thread_id, []))

    def fork(
        self,
        thread_id: str,
        step: int,
        new_thread_id: str,
        updates: Mapping[str, Any] | None = None,
    ) -> None:
        """Time travel: copy a thread up to `step` into a new thread, optionally editing state."""
        if new_thread_id in self._threads:
            raise GraphError(f"thread '{new_thread_id}' already exists")
        kept = [c for c in self._threads.get(thread_id, []) if c.step <= step]
        if not kept:
            raise GraphError(f"no checkpoint at or before step {step}")
        forked = [replace(c, thread_id=new_thread_id, state=_clone(c.state)) for c in kept]
        last = forked[-1]
        # Status resets to "running" so an interrupt fires again: an approval never carries over.
        forked[-1] = replace(last, state={**last.state, **(updates or {})}, status="running")
        self._threads[new_thread_id] = forked


@dataclass(frozen=True)
class RunResult(Generic[S]):
    thread_id: str
    status: Status
    state: S
    next_node: str
    path: tuple[str, ...]


class StateGraph(Generic[S]):
    def __init__(
        self,
        *,
        checkpointer: Checkpointer | None = None,
        recursion_limit: int = 25,
        max_repeats: int = 3,
        fallback: str | None = None,
        interrupt_before: Iterable[str] = (),
    ) -> None:
        self.checkpointer: Checkpointer = (
            InMemoryCheckpointer() if checkpointer is None else checkpointer
        )
        self.recursion_limit = recursion_limit
        self.max_repeats = max_repeats
        self.fallback = fallback
        self._interrupt = frozenset(interrupt_before)
        self._nodes: dict[str, NodeFn[S]] = {}
        self._edges: dict[str, str] = {}
        self._routers: dict[str, Router[S]] = {}
        self._entry: str | None = None

    # ------------------------------------------------------------------ wiring

    def add_node(self, name: str, fn: NodeFn[S]) -> None:
        if name == END or name in self._nodes:
            raise GraphError(f"invalid or duplicate node name '{name}'")
        self._nodes[name] = fn

    def add_edge(self, src: str, dst: str) -> None:
        self._check_new_edge(src)
        self._edges[src] = dst

    def add_conditional_edges(self, src: str, router: Router[S]) -> None:
        """The router reads the state and returns the next node's name (or END)."""
        self._check_new_edge(src)
        self._routers[src] = router

    def set_entry(self, name: str) -> None:
        self._entry = name

    def _check_new_edge(self, src: str) -> None:
        if src not in self._nodes:
            raise GraphError(f"unknown node '{src}'")
        if src in self._edges or src in self._routers:
            raise GraphError(f"node '{src}' already has an outgoing edge")

    def _validate(self) -> str:
        if self._entry is None or self._entry not in self._nodes:
            raise GraphError("entry node is not set")
        for name in self._nodes:
            if name not in self._edges and name not in self._routers:
                raise GraphError(f"node '{name}' has no outgoing edge")
        for dst in self._edges.values():
            if dst != END and dst not in self._nodes:
                raise GraphError(f"edge points at unknown node '{dst}'")
        extra = [*self._interrupt, *([self.fallback] if self.fallback else [])]
        for name in extra:
            if name not in self._nodes:
                raise GraphError(f"unknown node '{name}'")
        return self._entry

    # --------------------------------------------------------------- execution

    async def invoke(self, initial: S, thread_id: str) -> RunResult[S]:
        entry = self._validate()
        if self.checkpointer.latest(thread_id) is not None:
            raise GraphError(f"thread '{thread_id}' already exists; use resume()")
        start = Checkpoint(thread_id, 0, "", entry, "running", _clone(initial))
        self.checkpointer.put(start)
        return await self._run(thread_id, start, _clone(initial), skip_interrupt=False)

    async def resume(
        self, thread_id: str, updates: Mapping[str, Any] | None = None
    ) -> RunResult[S]:
        """Continue from the latest checkpoint. `updates` carry the human's decision."""
        self._validate()
        cp = self.checkpointer.latest(thread_id)
        if cp is None:
            raise GraphError(f"unknown thread '{thread_id}'")
        if cp.status == "completed":
            raise GraphError(f"thread '{thread_id}' already finished")
        state = {**_clone(cp.state), **(updates or {})}
        return await self._run(thread_id, cp, state, skip_interrupt=cp.status == "interrupted")

    async def _run(
        self, thread_id: str, cp: Checkpoint, state: dict[str, Any], *, skip_interrupt: bool
    ) -> RunResult[S]:
        current = cast(S, state)
        node, step, prev = cp.next_node, cp.step, cp.node
        seen: Counter[tuple[str, str]] = Counter()
        steps_run, tripped = 0, False
        while node != END:
            if node in self._interrupt and not skip_interrupt:
                self._save(thread_id, step, prev, node, "interrupted", current)
                return self._result(thread_id, "interrupted", current, node)
            skip_interrupt = False
            update = await self._nodes[node](current)
            current = cast(S, {**current, **update})
            step, steps_run = step + 1, steps_run + 1
            nxt = self._route(node, current)
            if not tripped:
                reason = self._stuck(node, update, steps_run, nxt, seen)
                if reason is not None:
                    if self.fallback is None:
                        raise GraphRecursionError(reason)
                    current = cast(S, {**current, HALT_KEY: reason})
                    nxt, tripped = self.fallback, True
            status: Status = "completed" if nxt == END else "running"
            self._save(thread_id, step, node, nxt, status, current)
            prev, node = node, nxt
        return self._result(thread_id, "completed", current, END)

    def _route(self, node: str, state: S) -> str:
        router = self._routers.get(node)
        target = router(state) if router is not None else self._edges[node]
        if target != END and target not in self._nodes:
            raise GraphError(f"'{node}' routed to unknown node '{target}'")
        return target

    def _stuck(
        self,
        node: str,
        update: Mapping[str, Any],
        steps_run: int,
        nxt: str,
        seen: Counter[tuple[str, str]],
    ) -> str | None:
        """Nodes that may loop publish a `last_action` fingerprint; repeats mean no progress."""
        action = update.get("last_action")
        if action is not None:
            key = (node, str(action))
            seen[key] += 1
            if seen[key] >= self.max_repeats:
                return f"stuck in a loop: '{node}' repeated the same action {seen[key]} times"
        if steps_run >= self.recursion_limit and nxt != END:
            return f"recursion limit reached ({self.recursion_limit} steps)"
        return None

    def _save(
        self, thread_id: str, step: int, node: str, nxt: str, status: Status, state: S
    ) -> None:
        self.checkpointer.put(Checkpoint(thread_id, step, node, nxt, status, dict(state)))

    def _result(self, thread_id: str, status: Status, state: S, next_node: str) -> RunResult[S]:
        path = tuple(c.node for c in self.checkpointer.history(thread_id) if c.node)
        return RunResult(thread_id, status, state, next_node, path)
