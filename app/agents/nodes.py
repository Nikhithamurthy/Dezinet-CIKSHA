"""Agent wiring:

planner -> researcher -> coder -> executor --error--> debugger -> executor (loop, bounded)
                                     |ok
                                  writer -> reviewer --fail--> writer (loop, bounded)
                                               |pass
                                  [HUMAN APPROVAL] act -> output
Anything stuck (repeated action, step limit, retries exhausted) goes to `fallback`."""

import hashlib
from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict

from app.agents.graph import END, Checkpointer, StateGraph
from app.agents.sandbox import run_python
from app.tools.builtin import SearchBackend

MAX_DEBUG_ATTEMPTS = 3
MAX_REVISIONS = 2


class AgentState(TypedDict, total=False):
    goal: str
    recipient: str
    plan: list[str]
    context: list[str]
    code: str
    code_output: str
    error: str
    debug_attempts: int
    draft: str
    feedback: str
    review_passed: bool
    revisions: int
    approved: bool
    outcome: str
    final: str
    status: str
    last_action: str
    halt_reason: str


@dataclass(frozen=True)
class Review:
    passed: bool
    feedback: str


class AgentModel(Protocol):
    async def plan(self, goal: str) -> list[str]: ...
    async def write_code(self, goal: str, context: list[str]) -> str: ...
    async def fix_code(self, code: str, error: str) -> str: ...
    async def draft(self, goal: str, context: list[str], result: str, feedback: str) -> str: ...
    async def review(self, goal: str, draft: str, context: list[str]) -> Review: ...


class Outbox(Protocol):
    async def send(self, to: str, body: str) -> str: ...


@dataclass
class InMemoryOutbox:
    """The destructive side effect. A real one would call an SMTP or transaction API."""

    sent: list[tuple[str, str]] = field(default_factory=list)

    async def send(self, to: str, body: str) -> str:
        self.sent.append((to, body))
        return f"msg-{len(self.sent)}"


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def build_agent_graph(
    model: AgentModel,
    retriever: SearchBackend,
    outbox: Outbox,
    *,
    checkpointer: Checkpointer | None = None,
    recursion_limit: int = 25,
    max_repeats: int = 3,
) -> StateGraph[AgentState]:
    graph: StateGraph[AgentState] = StateGraph(
        checkpointer=checkpointer,
        recursion_limit=recursion_limit,
        max_repeats=max_repeats,
        fallback="fallback",
        interrupt_before=["act"],  # human approval gate before the destructive step
    )

    async def planner(state: AgentState) -> dict[str, Any]:
        return {"plan": await model.plan(state["goal"])}

    async def researcher(state: AgentState) -> dict[str, Any]:
        hits = await retriever.search(state["goal"], 3)
        return {"context": [f"{h['title']}: {h['snippet']}" for h in hits]}

    async def coder(state: AgentState) -> dict[str, Any]:
        code = await model.write_code(state["goal"], state.get("context", []))
        return {"code": code, "debug_attempts": 0, "error": ""}

    async def executor(state: AgentState) -> dict[str, Any]:
        code = state.get("code", "")
        result = await run_python(code)
        action = f"run:{_digest(code)}:{result.error}"  # same code + same error = no progress
        if result.ok:
            return {"code_output": result.output, "error": "", "last_action": action}
        return {"error": result.error, "last_action": action}

    async def debugger(state: AgentState) -> dict[str, Any]:
        fixed = await model.fix_code(state.get("code", ""), state.get("error", ""))
        return {"code": fixed, "debug_attempts": state.get("debug_attempts", 0) + 1}

    async def writer(state: AgentState) -> dict[str, Any]:
        draft = await model.draft(
            state["goal"],
            state.get("context", []),
            state.get("code_output", ""),
            state.get("feedback", ""),
        )
        return {"draft": draft, "revisions": state.get("revisions", 0) + 1}

    async def reviewer(state: AgentState) -> dict[str, Any]:
        review = await model.review(state["goal"], state.get("draft", ""), state.get("context", []))
        return {"review_passed": review.passed, "feedback": review.feedback}

    async def act(state: AgentState) -> dict[str, Any]:
        # Defence in depth: even if the interrupt were skipped, no approval means no send.
        if state.get("approved") is not True:
            return {"outcome": "not sent: approval was not granted"}
        receipt = await outbox.send(state["recipient"], state.get("draft", ""))
        return {"outcome": f"sent: {receipt}"}

    async def output(state: AgentState) -> dict[str, Any]:
        sent = state.get("approved") is True
        return {"final": state.get("draft", ""), "status": "completed" if sent else "cancelled"}

    async def fallback(state: AgentState) -> dict[str, Any]:
        reason = (
            state.get("halt_reason") or state.get("error") or state.get("feedback") or "unknown"
        )
        message = f"I could not finish this automatically: {reason}. Please clarify the request."
        return {"final": message, "status": "needs_help"}

    def after_executor(state: AgentState) -> str:
        if not state.get("error"):
            return "writer"
        return "debugger" if state.get("debug_attempts", 0) < MAX_DEBUG_ATTEMPTS else "fallback"

    def after_review(state: AgentState) -> str:
        if state.get("review_passed"):
            return "act"
        return "writer" if state.get("revisions", 0) <= MAX_REVISIONS else "fallback"

    nodes = {
        "planner": planner,
        "researcher": researcher,
        "coder": coder,
        "executor": executor,
        "debugger": debugger,
        "writer": writer,
        "reviewer": reviewer,
        "act": act,
        "output": output,
        "fallback": fallback,
    }
    for name, fn in nodes.items():
        graph.add_node(name, fn)
    graph.set_entry("planner")
    graph.add_edge("planner", "researcher")
    graph.add_edge("researcher", "coder")
    graph.add_edge("coder", "executor")
    graph.add_conditional_edges("executor", after_executor)
    graph.add_edge("debugger", "executor")
    graph.add_edge("writer", "reviewer")
    graph.add_conditional_edges("reviewer", after_review)
    graph.add_edge("act", "output")
    graph.add_edge("output", END)
    graph.add_edge("fallback", END)
    return graph
