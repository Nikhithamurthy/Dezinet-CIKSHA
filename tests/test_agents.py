import asyncio
from typing import Any

import pytest
from app.agents.nodes import InMemoryOutbox, build_agent_graph

from app.agents.graph import GraphError, GraphRecursionError, StateGraph
from app.agents.models import MockAgentModel

run = asyncio.run
GOAL = "Quote the annual refund price after a 20% discount"


class StubSearch:
    async def search(self, query: str, k: int) -> list[dict[str, Any]]:
        return [{"title": "Refund policy", "snippet": "Refunds are issued within 30 days."}]


def _agent(model: MockAgentModel | None = None, **kwargs: Any) -> Any:
    outbox = InMemoryOutbox()
    return build_agent_graph(model or MockAgentModel(), StubSearch(), outbox, **kwargs), outbox


def _state() -> Any:
    return {"goal": GOAL, "recipient": "boss@example.com"}


def test_pauses_for_approval_then_sends() -> None:
    graph, outbox = _agent()
    first = run(graph.invoke(_state(), "t1"))
    assert first.status == "interrupted"
    assert first.next_node == "act"
    assert outbox.sent == []
    done = run(graph.resume("t1", {"approved": True}))
    assert done.status == "completed"
    assert done.state["status"] == "completed"
    assert outbox.sent == [("boss@example.com", done.state["draft"])]
    assert done.path == (
        "planner",
        "researcher",
        "coder",
        "executor",
        "writer",
        "reviewer",
        "act",
        "output",
    )


def test_rejection_sends_nothing() -> None:
    graph, outbox = _agent()
    run(graph.invoke(_state(), "t1"))
    done = run(graph.resume("t1", {"approved": False}))
    assert outbox.sent == []
    assert done.state["status"] == "cancelled"


def test_resume_without_a_decision_is_not_approval() -> None:
    graph, outbox = _agent()
    run(graph.invoke(_state(), "t1"))
    done = run(graph.resume("t1"))
    assert outbox.sent == []
    assert done.state["outcome"].startswith("not sent")


def test_failed_code_routes_to_debugger_then_back() -> None:
    graph, _ = _agent(MockAgentModel(broken_code=True))
    result = run(graph.invoke(_state(), "t1"))
    assert result.path[3:6] == ("executor", "debugger", "executor")
    assert result.state["debug_attempts"] == 1
    assert result.state["error"] == ""
    assert result.state["code_output"].strip() == "96.0"


def test_reviewer_feedback_triggers_a_revision() -> None:
    graph, _ = _agent(MockAgentModel(sloppy_first_draft=True))
    result = run(graph.invoke(_state(), "t1"))
    assert result.path.count("writer") == 2
    assert result.path.count("reviewer") == 2
    assert result.state["revisions"] == 2
    assert "[1]" in result.state["draft"]


def test_revisions_are_bounded() -> None:
    graph, outbox = _agent(MockAgentModel(reviewer_always_fails=True))
    result = run(graph.invoke(_state(), "t1"))
    assert result.status == "completed"
    assert result.state["status"] == "needs_help"
    assert result.path.count("writer") == 3
    assert "Needs more detail" in result.state["final"]
    assert outbox.sent == []


def test_repeated_identical_failure_is_detected() -> None:
    graph, _ = _agent(MockAgentModel(unfixable=True))
    result = run(graph.invoke(_state(), "t1"))
    assert result.state["status"] == "needs_help"
    assert "stuck in a loop" in result.state["halt_reason"]
    assert result.path.count("executor") == 3
    assert result.path[-1] == "fallback"


def test_recursion_limit_routes_to_fallback() -> None:
    graph, _ = _agent(recursion_limit=3)
    result = run(graph.invoke(_state(), "t1"))
    assert result.state["status"] == "needs_help"
    assert "recursion limit" in result.state["halt_reason"]


def test_runaway_loop_raises_without_a_fallback() -> None:
    async def ping(state: dict[str, Any]) -> dict[str, Any]:
        return {"n": state.get("n", 0) + 1}

    graph: StateGraph[dict[str, Any]] = StateGraph(recursion_limit=5)
    graph.add_node("a", ping)
    graph.add_edge("a", "a")
    graph.set_entry("a")
    with pytest.raises(GraphRecursionError):
        run(graph.invoke({}, "t"))


def test_miswired_graph_is_rejected() -> None:
    async def noop(state: dict[str, Any]) -> dict[str, Any]:
        return {}

    graph: StateGraph[dict[str, Any]] = StateGraph()
    graph.add_node("a", noop)
    graph.set_entry("a")
    with pytest.raises(GraphError):
        run(graph.invoke({}, "t"))


def test_time_travel_fork_edits_state_and_reasks_for_approval() -> None:
    graph, outbox = _agent()
    run(graph.invoke(_state(), "t1"))
    paused = graph.checkpointer.latest("t1")
    assert paused is not None
    graph.checkpointer.fork("t1", paused.step, "alt", {"recipient": "cfo@example.com"})
    again = run(graph.resume("alt"))
    assert again.status == "interrupted"  # the approval does not carry over to the fork
    run(graph.resume("alt", {"approved": True}))
    assert [to for to, _ in outbox.sent] == ["cfo@example.com"]
    latest = graph.checkpointer.latest("t1")
    assert latest is not None
    assert latest.status == "interrupted"  # the original thread is untouched
