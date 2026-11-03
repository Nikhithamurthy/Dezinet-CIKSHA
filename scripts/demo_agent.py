"""Offline demo: plan, research, buggy code -> debugger, weak draft -> reviewer, human approval."""

import asyncio

from app.agents.nodes import AgentState, InMemoryOutbox, build_agent_graph

from app.agents.models import MockAgentModel
from app.tools.builtin import CorpusSearch

DOCS = [
    ("Refund policy", "Refunds for annual plans are issued within 30 days of purchase."),
    ("Support hours", "Support is available by email on weekdays from 9am to 6pm."),
]


async def main() -> None:
    outbox = InMemoryOutbox()
    model = MockAgentModel(broken_code=True, sloppy_first_draft=True)
    graph = build_agent_graph(model, CorpusSearch(DOCS), outbox)
    state = AgentState(
        goal="Refund policy for annual plans: price 120 after a 20% discount",
        recipient="finance@example.com",
    )
    result = await graph.invoke(state, "demo")
    print("ran:", " -> ".join(result.path))
    print("draft:", result.state.get("draft"))
    if result.status == "interrupted":
        answer = input(f"Send this to {state['recipient']}? [y/N] ")
        result = await graph.resume("demo", {"approved": answer.strip().lower() == "y"})
    print("ran:", " -> ".join(result.path))
    print("status:", result.state.get("status"), "| outbox:", outbox.sent)


if __name__ == "__main__":
    asyncio.run(main())
