import argparse
import asyncio
import time

from app.memory.store import InMemoryStore, Message
from app.memory.strategies import (
    EntityMemory,
    MemoryStrategy,
    SlidingWindowMemory,
    SummaryBufferMemory,
)
from app.memory.tokens import message_tokens
from app.rag.llm import get_llm

FACT = "ERR-5034"
SID = "bench"


async def run(memory: MemoryStrategy, turns: int) -> dict[str, object]:
    store = InMemoryStore(max_messages=1000)
    await store.append(
        SID,
        [
            Message("user", f"Our deployment fails with {FACT} on Dezinet Teams."),
            Message("assistant", "Noted."),
        ],
    )
    context: list[Message] = []
    start = time.perf_counter()
    for i in range(turns):
        await store.append(
            SID,
            [
                Message("user", f"Filler question {i} about billing and invoices."),
                Message("assistant", f"Filler answer {i}: invoices are emailed monthly."),
            ],
        )
        context = await memory.context(store, SID)
    elapsed_ms = (time.perf_counter() - start) * 1000 / turns
    text = " ".join(m.content for m in context)
    return {
        "strategy": memory.name,
        "tokens": sum(message_tokens(m) for m in context),
        "recalls_fact": FACT in text,
        "ms_per_turn": round(elapsed_ms, 2),
    }


async def main(turns: int, kind: str) -> None:
    llm = get_llm(kind)
    strategies: list[MemoryStrategy] = [
        SlidingWindowMemory(k_turns=4),
        SummaryBufferMemory(llm),
        EntityMemory(k_turns=2),
    ]
    print(f"{turns} filler turns after the turn that states {FACT}, llm={kind}")
    print(f"{'strategy':<16}{'tokens':>8}{'recalls_fact':>14}{'ms/turn':>10}")
    for memory in strategies:
        row = await run(memory, turns)
        print(
            f"{row['strategy']!s:<16}{row['tokens']!s:>8}"
            f"{row['recalls_fact']!s:>14}{row['ms_per_turn']!s:>10}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--turns", type=int, default=30)
    parser.add_argument("--llm", default="mock", choices=["mock", "openai"])
    args = parser.parse_args()
    asyncio.run(main(args.turns, args.llm))
