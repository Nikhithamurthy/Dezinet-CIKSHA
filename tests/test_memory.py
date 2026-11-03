import asyncio
import os

import pytest

from app.memory.chat import ChatService, sanitize
from app.memory.condense import condense_question
from app.memory.store import InMemoryStore, Message, RedisChatStore
from app.memory.strategies import (
    EntityMemory,
    SlidingWindowMemory,
    SummaryBufferMemory,
    extract_entities,
    fit_budget,
)
from app.memory.tokens import message_tokens
from app.rag.llm import MockLLM
from app.rag.pipeline import RagResult


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


async def _fill(store: InMemoryStore, sid: str, turns: int) -> None:
    for i in range(turns):
        await store.append(sid, [Message("user", f"q{i}"), Message("assistant", f"a{i}")])


def test_session_expires_after_ttl() -> None:
    clock = Clock()
    store = InMemoryStore(ttl_seconds=10, clock=clock)

    async def scenario() -> tuple[int, int]:
        await store.append("s", [Message("user", "hi")])
        before = len(await store.history("s"))
        clock.now = 11.0
        return before, len(await store.history("s"))

    assert asyncio.run(scenario()) == (1, 0)


def test_activity_refreshes_ttl() -> None:
    clock = Clock()
    store = InMemoryStore(ttl_seconds=10, clock=clock)

    async def scenario() -> int:
        await store.append("s", [Message("user", "hi")])
        clock.now = 8.0
        await store.history("s")  # touch: new deadline is 18
        clock.now = 15.0
        return len(await store.history("s"))

    assert asyncio.run(scenario()) == 1


def test_message_cap_keeps_newest() -> None:
    store = InMemoryStore(max_messages=4)

    async def scenario() -> list[Message]:
        await _fill(store, "s", 3)
        return await store.history("s")

    history = asyncio.run(scenario())
    assert [m.content for m in history] == ["q1", "a1", "q2", "a2"]


def test_fit_budget_drops_oldest_first() -> None:
    msgs = [Message("user", f"message {i} " + "x" * 40) for i in range(5)]
    budget = sum(message_tokens(m) for m in msgs[-2:])
    assert fit_budget(msgs, budget) == msgs[-2:]


def test_sliding_window_keeps_last_k_turns() -> None:
    store = InMemoryStore()

    async def scenario() -> list[Message]:
        await _fill(store, "s", 6)
        return await SlidingWindowMemory(k_turns=2).context(store, "s")

    context = asyncio.run(scenario())
    assert [m.content for m in context] == ["q4", "a4", "q5", "a5"]


def test_k_turns_zero_returns_no_history() -> None:
    store = InMemoryStore()

    async def scenario() -> list[Message]:
        await _fill(store, "s", 3)
        return await SlidingWindowMemory(k_turns=0).context(store, "s")

    assert asyncio.run(scenario()) == []


def test_summary_buffer_keeps_early_fact_and_recent_turns() -> None:
    store = InMemoryStore()
    memory = SummaryBufferMemory(MockLLM(), keep_turns=2, batch=6)

    async def scenario() -> list[Message]:
        await store.append(
            "s", [Message("user", "Our code is ERR-5034."), Message("assistant", "ok")]
        )
        await _fill(store, "s", 6)
        return await memory.context(store, "s")

    context = asyncio.run(scenario())
    assert context[0].role == "system"
    assert "ERR-5034" in context[0].content
    assert context[-1].content == "a5"


def test_extract_entities_finds_names_and_codes_but_not_sentence_starters() -> None:
    found = extract_entities("My plan is Dezinet Teams and the error is ERR-5034. Tell me more.")
    assert "Dezinet Teams" in found
    assert "ERR-5034" in found
    assert "Tell" not in found
    assert "My" not in found


def test_entity_memory_survives_after_window_drops_the_turn() -> None:
    store = InMemoryStore()

    async def scenario() -> list[Message]:
        await store.append(
            "s",
            [
                Message("user", "Deploys fail with ERR-5034 on Dezinet Teams."),
                Message("assistant", "ok"),
            ],
        )
        await _fill(store, "s", 10)
        return await EntityMemory(k_turns=2).context(store, "s")

    context = asyncio.run(scenario())
    assert "ERR-5034" in context[0].content
    assert all("ERR-5034" not in m.content for m in context[1:])


def test_condense_resolves_pronoun_using_history() -> None:
    history = [
        Message("user", "Tell me about Dezinet Teams pricing plans."),
        Message("assistant", "Dezinet Teams has two plans."),
    ]
    out = asyncio.run(condense_question(MockLLM(), history, "How much does it cost?"))
    assert "dezinet" in out.lower()


def test_condense_skips_llm_on_first_turn() -> None:
    result = asyncio.run(condense_question(MockLLM(), [], "What is the refund policy?"))
    assert result == "What is the refund policy?"


def test_sanitize_strips_control_characters_and_caps_length() -> None:
    assert sanitize("hi\x00 there\x07") == "hi there"
    assert len(sanitize("x" * 5000)) == 2000


class StubAsker:
    def __init__(self) -> None:
        self.questions: list[str] = []

    async def ask(self, question: str) -> RagResult:
        self.questions.append(question)
        return RagResult(
            question=question,
            answer="Dezinet Teams costs 10 dollars. [d1]",
            status="answered",
        )


def test_chat_service_condenses_follow_up_and_persists_turns() -> None:
    asker = StubAsker()
    store = InMemoryStore()
    service = ChatService(asker, store, SlidingWindowMemory(), MockLLM())

    async def scenario() -> tuple[str, int]:
        await service.ask("s1", "Tell me about Dezinet Teams pricing plans.")
        turn = await service.ask("s1", "How much does it cost?")
        return turn.standalone_query, len(await store.history("s1"))

    standalone, stored = asyncio.run(scenario())
    assert asker.questions[0] == "Tell me about Dezinet Teams pricing plans."
    assert "dezinet" in standalone.lower()
    assert stored == 4


needs_redis = pytest.mark.skipif(not os.getenv("REDIS_URL"), reason="REDIS_URL not set")


@needs_redis
def test_redis_store_roundtrip_and_ttl() -> None:
    async def scenario() -> tuple[list[Message], tuple[str, int], int]:
        import redis.asyncio as aioredis

        client = aioredis.from_url(os.environ["REDIS_URL"], decode_responses=True)
        store = RedisChatStore(client, ttl_seconds=60, prefix="test_chat")
        try:
            await store.clear("t1")
            await store.append("t1", [Message("user", "hello"), Message("assistant", "hi")])
            await store.set_summary("t1", "sum", 2)
            return (
                await store.history("t1"),
                await store.get_summary("t1"),
                int(await client.ttl("test_chat:t1:msgs")),
            )
        finally:
            await store.clear("t1")
            await client.aclose()

    history, summary, ttl = asyncio.run(scenario())
    assert [m.content for m in history] == ["hello", "hi"]
    assert summary == ("sum", 2)
    assert 0 < ttl <= 60
