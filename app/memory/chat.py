import re
from dataclasses import dataclass
from typing import Protocol

from app.memory.condense import condense_question
from app.memory.store import ChatStore, Message
from app.memory.strategies import MemoryStrategy
from app.rag.llm import LLM
from app.rag.pipeline import RagResult

_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def sanitize(text: str, max_chars: int = 2000) -> str:
    """Strip control characters and cap length before anything is cached."""
    return _CONTROL.sub("", text).strip()[:max_chars]


class Asker(Protocol):
    async def ask(self, question: str) -> RagResult: ...


@dataclass
class ChatTurn:
    session_id: str
    standalone_query: str
    result: RagResult


class ChatService:
    """History -> condense follow-up -> RAG -> store both turns."""

    def __init__(self, asker: Asker, store: ChatStore, memory: MemoryStrategy, llm: LLM) -> None:
        self._asker = asker
        self._store = store
        self._memory = memory
        self._llm = llm

    async def ask(self, session_id: str, question: str) -> ChatTurn:
        clean = sanitize(question)
        context = await self._memory.context(self._store, session_id)
        standalone = await condense_question(self._llm, context, clean)
        result = await self._asker.ask(standalone)
        await self._store.append(
            session_id,
            [Message("user", clean), Message("assistant", sanitize(result.answer))],
        )
        return ChatTurn(session_id, standalone, result)
