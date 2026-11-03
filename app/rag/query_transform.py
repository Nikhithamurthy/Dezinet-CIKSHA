import re

from app.rag.llm import LLM
from app.rag.prompts import HYDE_SYSTEM, expand_system

_NUMBERING = re.compile(r"^\s*(?:[-*]|\d+[.)])\s*")


async def expand_queries(llm: LLM, question: str, n: int = 3) -> list[str]:
    """Multi-query expansion: up to n distinct rewrites, never repeating the original."""
    raw = await llm.complete(expand_system(n), question)
    seen = {question.strip().lower()}
    variants: list[str] = []
    for line in raw.splitlines():
        text = _NUMBERING.sub("", line).strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            variants.append(text)
    return variants[:n]


async def hyde_passage(llm: LLM, question: str) -> str:
    """HyDE: a hypothetical answer, searched instead of the question itself."""
    return (await llm.complete(HYDE_SYSTEM, question)).strip()
