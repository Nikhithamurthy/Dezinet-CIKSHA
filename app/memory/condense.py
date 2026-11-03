from app.memory.store import Message
from app.rag.llm import LLM

CONDENSE_SYSTEM = (
    "TASK: CONDENSE\n"
    "Rewrite the final user question as a standalone search query, using the "
    "conversation to resolve pronouns and references. Do not answer it. "
    "Output only the query."
)


async def condense_question(llm: LLM, context: list[Message], question: str) -> str:
    """Follow-up -> standalone query. Skips the LLM call on the first turn."""
    if not context:
        return question
    transcript = "\n".join(f"{m.role}: {m.content}" for m in context)
    raw = (await llm.complete(CONDENSE_SYSTEM, f"{transcript}\n\nQuestion: {question}")).strip()
    first_line = raw.splitlines()[0].strip() if raw else ""
    return first_line or question
