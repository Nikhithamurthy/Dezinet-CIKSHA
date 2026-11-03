NO_ANSWER = "I don't know based on the provided documents."
FALLBACK_MESSAGE = (
    "I don't know based on the provided documents. "
    "Try rephrasing your question with more specific terms."
)

GROUNDED_SYSTEM = (
    "TASK: ANSWER\n"
    "You are a careful enterprise assistant.\n"
    "Answer ONLY using the facts in the context chunks provided by the user.\n"
    "Rules:\n"
    "1. If the context does not contain the answer, reply exactly: "
    "I don't know based on the provided documents.\n"
    "2. After every claim, cite the chunk it came from as [chunk_id], for example [d12].\n"
    "3. Never use outside knowledge. Never cite a chunk that is not in the context.\n"
    "4. Be concise."
)

HYDE_SYSTEM = (
    "TASK: HYDE\n"
    "Write a short passage (two sentences) that would answer the question, "
    "as if copied from a company document."
)


def expand_system(n: int) -> str:
    return (
        "TASK: EXPAND\n"
        f"Write {n} alternative search queries for the user's question, "
        "one per line, no numbering."
    )
