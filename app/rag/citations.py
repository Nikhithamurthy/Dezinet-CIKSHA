import re
from collections.abc import Sequence
from dataclasses import dataclass

from app.rag.context import ContextChunk

CITATION_RE = re.compile(r"\[([A-Za-z0-9][A-Za-z0-9_.-]*)\]")


@dataclass(frozen=True)
class Citation:
    chunk_id: str
    source: str
    page: int | None


def extract_cited_ids(answer: str) -> list[str]:
    return list(dict.fromkeys(CITATION_RE.findall(answer)))


def map_citations(answer: str, chunks: Sequence[ContextChunk]) -> tuple[list[Citation], list[str]]:
    """Returns (valid citations with source and page, ids the model invented)."""
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    valid: list[Citation] = []
    invalid: list[str] = []
    for cited in extract_cited_ids(answer):
        chunk = by_id.get(cited)
        if chunk is None:
            invalid.append(cited)
        else:
            valid.append(Citation(chunk.chunk_id, chunk.source, chunk.page))
    return valid, invalid
