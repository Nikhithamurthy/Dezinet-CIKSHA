import re
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ContextChunk:
    chunk_id: str
    text: str
    source: str = "unknown"
    page: int | None = None
    score: float = 0.0


_CHUNK_RE = re.compile(
    r"<<< chunk_id=(\S+) \| source=(.*?) \| page=(.*?) >>>\n(.*?)\n<<< end \1 >>>",
    re.DOTALL,
)


def order_for_attention(chunks: Sequence[ContextChunk]) -> list[ContextChunk]:
    """Counter 'lost in the middle': best chunk first, second best last,
    weaker chunks in the middle. Input must be sorted best-first."""
    front: list[ContextChunk] = []
    back: list[ContextChunk] = []
    for index, chunk in enumerate(chunks):
        if index % 2 == 0:
            front.append(chunk)
        else:
            back.append(chunk)
    return front + back[::-1]


def format_chunk(chunk: ContextChunk) -> str:
    page = "n/a" if chunk.page is None else str(chunk.page)
    # A chunk must not be able to forge a boundary marker (indirect prompt injection).
    safe = chunk.text.replace("<<<", "< < <").replace(">>>", "> > >")
    header = f"<<< chunk_id={chunk.chunk_id} | source={chunk.source} | page={page} >>>"
    return f"{header}\n{safe}\n<<< end {chunk.chunk_id} >>>"


def assemble_context(chunks: Sequence[ContextChunk]) -> str:
    return "\n\n".join(format_chunk(chunk) for chunk in order_for_attention(chunks))


def parse_context(text: str) -> list[ContextChunk]:
    chunks: list[ContextChunk] = []
    for match in _CHUNK_RE.finditer(text):
        page_raw = match.group(3).strip()
        chunks.append(
            ContextChunk(
                chunk_id=match.group(1),
                text=match.group(4),
                source=match.group(2),
                page=int(page_raw) if page_raw.isdigit() else None,
            )
        )
    return chunks
