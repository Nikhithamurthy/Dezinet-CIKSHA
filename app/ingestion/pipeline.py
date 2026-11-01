import hashlib
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from app.ingestion.chunkers import (
    fixed_chars,
    header_split,
    recursive_split,
    semantic_split,
    sliding_window,
)
from app.ingestion.models import Chunk, ParsedDocument
from app.ingestion.parsers import parse_file
from app.ingestion.tokens import count_tokens

STRATEGIES: dict[str, Callable[[str], list[str]]] = {
    "fixed": fixed_chars,
    "recursive": recursive_split,
    "sliding": sliding_window,
    "markdown_header": header_split,
    "semantic": semantic_split,
}


def chunk_document(doc: ParsedDocument, strategy: str = "recursive") -> list[Chunk]:
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from {sorted(STRATEGIES)}")
    splitter = STRATEGIES[strategy]
    ingested_at = datetime.now(UTC).isoformat()
    chunks: list[Chunk] = []
    for section in doc.sections:
        for piece in splitter(section.text):
            text = piece.strip()
            if not text:
                continue
            index = len(chunks)
            key = f"{doc.filename}|{strategy}|{index}|{text}"
            digest = hashlib.sha256(key.encode()).hexdigest()
            chunks.append(
                Chunk(
                    chunk_id=digest[:16],
                    chunk_index=index,
                    text=text,
                    token_count=count_tokens(text),
                    strategy=strategy,
                    source=doc.filename,
                    file_type=doc.file_type,
                    page=section.page,
                    section_header=section.header,
                    author=doc.author,
                    created_at=doc.created_at,
                    ingested_at=ingested_at,
                )
            )
    return chunks


def ingest_file(path: Path, strategy: str = "recursive") -> list[Chunk]:
    return chunk_document(parse_file(path), strategy)
