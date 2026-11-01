from pydantic import BaseModel


class Section(BaseModel):
    text: str
    page: int | None = None
    header: str | None = None


class ParsedDocument(BaseModel):
    filename: str
    file_type: str
    author: str | None = None
    created_at: str | None = None
    sections: list[Section]


class Chunk(BaseModel):
    chunk_id: str
    chunk_index: int
    text: str
    token_count: int
    strategy: str
    source: str
    file_type: str
    page: int | None = None
    section_header: str | None = None
    author: str | None = None
    created_at: str | None = None
    ingested_at: str
