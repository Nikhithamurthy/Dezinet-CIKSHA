import re
from datetime import UTC, datetime
from pathlib import Path

import docx
from bs4 import BeautifulSoup
from docx.table import Table
from docx.text.paragraph import Paragraph
from pypdf import PdfReader

from app.ingestion.cleaning import clean_text, table_to_markdown
from app.ingestion.models import ParsedDocument, Section

FENCE = "`" * 3
_HEADER = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def _mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()


def _flush(sections: list[Section], header: str | None, buf: list[str]) -> None:
    body = clean_text("\n".join(buf))
    if not body:
        return
    text = f"{header}\n\n{body}" if header else body
    sections.append(Section(text=text, header=header))


def _split_markdown(raw: str) -> list[Section]:
    """Split on headings (ignoring code fences); each section keeps its header."""
    sections: list[Section] = []
    header: str | None = None
    buf: list[str] = []
    in_fence = False
    for line in raw.splitlines():
        if line.strip().startswith(FENCE):
            in_fence = not in_fence
        match = None if in_fence else _HEADER.match(line)
        if match:
            _flush(sections, header, buf)
            header = match.group(2).strip()
            buf = []
        else:
            buf.append(line)
    _flush(sections, header, buf)
    return sections


def parse_pdf(path: Path) -> ParsedDocument:
    reader = PdfReader(str(path))
    sections: list[Section] = []
    for number, page in enumerate(reader.pages, start=1):
        text = clean_text(str(page.extract_text() or ""))
        if text:
            sections.append(Section(text=text, page=number))
    if not sections:
        raise ValueError(f"{path.name}: no extractable text (scanned PDF? run OCR first)")
    author: str | None = None
    created: str | None = None
    meta = reader.metadata
    if meta is not None:
        author = str(meta.author) if meta.author else None
        try:
            created = meta.creation_date.isoformat() if meta.creation_date else None
        except Exception:
            created = None
    return ParsedDocument(
        filename=path.name,
        file_type="pdf",
        author=author,
        created_at=created or _mtime_iso(path),
        sections=sections,
    )


def parse_markdown(path: Path) -> ParsedDocument:
    raw = path.read_text(encoding="utf-8")
    return ParsedDocument(
        filename=path.name,
        file_type="markdown",
        created_at=_mtime_iso(path),
        sections=_split_markdown(raw),
    )


def parse_html(path: Path) -> ParsedDocument:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    meta = soup.find("meta", attrs={"name": "author"})
    author = str(meta.get("content")) if meta is not None and meta.get("content") else None
    for tag in soup(["script", "style", "nav", "footer", "aside", "noscript"]):
        tag.decompose()  # boilerplate noise
    for table in soup.find_all("table"):
        rows = [
            [cell.get_text(" ", strip=True) for cell in tr.find_all(["th", "td"])]
            for tr in table.find_all("tr")
        ]
        table.replace_with("\n\n" + table_to_markdown(rows) + "\n\n")
    for heading in soup.find_all(re.compile(r"^h[1-6]$")):
        level = int(heading.name[1])
        heading.replace_with(f"\n\n{'#' * level} {heading.get_text(' ', strip=True)}\n\n")
    for item in soup.find_all("li"):
        item.insert(0, "- ")
    for block in soup.find_all(["p", "li", "div", "br", "tr"]):
        block.insert_after("\n\n")
    return ParsedDocument(
        filename=path.name,
        file_type="html",
        author=author,
        created_at=_mtime_iso(path),
        sections=_split_markdown(str(soup.get_text(""))),
    )


def parse_docx(path: Path) -> ParsedDocument:
    document = docx.Document(str(path))
    blocks: list[str] = []
    for child in document.element.body.iterchildren():
        if child.tag.endswith("}p"):
            paragraph = Paragraph(child, document)
            text = paragraph.text.strip()
            if not text:
                continue
            style = str(paragraph.style.name) if paragraph.style is not None else ""
            if style == "Title":
                blocks.append(f"# {text}")
            elif style.startswith("Heading "):
                level = style.removeprefix("Heading ").strip()
                depth = int(level) if level.isdigit() else 1
                blocks.append(f"{'#' * min(depth, 6)} {text}")
            else:
                blocks.append(text)
        elif child.tag.endswith("}tbl"):
            table = Table(child, document)
            rows = [[cell.text for cell in row.cells] for row in table.rows]
            blocks.append(table_to_markdown(rows))
    props = document.core_properties
    return ParsedDocument(
        filename=path.name,
        file_type="docx",
        author=str(props.author) if props.author else None,
        created_at=props.created.isoformat() if props.created else _mtime_iso(path),
        sections=_split_markdown("\n\n".join(blocks)),
    )


_PARSERS = {
    ".pdf": parse_pdf,
    ".md": parse_markdown,
    ".markdown": parse_markdown,
    ".html": parse_html,
    ".htm": parse_html,
    ".docx": parse_docx,
}


def parse_file(path: Path) -> ParsedDocument:
    parser = _PARSERS.get(path.suffix.lower())
    if parser is None:
        raise ValueError(f"Unsupported file type: {path.suffix or path.name}")
    return parser(path)
