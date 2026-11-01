import re
import unicodedata

_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff\u00ad"), None)


def clean_text(text: str) -> str:
    """Normalize unicode, strip invisible junk, collapse whitespace."""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_ZERO_WIDTH)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\t\u2000-\u200a\u202f\u205f\u3000]+", " ", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join words hyphenated across lines
    text = re.sub(r" +\n", "\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def table_to_markdown(rows: list[list[str]]) -> str:
    """Keep row/column relationships by emitting a Markdown table."""
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    norm = [[c.replace("\n", " ").strip() for c in row] + [""] * (width - len(row)) for row in rows]
    header, *body = norm
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)
