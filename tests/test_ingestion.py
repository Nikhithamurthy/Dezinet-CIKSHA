from pathlib import Path

import docx

from app.ingestion.chunkers import recursive_split, semantic_split, sliding_window
from app.ingestion.cleaning import clean_text, table_to_markdown
from app.ingestion.parsers import parse_docx, parse_file, parse_html, parse_markdown
from app.ingestion.pipeline import STRATEGIES, ingest_file
from app.ingestion.tokens import count_tokens

SAMPLES = Path(__file__).resolve().parent.parent / "data" / "samples"
MD = SAMPLES / "handbook.md"
HTML = SAMPLES / "help_center.html"


def test_clean_text_normalizes() -> None:
    assert clean_text("A\u00a0\u00a0B\u200bC\n\n\n\n\nD  E") == "A BC\n\nD E"
    assert clean_text("inter-\nnational") == "international"


def test_table_to_markdown() -> None:
    rows = [["Plan", "Price"], ["Pro", "$10"]]
    assert table_to_markdown(rows) == "| Plan | Price |\n| --- | --- |\n| Pro | $10 |"


def test_count_tokens() -> None:
    assert count_tokens("hello world") >= 2


def test_parse_markdown_sections() -> None:
    doc = parse_markdown(MD)
    headers = [s.header for s in doc.sections]
    assert "Leave Policy" in headers
    assert "Pricing" in headers


def test_parse_html_strips_noise_and_keeps_tables() -> None:
    doc = parse_html(HTML)
    text = "\n".join(s.text for s in doc.sections)
    assert "tracking" not in text
    assert "Home | Pricing" not in text
    assert "| Plan | Price | Seats |" in text
    assert doc.author == "Dezinet Docs Team"


def test_parse_docx(tmp_path: Path) -> None:
    d = docx.Document()
    d.add_heading("Leave Policy", level=1)
    d.add_paragraph("Employees get 20 days of leave.")
    table = d.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Plan"
    table.cell(0, 1).text = "Price"
    table.cell(1, 0).text = "Pro"
    table.cell(1, 1).text = "$10"
    path = tmp_path / "policy.docx"
    d.save(str(path))
    parsed = parse_docx(path)
    assert parsed.sections[0].header == "Leave Policy"
    assert "| Plan | Price |" in parsed.sections[0].text


def test_unsupported_file_type() -> None:
    try:
        parse_file(Path("malware.exe"))
    except ValueError as exc:
        assert "Unsupported" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_recursive_respects_token_budget() -> None:
    text = MD.read_text(encoding="utf-8") * 3
    chunks = recursive_split(text, chunk_tokens=200, overlap_tokens=30)
    assert len(chunks) > 1
    assert all(count_tokens(c) <= 200 for c in chunks)


def test_sliding_window_overlaps() -> None:
    text = " ".join(f"word{i}" for i in range(300))
    chunks = sliding_window(text, window_tokens=50, overlap_tokens=10)
    assert len(chunks) > 2
    assert set(chunks[0].split()[-10:]) & set(chunks[1].split()[:10])


def test_semantic_split_breaks_on_topic_change() -> None:
    cats = "Cats purr and cats sleep all day. " * 6
    money = "Invoices are paid monthly and invoices include tax. " * 6
    chunks = semantic_split(cats + money, max_tokens=500, min_tokens=20)
    assert len(chunks) >= 2


def test_chunks_carry_metadata() -> None:
    chunks = ingest_file(MD, "recursive")
    assert chunks
    assert all(c.source == "handbook.md" and c.token_count > 0 for c in chunks)
    assert any(c.section_header == "Pricing" for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_every_strategy_runs() -> None:
    for name in STRATEGIES:
        assert ingest_file(HTML, name)
