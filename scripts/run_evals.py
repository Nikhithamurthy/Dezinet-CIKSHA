"""CI gate: score the RAG pipeline on the golden set. Exit code 1 if any metric regresses.

uv run python -m scripts.run_evals                # uses data/golden_qa.jsonl
uv run python -m scripts.run_evals --regenerate   # rebuild the golden set first"""

import asyncio
import sys
from pathlib import Path

from app.evals.dataset import (
    TemplateWriter,
    build_corpus,
    generate_golden,
    load_golden,
    save_golden,
)
from app.evals.pipeline import ExtractiveAnswerer, LexicalRetriever, RagPipeline
from app.evals.runner import evaluate
from app.safety.guard import Guard

GOLDEN_PATH = Path("data") / "golden_qa.jsonl"


async def main() -> int:
    docs = build_corpus()
    if GOLDEN_PATH.exists() and "--regenerate" not in sys.argv:
        golden = load_golden(GOLDEN_PATH)
    else:
        golden = await generate_golden(docs, TemplateWriter(), n=50)
        save_golden(golden, GOLDEN_PATH)
        print(f"wrote {len(golden)} golden pairs to {GOLDEN_PATH}")
    pipeline = RagPipeline(LexicalRetriever(docs), ExtractiveAnswerer(), guard=Guard())
    report = await evaluate(pipeline, golden)
    print(report.format())
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
