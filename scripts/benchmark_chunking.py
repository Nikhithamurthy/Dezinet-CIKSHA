import sys
import time
from pathlib import Path

from app.ingestion.parsers import parse_file
from app.ingestion.pipeline import STRATEGIES, chunk_document


def main() -> None:
    paths = [Path(a) for a in sys.argv[1:]] or sorted(
        p for p in Path("data/samples").iterdir() if p.is_file()
    )
    docs = [parse_file(p) for p in paths]
    print(f"Files: {', '.join(d.filename for d in docs)}\n")
    print("| Strategy | Chunks | Avg tokens | Max tokens | Mid-sentence cuts | Time (ms) |")
    print("|---|---|---|---|---|---|")
    for name in STRATEGIES:
        start = time.perf_counter()
        chunks = [c for d in docs for c in chunk_document(d, name)]
        elapsed = (time.perf_counter() - start) * 1000
        tokens = [c.token_count for c in chunks]
        cuts = sum(1 for c in chunks if c.text.rstrip()[-1] not in ".!?:|)")
        print(
            f"| {name} | {len(chunks)} | {sum(tokens) / len(tokens):.0f} | {max(tokens)} "
            f"| {cuts / len(chunks):.0%} | {elapsed:.0f} |"
        )


if __name__ == "__main__":
    main()
