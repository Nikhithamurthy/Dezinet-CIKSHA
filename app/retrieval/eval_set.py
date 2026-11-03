import json
from dataclasses import dataclass
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parents[2] / "data" / "eval"


@dataclass(frozen=True)
class Doc:
    id: str
    text: str


@dataclass(frozen=True)
class EvalQuery:
    category: str
    query: str
    relevant: tuple[str, ...]


def load_docs() -> list[Doc]:
    raw = json.loads((EVAL_DIR / "corpus.json").read_text(encoding="utf-8"))
    return [Doc(id=item["id"], text=item["text"]) for item in raw]


def load_queries() -> list[EvalQuery]:
    raw = json.loads((EVAL_DIR / "queries.json").read_text(encoding="utf-8"))
    return [EvalQuery(item["category"], item["query"], tuple(item["relevant"])) for item in raw]
