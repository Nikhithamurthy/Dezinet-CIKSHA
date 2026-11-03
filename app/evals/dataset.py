"""Synthetic golden set: 50 question / ground-truth pairs generated from source documents."""

import json
import re
from dataclasses import asdict
from itertools import zip_longest
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from app.evals.core import Document, GoldenItem, support
from app.rag.llm import LLM
from app.tools.structured import extract_structured

GROUNDING_MIN = 0.8
_PLANS = (
    "Starter", "Growth", "Business", "Enterprise", "Education",
    "Nonprofit", "Developer", "Startup", "Partner", "Premium",
)  # fmt: skip
_PRICES = (12, 29, 79, 199, 9, 0, 19, 39, 59, 99)
_SEATS = (5, 25, 100, 500, 50, 20, 3, 15, 10, 250)
_STORAGE = (10, 100, 500, 2000, 250, 150, 50, 75, 300, 1000)
_SUPPORT = (
    "email", "live chat", "phone", "a dedicated manager", "a community forum",
    "a ticket portal", "Slack", "video calls", "WhatsApp", "postal mail",
)  # fmt: skip
_REFUND = (14, 30, 45, 60, 7, 21, 10, 15, 28, 90)
_FACT = re.compile(r"^The (?P<attr>.+?) of the (?P<entity>.+?) is (?P<value>.+)\.$")


class QAPair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=5, max_length=300)
    ground_truth: str = Field(min_length=3, max_length=500)


class QAPairs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pairs: list[QAPair]


def build_corpus() -> list[Document]:
    """Ten product documents with five facts each: enough for exactly 50 golden pairs."""
    docs: list[Document] = []
    for i, plan in enumerate(_PLANS):
        name = f"{plan} plan"
        facts = (
            f"The price of the {name} is ${_PRICES[i]} per month.",
            f"The seat limit of the {name} is {_SEATS[i]} seats.",
            f"The storage quota of the {name} is {_STORAGE[i]} GB.",
            f"The support channel of the {name} is {_SUPPORT[i]}.",
            f"The refund window of the {name} is {_REFUND[i]} days.",
        )
        docs.append(Document(name, " ".join(facts)))
    return docs


class QuestionWriter(Protocol):
    async def write(self, document: Document) -> list[QAPair]: ...


class TemplateWriter:
    """Offline writer: turns 'The X of the Y is Z.' sentences into questions."""

    async def write(self, document: Document) -> list[QAPair]:
        pairs: list[QAPair] = []
        for sentence in document.text.split(". "):
            fact = sentence if sentence.endswith(".") else sentence + "."
            match = _FACT.match(fact)
            if match:
                question = f"What is the {match['attr']} of the {match['entity']}?"
                pairs.append(QAPair(question=question, ground_truth=fact))
        return pairs


class LlmWriter:
    """LLM-as-generator: asks the model for pairs and validates them with Pydantic."""

    def __init__(self, llm: LLM, per_doc: int = 5) -> None:
        self._llm = llm
        self._per_doc = per_doc

    async def write(self, document: Document) -> list[QAPair]:
        prompt = (
            f"Write {self._per_doc} question/ground_truth pairs answerable only from the "
            "document below. Each ground_truth must be copied or closely paraphrased from it.\n\n"
            f"Title: {document.title}\n{document.text}"
        )
        result = await extract_structured(self._llm, QAPairs, prompt)
        return result.pairs


async def generate_golden(
    docs: list[Document], writer: QuestionWriter, n: int = 50
) -> list[GoldenItem]:
    """Interleave documents, drop duplicates, and drop pairs whose ground truth is not grounded
    in the source document (this filter catches hallucinated pairs from an LLM writer)."""
    per_doc = [await writer.write(doc) for doc in docs]
    items: list[GoldenItem] = []
    seen: set[str] = set()
    for group in zip_longest(*per_doc):
        for doc, pair in zip(docs, group, strict=True):
            if pair is None:
                continue
            key = pair.question.strip().lower()
            if key in seen or support(pair.ground_truth, doc.text) < GROUNDING_MIN:
                continue
            seen.add(key)
            items.append(GoldenItem(pair.question.strip(), pair.ground_truth.strip(), doc.title))
            if len(items) == n:
                return items
    return items


def save_golden(items: list[GoldenItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(asdict(item), ensure_ascii=False) for item in items]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_golden(path: Path) -> list[GoldenItem]:
    text = path.read_text(encoding="utf-8")
    return [GoldenItem(**json.loads(line)) for line in text.splitlines() if line.strip()]
