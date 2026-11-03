import argparse
import asyncio
from typing import cast

from app.rag.grounding import support_score
from app.rag.llm import get_llm
from app.rag.pipeline import RagPipeline
from app.retrieval.eval_set import Doc
from app.retrieval.pipeline import HybridRetriever

DOCS = [
    Doc("d1", "Refunds for annual plans are issued within 30 days of purchase."),
    Doc("d2", "Support is available by email on weekdays from 9am to 6pm."),
    Doc("d3", "Passwords must be reset every 90 days and be at least 12 characters."),
    Doc("d4", "Invoices are emailed on the first day of each billing cycle."),
]

QUESTIONS = [
    "How do refunds work for annual plans?",
    "When is support available?",
    "What is the CEO's favourite colour?",  # unanswerable: should fall back
]


class LexicalRetriever:
    """Stand-in for HybridRetriever so the demo runs with no database or models."""

    async def hybrid_rerank(self, query: str, k: int = 5) -> list[str]:
        scored = sorted(((support_score(query, d.text), d.id) for d in DOCS), reverse=True)
        return [doc_id for score, doc_id in scored if score > 0][:k]


async def main(kind: str) -> None:
    retriever = cast(HybridRetriever, LexicalRetriever())
    pipeline = RagPipeline(retriever, DOCS, get_llm(kind))
    for question in QUESTIONS:
        result = await pipeline.ask(question)
        print(f"\nQ: {question}")
        print(f"status={result.status} grounded={result.grounded} support={result.support:.2f}")
        print(f"A: {result.answer}")
        print(f"citations={[c.chunk_id for c in result.citations]} queries={len(result.queries)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", default="mock", choices=["mock", "openai"])
    asyncio.run(main(parser.parse_args().llm))
