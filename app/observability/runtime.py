import asyncio
import os
from pathlib import Path
from typing import cast

from app.embeddings.embedder import MockEmbedder
from app.observability.cost import BudgetTracker, UsageLedger
from app.observability.feedback import FeedbackStore
from app.observability.instrument import TracedLLM, TracedRetriever
from app.observability.tracing import Tracer, TraceStore
from app.rag.llm import MockLLM
from app.rag.pipeline import RagPipeline
from app.retrieval.eval_set import load_docs
from app.retrieval.pipeline import HybridRetriever
from app.retrieval.reranker import LexicalReranker


class Runtime:
    """Process-wide observability state. Everything is in memory, so it resets
    when the server restarts (a production system would use Redis or a database)."""

    def __init__(
        self, *, default_budget: float | None = None, budgets: dict[str, float] | None = None
    ) -> None:
        trace_file = os.getenv("TRACE_FILE")
        self.store = TraceStore(path=Path(trace_file) if trace_file else None)
        self.tracer = Tracer(self.store)
        self.ledger = UsageLedger()
        if default_budget is None:
            default_budget = float(os.getenv("TENANT_BUDGET_USD", "5.0"))
        self.budget = BudgetTracker(self.ledger, default_budget=default_budget, budgets=budgets)
        self.feedback = FeedbackStore(self.store)
        self._pipeline: RagPipeline | None = None
        self._lock = asyncio.Lock()

    async def get_pipeline(self) -> RagPipeline:
        async with self._lock:
            if self._pipeline is None:
                docs = load_docs()
                reranker = LexicalReranker()
                inner = await HybridRetriever.build(docs, MockEmbedder(), reranker)
                retriever = TracedRetriever(inner, docs, reranker)
                llm = TracedLLM(MockLLM(), self.ledger)
                self._pipeline = RagPipeline(cast(HybridRetriever, retriever), docs, llm)
            return self._pipeline


runtime = Runtime()
