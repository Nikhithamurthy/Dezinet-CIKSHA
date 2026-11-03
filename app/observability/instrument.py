from collections.abc import Sequence

from app.ingestion.tokens import count_tokens
from app.observability.cost import UsageLedger, UsageRecord, estimate_cost
from app.observability.tracing import current_trace, span
from app.rag.llm import LLM
from app.retrieval.eval_set import Doc
from app.retrieval.pipeline import HybridRetriever
from app.retrieval.reranker import Reranker
from app.retrieval.rrf import reciprocal_rank_fusion

STAGES = (
    ("TASK: EXPAND", "query_expansion"),
    ("TASK: HYDE", "hyde"),
    ("TASK: ANSWER", "llm_generation"),
    ("TASK: CONDENSE", "query_rewrite"),
    ("TASK: SUMMARI", "summarize"),
)


class TracedLLM:
    """Wraps any LLM: one span per call with tokens and estimated cost, plus a
    ledger entry tagged with tenant, user and endpoint from the active trace.
    Prompts and outputs are NOT stored unless capture_content=True (they may hold PII)."""

    def __init__(
        self,
        inner: LLM,
        ledger: UsageLedger,
        *,
        default_billing: str = "gpt-4o",
        capture_content: bool = False,
    ) -> None:
        self.name = inner.name
        self._inner = inner
        self._ledger = ledger
        self._default = default_billing
        self._capture = capture_content

    async def complete(self, system: str, user: str) -> str:
        trace = current_trace()
        billing = self._default
        if trace is not None:
            billing = str(trace.attributes.get("billing_model", self._default))
        stage = next((label for prefix, label in STAGES if system.startswith(prefix)), "llm_call")
        with span(stage, model=self._inner.name, billing_model=billing) as current:
            output = await self._inner.complete(system, user)
            prompt_tokens = count_tokens(system) + count_tokens(user)
            completion_tokens = count_tokens(output)
            cost = round(estimate_cost(billing, prompt_tokens, completion_tokens), 8)
            current.attributes.update(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost,
            )
            if self._capture:
                current.attributes.update(
                    prompt=(system + "\n" + user)[:2000], output=output[:2000]
                )
        if trace is not None:
            self._ledger.add(
                UsageRecord(
                    trace.trace_id,
                    trace.tenant_id,
                    trace.user_id,
                    trace.endpoint,
                    stage,
                    billing,
                    prompt_tokens,
                    completion_tokens,
                    cost,
                )
            )
        return output


class TracedRetriever:
    """Same result as HybridRetriever.hybrid_rerank, but each stage gets its own span:
    vector, BM25, fusion and rerank. That shows which stage is slow."""

    def __init__(
        self,
        inner: HybridRetriever,
        docs: Sequence[Doc],
        reranker: Reranker | None,
        *,
        candidates: int = 50,
        rrf_k: int = 60,
    ) -> None:
        self._inner = inner
        self._texts = {doc.id: doc.text for doc in docs}
        self._reranker = reranker
        self._candidates = candidates
        self._rrf_k = rrf_k

    async def hybrid_rerank(self, query: str, k: int = 5) -> list[str]:
        with span("retrieval", query_chars=len(query), k=k) as top:
            with span("retrieval.vector") as vector_span:
                dense = await self._inner.vector_search(query, self._candidates)
                vector_span.attributes["returned"] = len(dense)
            with span("retrieval.bm25") as bm25_span:
                sparse = await self._inner.bm25_search(query, self._candidates)
                bm25_span.attributes["returned"] = len(sparse)
            with span("retrieval.fusion") as fusion_span:
                fused_pairs = reciprocal_rank_fusion([dense, sparse], k=self._rrf_k)
                fused = [doc_id for doc_id, _ in fused_pairs[: self._candidates]]
                fusion_span.attributes["returned"] = len(fused)
            ranked = fused[:k]
            if self._reranker is not None:
                with span("retrieval.rerank", reranker=self._reranker.name) as rerank_span:
                    pairs = [(doc_id, self._texts[doc_id]) for doc_id in fused]
                    scored = await self._reranker.rerank(query, pairs, k)
                    ranked = [doc_id for doc_id, _ in scored]
                    rerank_span.attributes["candidates_in"] = len(pairs)
                    rerank_span.attributes["returned"] = len(ranked)
            top.attributes["top_ids"] = ranked
        return ranked
