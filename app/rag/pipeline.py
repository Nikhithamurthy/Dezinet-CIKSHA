from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from app.rag.citations import Citation, map_citations
from app.rag.context import ContextChunk, assemble_context
from app.rag.grounding import support_score
from app.rag.llm import LLM
from app.rag.prompts import FALLBACK_MESSAGE, GROUNDED_SYSTEM, NO_ANSWER
from app.rag.query_transform import expand_queries, hyde_passage
from app.retrieval.eval_set import Doc
from app.retrieval.pipeline import HybridRetriever
from app.retrieval.rrf import reciprocal_rank_fusion


@dataclass
class RagResult:
    question: str
    answer: str
    status: str  # "answered" or "fallback"
    citations: list[Citation] = field(default_factory=list)
    unsupported_citations: list[str] = field(default_factory=list)
    context_ids: list[str] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    support: float = 0.0
    hyde: str | None = None

    @property
    def grounded(self) -> bool:
        """A refusal is grounded. An answer needs at least one valid citation
        and no invented ones."""
        if self.status == "fallback":
            return True
        return bool(self.citations) and not self.unsupported_citations


class RagPipeline:
    """Query transform -> hybrid retrieval per query -> RRF -> support check
    -> context assembly -> grounded generation -> citation mapping."""

    def __init__(
        self,
        retriever: HybridRetriever,
        docs: Sequence[Doc],
        llm: LLM,
        *,
        meta: Mapping[str, tuple[str, int | None]] | None = None,
        use_expansion: bool = True,
        use_hyde: bool = True,
        n_queries: int = 3,
        per_query_k: int = 5,
        context_k: int = 4,
        min_support: float = 0.5,
    ) -> None:
        self._retriever = retriever
        self._docs = {doc.id: doc for doc in docs}
        self._llm = llm
        self._meta = meta or {}
        self._use_expansion = use_expansion
        self._use_hyde = use_hyde
        self._n_queries = n_queries
        self._per_query_k = per_query_k
        self._context_k = context_k
        self._min_support = min_support

    async def retrieve(self, question: str) -> tuple[list[ContextChunk], list[str], str | None]:
        queries = [question]
        if self._use_expansion:
            queries += await expand_queries(self._llm, question, self._n_queries)
        hypothetical: str | None = None
        if self._use_hyde:
            hypothetical = await hyde_passage(self._llm, question)
            if hypothetical:
                queries.append(hypothetical)
        rankings = [await self._retriever.hybrid_rerank(q, self._per_query_k) for q in queries]
        fused = reciprocal_rank_fusion(rankings)[: self._context_k]
        chunks: list[ContextChunk] = []
        for doc_id, _ in fused:
            doc = self._docs[doc_id]
            source, page = self._meta.get(doc_id, ("corpus.json", None))
            chunks.append(
                ContextChunk(
                    chunk_id=doc_id,
                    text=doc.text,
                    source=source,
                    page=page,
                    score=support_score(question, doc.text),
                )
            )
        return chunks, queries, hypothetical

    async def ask(self, question: str) -> RagResult:
        chunks, queries, hypothetical = await self.retrieve(question)
        ids = [chunk.chunk_id for chunk in chunks]
        best = max((chunk.score for chunk in chunks), default=0.0)
        if best < self._min_support:
            # Weak evidence: refuse without spending an LLM call.
            return RagResult(
                question=question,
                answer=FALLBACK_MESSAGE,
                status="fallback",
                context_ids=ids,
                queries=queries,
                support=best,
                hyde=hypothetical,
            )
        prompt = f"{assemble_context(chunks)}\n\nQuestion: {question}"
        answer = (await self._llm.complete(GROUNDED_SYSTEM, prompt)).strip()
        if answer.startswith(NO_ANSWER):
            return RagResult(
                question=question,
                answer=answer,
                status="fallback",
                context_ids=ids,
                queries=queries,
                support=best,
                hyde=hypothetical,
            )
        citations, invalid = map_citations(answer, chunks)
        return RagResult(
            question=question,
            answer=answer,
            status="answered",
            citations=citations,
            unsupported_citations=invalid,
            context_ids=ids,
            queries=queries,
            support=best,
            hyde=hypothetical,
        )
