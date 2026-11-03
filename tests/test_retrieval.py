import asyncio

import pytest

from app.embeddings.embedder import MockEmbedder
from app.retrieval.bm25 import BM25Index, tokenize
from app.retrieval.eval_set import load_docs, load_queries
from app.retrieval.metrics import hit_at_k, precision_at_k, recall_at_k, reciprocal_rank
from app.retrieval.pipeline import HybridRetriever
from app.retrieval.reranker import LexicalReranker
from app.retrieval.rrf import reciprocal_rank_fusion


def test_rrf_formula() -> None:
    fused = dict(reciprocal_rank_fusion([["a", "b", "c"], ["c", "a"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["b"] == pytest.approx(1 / 62)
    assert fused["c"] == pytest.approx(1 / 63 + 1 / 61)


def test_rrf_rewards_agreement_between_lists() -> None:
    fused = reciprocal_rank_fusion([["x", "a"], ["y", "a"]])
    assert fused[0][0] == "a"


def test_tokenizer_keeps_identifiers_whole() -> None:
    assert tokenize("See ERR-5034, clause 9.4.") == ["see", "err-5034", "clause", "9.4"]


def test_bm25_finds_exact_identifier() -> None:
    docs = [(d.id, d.text) for d in load_docs()]
    hits = BM25Index(docs).search("ERR-5034", k=3)
    assert [doc_id for doc_id, _ in hits] == ["d01"]


def test_metrics() -> None:
    ranked = ["a", "b", "c"]
    assert reciprocal_rank(ranked, {"c"}, 5) == pytest.approx(1 / 3)
    assert reciprocal_rank(ranked, {"c"}, 2) == 0.0
    assert hit_at_k(ranked, {"b"}, 2) == 1.0
    assert precision_at_k(["a", "b"], {"a"}, 2) == 0.5
    assert recall_at_k(ranked, {"a", "z"}, 3) == 0.5


def test_eval_data_is_consistent() -> None:
    ids = {doc.id for doc in load_docs()}
    queries = load_queries()
    assert len(queries) >= 20
    assert all(rel in ids for q in queries for rel in q.relevant)


def test_hybrid_returns_exact_id_match_first() -> None:
    retriever = asyncio.run(HybridRetriever.build(load_docs(), MockEmbedder()))
    assert asyncio.run(retriever.hybrid("ERR-5034", 5))[0] == "d01"


def test_lexical_reranker_prefers_the_matching_document() -> None:
    docs = [("a", "pricing plans per user"), ("b", "reset your password from the login page")]
    ranked = asyncio.run(LexicalReranker().rerank("reset password", docs, 2))
    assert ranked[0][0] == "b"


def test_hybrid_rerank_returns_top_k() -> None:
    retriever = asyncio.run(HybridRetriever.build(load_docs(), MockEmbedder(), LexicalReranker()))
    assert len(asyncio.run(retriever.hybrid_rerank("how do I reset my password", 5))) == 5
