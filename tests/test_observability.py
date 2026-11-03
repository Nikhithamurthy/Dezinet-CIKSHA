import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from app.embeddings.embedder import MockEmbedder
from app.observability.chat import BudgetExceeded, finish_chat, prepare_chat, stream_answer
from app.observability.cost import BudgetTracker, UsageLedger, UsageRecord, estimate_cost
from app.observability.feedback import FeedbackStore, UnknownTrace
from app.observability.instrument import TracedLLM, TracedRetriever
from app.observability.latency import StreamMeter
from app.observability.runtime import Runtime
from app.observability.tracing import Tracer, TraceStore, render_tree, span
from app.rag.llm import MockLLM
from app.rag.prompts import expand_system
from app.retrieval.eval_set import load_docs
from app.retrieval.pipeline import HybridRetriever
from app.retrieval.reranker import LexicalReranker
from tests.conftest import HEADERS


def _tracer() -> Tracer:
    return Tracer(TraceStore())


def test_spans_nest_and_record_timing() -> None:
    tracer = _tracer()
    with tracer.trace("t", tenant_id="acme") as trace:
        with span("outer"):
            with span("inner", note="x"):
                pass
    by_name = {s.name: s for s in trace.spans}
    assert by_name["inner"].parent_id == by_name["outer"].span_id
    assert by_name["outer"].parent_id is None
    assert by_name["inner"].attributes == {"note": "x"}
    assert trace.finished and trace.duration_ms >= by_name["outer"].duration_ms
    assert "inner" in render_tree(trace)


def test_span_outside_a_trace_is_a_noop() -> None:
    with span("nothing") as current:
        current.attributes["x"] = 1


def test_span_records_errors_and_reraises() -> None:
    tracer = _tracer()
    with pytest.raises(RuntimeError):
        with tracer.trace("t") as trace:
            with span("boom"):
                raise RuntimeError("bad")
    assert trace.spans[0].error == "RuntimeError: bad"
    assert "RuntimeError" in trace.attributes["error"]


def test_estimate_cost() -> None:
    assert estimate_cost("gpt-4o-mini", 1_000_000, 1_000_000) == pytest.approx(0.75)
    with pytest.raises(ValueError):
        estimate_cost("unknown-model", 1, 1)


def _record(tenant: str, endpoint: str, cost: float) -> UsageRecord:
    return UsageRecord("t1", tenant, "u1", endpoint, "llm_generation", "gpt-4o", 10, 5, cost)


def test_ledger_groups_by_tenant_and_endpoint() -> None:
    ledger = UsageLedger()
    ledger.add(_record("acme", "/a", 0.5))
    ledger.add(_record("acme", "/b", 0.25))
    ledger.add(_record("globex", "/a", 1.0))
    assert ledger.totals("tenant_id")["acme"].cost_usd == pytest.approx(0.75)
    assert ledger.totals("endpoint", "acme")["/b"].calls == 1
    assert ledger.tenant_spend("globex") == pytest.approx(1.0)
    with pytest.raises(ValueError):
        ledger.totals("password")


def test_budget_allows_then_downgrades_then_blocks() -> None:
    ledger = UsageLedger()
    tracker = BudgetTracker(ledger, budgets={"acme": 1.0})
    assert tracker.decide("acme").action == "allow"
    ledger.add(_record("acme", "/a", 0.85))
    downgraded = tracker.decide("acme")
    assert downgraded.action == "downgrade" and downgraded.model == "gpt-4o-mini"
    ledger.add(_record("acme", "/a", 0.2))
    assert tracker.decide("acme").action == "block"
    assert tracker.decide("other").action == "allow"


def test_stream_meter_ttft_and_inter_token_latency() -> None:
    ticks = iter([0.0, 0.1, 0.15, 0.25])
    meter = StreamMeter(clock=lambda: next(ticks))
    for _ in range(3):
        meter.on_token()
    m = meter.summary()
    assert m.ttft_ms == pytest.approx(100)
    assert m.itl_mean_ms == pytest.approx(75)
    assert m.itl_p95_ms == pytest.approx(100)
    assert m.total_ms == pytest.approx(250)
    assert m.tokens == 3 and m.tokens_per_s == pytest.approx(2 / 0.15)


def test_traced_llm_records_tokens_cost_and_ledger_entry() -> None:
    ledger = UsageLedger()
    llm = TracedLLM(MockLLM(), ledger, default_billing="gpt-4o-mini")
    tracer = _tracer()
    with tracer.trace("t", tenant_id="acme", user_id="u1", endpoint="/x") as trace:
        asyncio.run(llm.complete(expand_system(3), "how much does the Team plan cost"))
    assert trace.spans[0].name == "query_expansion"
    attrs = trace.spans[0].attributes
    assert attrs["prompt_tokens"] > 0 and attrs["cost_usd"] > 0
    assert "prompt" not in attrs  # content is not captured by default
    assert ledger.totals("tenant_id")["acme"].calls == 1


def test_traced_retriever_matches_inner_and_emits_stage_spans() -> None:
    docs = load_docs()
    reranker = LexicalReranker()
    inner = asyncio.run(HybridRetriever.build(docs, MockEmbedder(), reranker))
    traced = TracedRetriever(inner, docs, reranker)
    query = "how do I reset my password"
    with _tracer().trace("t") as trace:
        ids = asyncio.run(traced.hybrid_rerank(query, 5))
    assert ids == asyncio.run(inner.hybrid_rerank(query, 5))
    names = {s.name for s in trace.spans}
    assert {"retrieval", "retrieval.vector", "retrieval.bm25"} <= names
    assert {"retrieval.fusion", "retrieval.rerank"} <= names
    top = next(s for s in trace.spans if s.name == "retrieval")
    assert next(s for s in trace.spans if s.name == "retrieval.vector").parent_id == top.span_id


def test_feedback_is_linked_to_a_trace_and_tenant() -> None:
    tracer = _tracer()
    store = FeedbackStore(tracer.store)
    with tracer.trace("t", tenant_id="acme") as trace:
        trace.attributes.update(question="q?", answer="wrong")
    with pytest.raises(UnknownTrace):
        store.add("missing", "acme", "u1", "up")
    with pytest.raises(UnknownTrace):
        store.add(trace.trace_id, "globex", "u1", "up")
    with pytest.raises(ValueError):
        store.add(trace.trace_id, "acme", "u1", "correction")
    store.add(trace.trace_id, "acme", "u1", "correction", correction="right answer")
    store.add(trace.trace_id, "acme", "u2", "up")
    assert store.summary() == {"up": 1, "down": 0, "correction": 1}
    cases = store.regression_cases()
    assert len(cases) == 1 and cases[0]["expected_answer"] == "right answer"


def test_chat_run_is_traced_costed_and_measured() -> None:
    rt = Runtime()

    async def scenario():  # type: ignore[no-untyped-def]
        run = await prepare_chat(
            rt, "how do I reset my password", tenant_id="acme", user_id="u1", endpoint="/t"
        )
        meter = StreamMeter(started=run.trace.t0)
        async for _ in stream_answer(run, meter=meter, delay=0.0):
            pass
        return run, finish_chat(rt, run, meter)

    run, metrics = asyncio.run(scenario())
    names = {s.name for s in run.trace.spans}
    assert {"retrieval", "retrieval.rerank", "llm_generation", "hyde", "stream"} <= names
    assert metrics.tokens > 0 and metrics.ttft_ms > 0
    assert rt.ledger.tenant_spend("acme") > 0


def test_exhausted_budget_blocks_before_any_work() -> None:
    rt = Runtime(default_budget=0.0)
    with pytest.raises(BudgetExceeded):
        asyncio.run(prepare_chat(rt, "hello", tenant_id="acme", user_id="u", endpoint="/t"))


def _events(body: str) -> list[tuple[str, dict[str, object]]]:
    out: list[tuple[str, dict[str, object]]] = []
    for block in body.strip().split("\n\n"):
        name, data = block.split("\n", 1)
        out.append((name.removeprefix("event: "), json.loads(data.removeprefix("data: "))))
    return out


def test_api_stream_feedback_and_trace_roundtrip(client: TestClient) -> None:
    headers = {**HEADERS, "X-Tenant-ID": "acme", "X-User-ID": "u1"}
    with client.stream(
        "POST", "/v1/chat/stream", json={"question": "how do I reset my password"}, headers=headers
    ) as response:
        body = "".join(response.iter_text())
    events = _events(body)
    assert events[0][0] == "meta" and events[-1][0] == "done"
    trace_id = str(events[0][1]["trace_id"])
    done = events[-1][1]
    assert done["metrics"]["ttft_ms"] > 0  # type: ignore[index]
    assert done["usage"]["cost_usd"] > 0  # type: ignore[index]
    sent = {"trace_id": trace_id, "rating": "down", "comment": "wrong"}
    assert client.post("/v1/feedback", json=sent, headers=headers).status_code == 200
    trace = client.get(f"/v1/traces/{trace_id}", headers=headers).json()
    assert {"retrieval", "llm_generation", "stream"} <= {s["name"] for s in trace["spans"]}
    other = {**HEADERS, "X-Tenant-ID": "globex"}
    assert client.get(f"/v1/traces/{trace_id}", headers=other).status_code == 404
    assert client.get("/v1/usage", headers=headers).json()["spent_usd"] > 0
