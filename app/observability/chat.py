import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import asdict, dataclass

from app.observability.cost import BudgetDecision
from app.observability.latency import StreamMeter, StreamMetrics
from app.observability.runtime import Runtime
from app.observability.tracing import Trace
from app.rag.pipeline import RagResult
from app.safety.pii import redact

STREAM_DELAY_S = 0.02  # simulated pacing: the mock LLM does not stream by itself


class BudgetExceeded(Exception):
    def __init__(self, decision: BudgetDecision) -> None:
        super().__init__("tenant budget exhausted")
        self.decision = decision


@dataclass
class ChatRun:
    trace: Trace
    result: RagResult
    decision: BudgetDecision


async def prepare_chat(
    rt: Runtime, question: str, *, tenant_id: str, user_id: str, endpoint: str
) -> ChatRun:
    """Budget check, then run the RAG pipeline under a trace. PII is redacted
    before the question reaches the pipeline or the trace."""
    decision = rt.budget.decide(tenant_id)
    if decision.action == "block":
        raise BudgetExceeded(decision)
    trace = rt.tracer.start("chat", tenant_id=tenant_id, user_id=user_id, endpoint=endpoint)
    clean = redact(question).text
    trace.attributes.update(
        question=clean, billing_model=decision.model, budget_action=decision.action
    )
    pipeline = await rt.get_pipeline()
    try:
        with rt.tracer.activate(trace):
            result = await pipeline.ask(clean)
    except Exception as exc:
        trace.attributes["error"] = f"{type(exc).__name__}: {exc}"
        rt.tracer.finish(trace)
        raise
    trace.attributes["answer"] = redact(result.answer).text
    trace.attributes["status"] = result.status
    return ChatRun(trace, result, decision)


async def stream_answer(
    run: ChatRun,
    *,
    meter: StreamMeter,
    delay: float = STREAM_DELAY_S,
    is_disconnected: Callable[[], Awaitable[bool]] | None = None,
) -> AsyncIterator[str]:
    for word in run.result.answer.split():
        if is_disconnected is not None and await is_disconnected():
            run.trace.attributes["client_disconnected"] = True
            return
        await asyncio.sleep(delay)
        meter.on_token()
        yield word + " "


def finish_chat(rt: Runtime, run: ChatRun, meter: StreamMeter) -> StreamMetrics:
    metrics = meter.summary()
    run.trace.attributes["metrics"] = asdict(metrics)
    if metrics.tokens:
        run.trace.add_span(
            "stream", metrics.ttft_ms, metrics.total_ms - metrics.ttft_ms, tokens=metrics.tokens
        )
    rt.tracer.finish(run.trace)
    return metrics
