import json
from collections.abc import AsyncIterator
from dataclasses import asdict
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.dependencies import rate_limit, verify_api_key
from app.observability.chat import BudgetExceeded, finish_chat, prepare_chat, stream_answer
from app.observability.feedback import UnknownTrace
from app.observability.latency import StreamMeter
from app.observability.runtime import runtime

router = APIRouter(prefix="/v1", tags=["observability"])
TenantHeader = Annotated[str, Header()]
UserHeader = Annotated[str, Header()]


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class FeedbackRequest(BaseModel):
    trace_id: str = Field(min_length=8, max_length=64)
    rating: Literal["up", "down", "correction"]
    comment: str | None = Field(default=None, max_length=1000)
    correction: str | None = Field(default=None, max_length=2000)


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/chat/stream", dependencies=[Depends(rate_limit)])
async def chat_stream(
    payload: ChatRequest,
    request: Request,
    x_tenant_id: TenantHeader = "default",
    x_user_id: UserHeader = "anonymous",
) -> StreamingResponse:
    """Traced RAG answer over SSE. The first event carries the trace_id, the last
    carries TTFT/ITL metrics, token usage and cost."""
    try:
        run = await prepare_chat(
            runtime,
            payload.question,
            tenant_id=x_tenant_id,
            user_id=x_user_id,
            endpoint="/v1/chat/stream",
        )
    except BudgetExceeded as exc:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Tenant budget exhausted (${exc.decision.spent:.4f} of ${exc.decision.budget:.4f})"
            ),
            headers={"Retry-After": "3600"},
        ) from exc

    async def events() -> AsyncIterator[str]:
        meter = StreamMeter(started=run.trace.t0)
        try:
            meta = {
                "trace_id": run.trace.trace_id,
                "status": run.result.status,
                "model_tier": run.decision.action,
            }
            yield _sse("meta", meta)
            async for token in stream_answer(
                run, meter=meter, is_disconnected=request.is_disconnected
            ):
                yield _sse("token", {"token": token})
            metrics = finish_chat(runtime, run, meter)
            yield _sse(
                "done",
                {
                    "trace_id": run.trace.trace_id,
                    "metrics": asdict(metrics),
                    "usage": asdict(runtime.ledger.trace_total(run.trace.trace_id)),
                    "citations": [asdict(c) for c in run.result.citations],
                },
            )
        finally:
            if not run.trace.finished:
                finish_chat(runtime, run, meter)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/feedback", dependencies=[Depends(verify_api_key)])
async def submit_feedback(
    payload: FeedbackRequest,
    x_tenant_id: TenantHeader = "default",
    x_user_id: UserHeader = "anonymous",
) -> dict[str, Any]:
    try:
        record = runtime.feedback.add(
            payload.trace_id,
            x_tenant_id,
            x_user_id,
            payload.rating,
            payload.comment,
            payload.correction,
        )
    except UnknownTrace as exc:
        raise HTTPException(status_code=404, detail="Trace not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return asdict(record)


@router.get("/traces/{trace_id}", dependencies=[Depends(verify_api_key)])
async def get_trace(trace_id: str, x_tenant_id: TenantHeader = "default") -> dict[str, Any]:
    trace = runtime.store.get(trace_id)
    if trace is None or trace.tenant_id != x_tenant_id:
        raise HTTPException(status_code=404, detail="Trace not found")
    return trace.to_dict()


@router.get("/usage", dependencies=[Depends(verify_api_key)])
async def get_usage(x_tenant_id: TenantHeader = "default") -> dict[str, Any]:
    by_endpoint = runtime.ledger.totals("endpoint", x_tenant_id)
    return {
        "tenant_id": x_tenant_id,
        "spent_usd": round(runtime.ledger.tenant_spend(x_tenant_id), 8),
        "budget": asdict(runtime.budget.decide(x_tenant_id)),
        "by_endpoint": {key: asdict(value) for key, value in by_endpoint.items()},
    }
