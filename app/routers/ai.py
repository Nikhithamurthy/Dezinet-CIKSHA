import json
import time
from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.config import APP_VERSION
from app.db import DBSession
from app.dependencies import get_db_session, rate_limit
from app.schemas import (
    EmbeddingRequest,
    EmbeddingResponse,
    InferenceRequest,
    InferenceResponse,
    StatusResponse,
)
from app.services import llm

router = APIRouter(prefix="/v1", tags=["ai"])


def sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.get("/status", response_model=StatusResponse)
async def get_status(
    request: Request, db: Annotated[DBSession, Depends(get_db_session)]
) -> StatusResponse:
    return StatusResponse(
        status="ok",
        version=APP_VERSION,
        uptime_seconds=round(time.time() - request.app.state.started_at, 2),
        database=await db.ping(),
    )


@router.post("/infer", response_model=InferenceResponse, dependencies=[Depends(rate_limit)])
async def infer(payload: InferenceRequest) -> InferenceResponse:
    tokens = [t async for t in llm.stream_tokens(payload.prompt, payload.max_tokens)]
    return InferenceResponse(
        model=payload.model,
        output="".join(tokens).strip(),
        prompt_tokens=len(payload.prompt.split()),
        completion_tokens=len(tokens),
    )


@router.post("/embeddings", response_model=EmbeddingResponse, dependencies=[Depends(rate_limit)])
async def create_embeddings(payload: EmbeddingRequest) -> EmbeddingResponse:
    vectors = await llm.embed_texts(payload.texts)
    return EmbeddingResponse(model=payload.model, dimensions=len(vectors[0]), embeddings=vectors)


@router.post("/stream", dependencies=[Depends(rate_limit)])
async def stream_sse(payload: InferenceRequest, request: Request) -> StreamingResponse:
    """Server-Sent Events: one `token` event per token, then a `done` event."""

    async def event_generator() -> AsyncIterator[str]:
        count = 0
        async for token in llm.stream_tokens(payload.prompt, payload.max_tokens):
            if await request.is_disconnected():
                return  # client left: stop generating, stop paying for tokens
            count += 1
            yield sse("token", {"token": token, "index": count})
        yield sse("done", {"completion_tokens": count})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/stream/chunked", dependencies=[Depends(rate_limit)])
async def stream_chunked(payload: InferenceRequest, request: Request) -> StreamingResponse:
    """Plain HTTP chunked transfer: raw tokens, no SSE framing."""

    async def token_generator() -> AsyncIterator[str]:
        async for token in llm.stream_tokens(payload.prompt, payload.max_tokens):
            if await request.is_disconnected():
                return
            yield token

    return StreamingResponse(token_generator(), media_type="text/plain; charset=utf-8")
