import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.config import APP_VERSION, get_settings
from app.db import DatabasePool
from app.errors import register_exception_handlers
from app.routers import ai, observability


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.db_pool = DatabasePool(size=5)  # startup
    app.state.started_at = time.time()
    yield
    await app.state.db_pool.close()  # shutdown


settings = get_settings()
app = FastAPI(title="Dezinet CIKSHA", version=APP_VERSION, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time-Ms", "X-Request-ID"],
)
register_exception_handlers(app)
app.include_router(ai.router)
app.include_router(observability.router)


class Health(BaseModel):
    status: str


@app.get("/health")
async def health() -> Health:
    return Health(status="ok")


@app.middleware("http")
async def timing_and_request_id(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    request.state.request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - start) * 1000:.2f}"
    response.headers["X-Request-ID"] = request.state.request_id
    return response
