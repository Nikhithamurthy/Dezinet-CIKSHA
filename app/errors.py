import logging
from collections.abc import Mapping
from typing import Any, cast

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("ciksha")

_CODES = {401: "unauthorized", 404: "not_found", 429: "rate_limited"}


def _error(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    details: Any = None,
    headers: Mapping[str, str] | None = None,
) -> JSONResponse:
    body = {
        "error": {
            "code": code,
            "message": message,
            "details": details,
            "request_id": getattr(request.state, "request_id", None),
        }
    }
    return JSONResponse(status_code=status_code, content=body, headers=headers)


async def http_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    http_exc = cast(StarletteHTTPException, exc)
    code = _CODES.get(http_exc.status_code, f"http_{http_exc.status_code}")
    return _error(
        request, http_exc.status_code, code, str(http_exc.detail), headers=http_exc.headers
    )


async def validation_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    val_exc = cast(RequestValidationError, exc)
    details = [
        {
            "field": ".".join(str(part) for part in err["loc"][1:]),
            "message": str(err["msg"]).removeprefix("Value error, "),
        }
        for err in val_exc.errors()
    ]
    return _error(request, 422, "validation_error", "Request validation failed", details)


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return _error(request, 500, "internal_error", "Internal server error")


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
