"""Request correlation IDs and structured JSON access logs."""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from collections.abc import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from src.api.metrics import METRICS, UNMATCHED

REQUEST_ID_HEADER = "X-Request-ID"
LOGGER_NAME = "hilbert_rl.api"
# Only accept simple client-supplied ids so they cannot be abused to forge log lines.
_VALID_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

logger = logging.getLogger(LOGGER_NAME)


class JsonFormatter(logging.Formatter):
    """Format log records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        """Render ``record`` (message plus any ``fields`` extra) as JSON."""
        payload = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}))
        return json.dumps(payload)


def configure_logging() -> None:
    """Send the API logger to stderr as JSON (once; safe to call repeatedly)."""
    if any(isinstance(h.formatter, JsonFormatter) for h in logger.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach a correlation id to every request and log one line per request."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Time the request, tag it with an id, and log the outcome."""
        incoming = request.headers.get(REQUEST_ID_HEADER, "")
        request_id = incoming if _VALID_ID.match(incoming) else uuid.uuid4().hex
        request.state.request_id = request_id
        start = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
        finally:
            elapsed = time.perf_counter() - start
            route = request.scope.get("route")
            METRICS.observe(
                request.method, getattr(route, "path", UNMATCHED), status, elapsed
            )
            logger.info(
                "request",
                extra={
                    "fields": {
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                        "status": status,
                        "duration_ms": round(elapsed * 1000, 2),
                    }
                },
            )
        response.headers[REQUEST_ID_HEADER] = request_id
        return response
