"""Optional API-key authentication and per-client rate limiting.

Both are off unless configured, so the public demo and local use need nothing:

* ``HILBERT_API_KEY``: when set, every ``/api/*`` route except ``/api/health``,
  and ``/metrics``, needs a matching ``X-API-Key`` header.
* ``HILBERT_RATE_LIMIT_PER_MIN``: when a positive integer, each client may make
  that many ``/api/*`` requests per minute (sliding window), otherwise 429.

The client is identified by its socket address. Behind a reverse proxy that is
the proxy's address, so rate-limit at the proxy instead or terminate here only
when clients connect directly. ``X-Forwarded-For`` is deliberately not trusted:
any caller could forge it.
"""

from __future__ import annotations

import hmac
import math
import os
import time
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

API_KEY_HEADER = "X-API-Key"
API_KEY_ENV = "HILBERT_API_KEY"
RATE_LIMIT_ENV = "HILBERT_RATE_LIMIT_PER_MIN"
OPEN_PATHS = {"/api/health"}  # liveness probes must work without a key


@dataclass(frozen=True)
class Settings:
    """Security settings for the API."""

    api_key: str | None = None
    rate_limit_per_min: int = 0

    @classmethod
    def from_env(cls) -> Settings:
        """Read settings from the environment.

        Raises:
            ValueError: If the rate limit is not a non-negative integer.
        """
        raw = os.environ.get(RATE_LIMIT_ENV, "").strip()
        try:
            limit = int(raw) if raw else 0
        except ValueError as e:
            raise ValueError(f"{RATE_LIMIT_ENV} must be an integer, got {raw!r}") from e
        if limit < 0:
            raise ValueError(f"{RATE_LIMIT_ENV} must not be negative, got {limit}")
        return cls(
            api_key=os.environ.get(API_KEY_ENV) or None, rate_limit_per_min=limit
        )


def is_api_path(path: str) -> bool:
    """Return True for routes that are rate limited (all of ``/api/`` but health)."""
    return path.startswith("/api/") and path not in OPEN_PATHS


def is_protected_path(path: str) -> bool:
    """Return True for routes that need the API key when one is configured."""
    return is_api_path(path) or path == "/metrics"


class SlidingWindowLimiter:
    """Allow at most ``limit`` events per ``window`` seconds for each client.

    Memory is bounded: at most ``max_clients`` clients are remembered, and the
    least recently seen is forgotten first.
    """

    def __init__(
        self,
        limit: int,
        window: float = 60.0,
        max_clients: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create a limiter."""
        self.limit = limit
        self.window = window
        self.max_clients = max_clients
        self._clock = clock
        self._hits: OrderedDict[str, deque[float]] = OrderedDict()

    def allow(self, client: str) -> tuple[bool, float]:
        """Record a request; return ``(allowed, seconds_until_allowed)``."""
        now = self._clock()
        hits = self._hits.setdefault(client, deque())
        self._hits.move_to_end(client)
        while hits and now - hits[0] >= self.window:
            hits.popleft()
        allowed = len(hits) < self.limit
        retry_after = 0.0
        if allowed:
            hits.append(now)
        else:
            retry_after = self.window - (now - hits[0])
        while len(self._hits) > self.max_clients:
            self._hits.popitem(last=False)
        return allowed, retry_after


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """Reject protected requests that lack the configured ``X-API-Key``."""

    def __init__(self, app, key: str) -> None:
        """Remember the expected key."""
        super().__init__(app)
        self._key = key.encode()

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Return 401 for a missing or wrong key on a protected path."""
        if is_protected_path(request.url.path):
            given = request.headers.get(API_KEY_HEADER, "").encode()
            if not hmac.compare_digest(given, self._key):
                return JSONResponse(
                    status_code=401,
                    content={"detail": f"missing or invalid {API_KEY_HEADER} header"},
                    headers={"WWW-Authenticate": "ApiKey"},
                )
        return await call_next(request)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Return 429 once a client exceeds its per-minute request budget."""

    def __init__(self, app, limiter: SlidingWindowLimiter) -> None:
        """Use ``limiter`` to decide each request."""
        super().__init__(app)
        self._limiter = limiter

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Count the request against its client and reject it when over budget."""
        if is_api_path(request.url.path):
            client = request.client.host if request.client else "unknown"
            allowed, retry_after = self._limiter.allow(client)
            if not allowed:
                seconds = max(1, math.ceil(retry_after))
                return JSONResponse(
                    status_code=429,
                    content={"detail": f"rate limit exceeded; retry in {seconds}s"},
                    headers={"Retry-After": str(seconds)},
                )
        return await call_next(request)
