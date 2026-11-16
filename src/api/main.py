"""FastAPI application.

Run:  uvicorn src.api.main:app --reload
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.api.middleware import RequestContextMiddleware, configure_logging
from src.api.routes import health, layout, optimize, solve
from src.api.routes import metrics as metrics_routes
from src.api.security import (
    ApiKeyMiddleware,
    RateLimitMiddleware,
    Settings,
    SlidingWindowLimiter,
)
from src.api.services import ServiceError

STATIC_DIR = Path(__file__).parent / "static"
DESCRIPTION = (
    "Warehouse pick routing with classical heuristics and a small "
    "reinforcement-learning agent."
)


async def handle_service_error(_: Request, exc: Exception) -> JSONResponse:
    """Turn a :class:`ServiceError` into a JSON error response."""
    if not isinstance(exc, ServiceError):  # only registered for ServiceError
        raise exc
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application with its routes and error handling.

    Args:
        settings: Security settings; read from the environment when omitted.
    """
    settings = settings or Settings.from_env()
    app = FastAPI(
        title="Hilbert-RL Spatial Optimizer", version="0.1.0", description=DESCRIPTION
    )
    configure_logging()
    # Added innermost first: the request context wraps everything so rejected
    # requests are still logged and counted, and the rate limit runs before the
    # key check so wrong keys cannot be guessed at full speed.
    if settings.api_key:
        app.add_middleware(ApiKeyMiddleware, key=settings.api_key)
    if settings.rate_limit_per_min > 0:
        app.add_middleware(
            RateLimitMiddleware,
            limiter=SlidingWindowLimiter(settings.rate_limit_per_min),
        )
    app.add_middleware(RequestContextMiddleware)
    app.add_exception_handler(ServiceError, handle_service_error)
    app.include_router(health.router)
    app.include_router(metrics_routes.router)
    app.include_router(layout.router)
    app.include_router(solve.router)
    app.include_router(optimize.router)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        """Serve the single-page web UI."""
        return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
