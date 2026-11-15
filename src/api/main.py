"""FastAPI application.

Run:  uvicorn src.api.main:app --reload
Docs: http://localhost:8000/docs
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.api.routes import health, layout
from src.api.services import ServiceError

DESCRIPTION = (
    "Warehouse pick routing with classical heuristics and a small "
    "reinforcement-learning agent."
)


async def handle_service_error(_: Request, exc: ServiceError) -> JSONResponse:
    """Turn a :class:`ServiceError` into a JSON error response."""
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


def create_app() -> FastAPI:
    """Build the FastAPI application with its routes and error handling."""
    app = FastAPI(
        title="Hilbert-RL Spatial Optimizer", version="0.1.0", description=DESCRIPTION
    )
    app.add_exception_handler(ServiceError, handle_service_error)
    app.include_router(health.router)
    app.include_router(layout.router)
    return app


app = create_app()
