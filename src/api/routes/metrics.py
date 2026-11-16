"""Prometheus metrics endpoint."""

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse

from src.api.metrics import METRICS

router = APIRouter(tags=["metrics"])


@router.get("/metrics", response_class=PlainTextResponse, include_in_schema=False)
def metrics() -> PlainTextResponse:
    """Expose request counts and latency histograms for Prometheus."""
    return PlainTextResponse(
        METRICS.render(), media_type="text/plain; version=0.0.4; charset=utf-8"
    )
