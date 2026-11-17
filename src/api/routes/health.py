"""Health endpoint."""

from fastapi import APIRouter

from src.api.schemas import HealthResponse
from src.api.services import METHODS, get_agent

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report service status, whether trained weights are loaded, and methods."""
    return HealthResponse(
        status="ok", agent_loaded=get_agent() is not None, methods=sorted(METHODS)
    )
