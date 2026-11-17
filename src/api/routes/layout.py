"""Layout and random-order endpoints."""

from fastapi import APIRouter

from src.api import services
from src.api.schemas import (
    InstanceRequest,
    InstanceResponse,
    LayoutRequest,
    LayoutResponse,
)
from src.hilbert import curve_segments

router = APIRouter(prefix="/api", tags=["layout"])


@router.post("/layout", response_model=LayoutResponse)
def layout(req: LayoutRequest) -> LayoutResponse:
    """Generate or parse a warehouse; return its pick cells and Hilbert curve."""
    wh = services.load_warehouse(req.ascii, req.aisles, req.height)
    return LayoutResponse(
        ascii=wh.to_ascii(),
        width=wh.width,
        height=wh.height,
        depot=list(wh.depot),
        pick_cells=[list(c) for c in wh.pick_cells()],
        hilbert_side=wh.hilbert_side(),
        hilbert_curve=[
            [list(p) for p in seg] for seg in curve_segments(wh.width, wh.height)
        ],
    )


@router.post("/instance", response_model=InstanceResponse)
def instance(req: InstanceRequest) -> InstanceResponse:
    """Return a reproducible random order of ``k`` items."""
    wh = services.load_warehouse(req.ascii, req.aisles, req.height)
    items = services.random_items(wh, req.k, req.seed)
    return InstanceResponse(items=[list(c) for c in items])
