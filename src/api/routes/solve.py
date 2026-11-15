"""Route one order with several methods."""

from fastapi import APIRouter

from src.api import services
from src.api.schemas import SolveRequest, SolveResponse

router = APIRouter(prefix="/api", tags=["routing"])


@router.post("/solve", response_model=SolveResponse)
def solve(req: SolveRequest) -> dict[str, object]:
    """Route one order and compare methods by route length.

    Returns each method's visiting order, round-trip length, timing and the full
    cell-by-cell route. When no ``methods`` are given all are run, plus the exact
    optimum if the order has at most 12 items.
    """
    wh = services.load_warehouse(req.ascii, req.aisles, req.height)
    items = services.check_items(wh, req.items)
    return services.solve_order(wh, items, req.methods)
