"""Bulk order optimisation (JSON and CSV)."""

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import PlainTextResponse

from src.api import services
from src.api.config import MAX_UPLOAD_BYTES
from src.api.schemas import OptimizeRequest, OptimizeResponse

router = APIRouter(prefix="/api", tags=["routing"])


@router.post("/optimize", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest) -> dict[str, object]:
    """Optimise many orders with one method and compare to the S-shape baseline."""
    wh = services.load_warehouse(req.ascii, req.aisles, req.height)
    orders = [(o.id, services.check_items(wh, o.items)) for o in req.orders]
    return services.optimize_orders(wh, orders, req.method)


@router.post("/optimize/csv", response_model=None)
async def optimize_csv(
    file: UploadFile = File(..., description="orders CSV with columns order_id,x,y"),
    layout_ascii: str = Form("", description="custom layout; empty = generated"),
    aisles: int = Form(8),
    height: int = Form(22),
    method: str = Form("rl_best_of_16"),
    as_json: bool = Form(False),
) -> PlainTextResponse | dict[str, object]:
    """Optimise orders uploaded as CSV; reply with CSV (or JSON if ``as_json``)."""
    raw = await file.read(MAX_UPLOAD_BYTES + 1)

    def work() -> dict[str, object]:
        """Parse and optimise off the event loop (the work is CPU-bound)."""
        wh = services.load_warehouse(layout_ascii, aisles, height)
        orders = services.parse_orders_csv(raw, wh)
        return services.optimize_orders(wh, orders, method)

    result = await run_in_threadpool(work)
    if as_json:
        return result
    return PlainTextResponse(services.render_routes_csv(result), media_type="text/csv")
