"""Application logic behind the API routes.

Nothing here imports FastAPI: failures are raised as :class:`ServiceError` and
turned into HTTP responses in :mod:`src.api.main`. That keeps the logic easy to
test and reuse.

Warehouses are deliberately **not** cached between requests: a ``Warehouse``
memoises one BFS map per source cell, so a long-lived instance would grow
without bound on a public service.
"""

from __future__ import annotations

import csv
import io
import time
from functools import lru_cache

import numpy as np

from src.api.config import MAX_CELLS, MAX_ITEMS, MAX_ORDERS, MAX_UPLOAD_BYTES
from src.environment import solvers as S
from src.environment.ortools_solver import OrToolsUnavailable, ortools_order
from src.environment.warehouse import (
    Cell,
    Instance,
    LayoutError,
    Warehouse,
    generate_layout,
)
from src.rl.agent import Agent

METHODS = {
    "snake": "S-shape sweep",
    "nearest_neighbor": "Nearest neighbour",
    "hilbert_sort": "Hilbert-curve sort",
    "two_opt": "2-opt (reference)",
    "ortools": "Google OR-Tools (guided local search)",
    "rl_greedy": "RL agent (greedy)",
    "rl_best_of_16": "RL agent (best of 16)",
    "optimal": "Optimal (exact, <=12 items)",
}
DEFAULT_METHODS = [
    "snake",
    "nearest_neighbor",
    "hilbert_sort",
    "rl_greedy",
    "rl_best_of_16",
    "two_opt",
    "ortools",
]


class ServiceError(Exception):
    """A request problem that maps to an HTTP error."""

    def __init__(self, status_code: int, detail: str) -> None:
        """Create the error with the HTTP status to report and a message."""
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


@lru_cache(maxsize=1)
def get_agent() -> Agent | None:
    """Load the trained agent once; return ``None`` if no weights are available.

    Call ``get_agent.cache_clear()`` after changing the weights file.
    """
    try:
        return Agent.load()
    except (OSError, ValueError):
        return None


def csv_safe(value: str) -> str:
    """Neutralise spreadsheet formula injection in user-supplied text."""
    return "'" + value if value[:1] in ("=", "+", "-", "@", "\t", "\r") else value


def load_warehouse(ascii_layout: str | None, aisles: int, height: int) -> Warehouse:
    """Build the requested warehouse (custom ASCII map or generated).

    Raises:
        ServiceError: 422 if the layout is invalid or too large.
    """
    try:
        text = (
            ascii_layout
            if ascii_layout and ascii_layout.strip()
            else generate_layout(aisles, height)
        )
        wh = Warehouse.from_ascii(text.strip("\n"))
        if wh.width * wh.height > MAX_CELLS:
            raise LayoutError(
                f"layout is too large for the API (max {MAX_CELLS} cells)"
            )
    except LayoutError as e:
        raise ServiceError(422, str(e)) from e
    return wh


def check_items(wh: Warehouse, items: list[list[int]]) -> list[Cell]:
    """Validate pick cells and drop duplicates (keeping order).

    Raises:
        ServiceError: 422 if there are too many items, an item is malformed, is
            not a walkable cell, or is the depot.
    """
    if len(items) > MAX_ITEMS:
        raise ServiceError(422, f"at most {MAX_ITEMS} items per trip")
    cells: list[Cell] = []
    for it in items:
        if len(it) != 2:
            raise ServiceError(422, "each item must be [x, y]")
        c = (int(it[0]), int(it[1]))
        if not wh.walkable(c):
            raise ServiceError(422, f"item {list(c)} is not a walkable cell")
        if c == wh.depot:
            raise ServiceError(422, "an item cannot be at the depot")
        cells.append(c)
    return list(dict.fromkeys(cells))


def build_instance(wh: Warehouse, items: list[Cell]) -> Instance:
    """Build a routing instance, mapping layout problems to 422 errors."""
    try:
        return wh.instance(items)
    except LayoutError as e:
        raise ServiceError(422, str(e)) from e


def run_method(name: str, inst: Instance) -> list[int]:
    """Route ``inst`` with the named method and return the visiting order.

    Raises:
        ServiceError: 422 for an unknown method or an order too big for
            ``optimal``; 503 if an RL method is requested without weights.
    """
    if name not in METHODS:
        raise ServiceError(
            422, f"unknown method {name!r}; choose from {sorted(METHODS)}"
        )
    if name.startswith("rl_"):
        agent = get_agent()
        if agent is None:
            raise ServiceError(
                503, "no trained weights found; run `python -m src.rl.cli train`"
            )
        if name == "rl_greedy":
            return agent.solve(inst, "greedy")
        return agent.solve(inst, "best_of_n", 16, np.random.default_rng(0))
    if name == "optimal":
        if inst.k > S.HELD_KARP_MAX_K:
            raise ServiceError(
                422, f"optimal solver supports at most {S.HELD_KARP_MAX_K} items"
            )
        return S.held_karp(inst)
    if name == "ortools":
        try:
            return ortools_order(inst)
        except OrToolsUnavailable as e:
            raise ServiceError(503, str(e)) from e
    classical = {
        "snake": S.snake_order,
        "nearest_neighbor": S.nearest_neighbor,
        "hilbert_sort": S.hilbert_order,
        "two_opt": S.best_heuristic,
    }
    return classical[name](inst)


def random_items(wh: Warehouse, k: int, seed: int) -> list[Cell]:
    """Pick ``k`` distinct random pick locations, reproducibly.

    Raises:
        ServiceError: 422 if the layout has fewer than ``k`` pick locations.
    """
    picks = wh.pick_cells()
    if k > len(picks):
        raise ServiceError(422, f"layout only has {len(picks)} pick locations")
    idx = np.random.default_rng(seed).choice(len(picks), size=k, replace=False)
    return [picks[i] for i in idx]


def solve_order(
    wh: Warehouse, items: list[Cell], methods: list[str] | None
) -> dict[str, object]:
    """Route one order with several methods and compare them.

    Returns:
        A dict matching :class:`~src.api.schemas.SolveResponse`.

    Raises:
        ServiceError: 422 if there are no items or the order is invalid.
    """
    if not items:
        raise ServiceError(422, "add at least one item to pick")
    inst = build_instance(wh, items)
    names = list(methods or DEFAULT_METHODS)
    if methods is None and inst.k <= S.HELD_KARP_MAX_K:
        names.append("optimal")
    results = []
    for name in names:
        t0 = time.perf_counter()
        order = run_method(name, inst)
        ms = (time.perf_counter() - t0) * 1000
        visit = [wh.depot] + [inst.cells[n] for n in order] + [wh.depot]
        results.append(
            {
                "method": name,
                "label": METHODS[name],
                "order": order,
                "length": inst.length(order),
                "ms": round(ms, 2),
                "visit": [list(c) for c in visit],
                "route": [list(c) for c in wh.route_path(visit)],
            }
        )
    best = min(r["length"] for r in results)
    for r in results:
        r["pct_over_best"] = (
            round(100.0 * (r["length"] / best - 1.0), 2) if best else 0.0
        )
    return {
        "depot": list(wh.depot),
        "items": [list(c) for c in items],
        "results": results,
    }


def optimize_orders(
    wh: Warehouse, orders: list[tuple[str, list[Cell]]], method: str
) -> dict[str, object]:
    """Route many orders with one method and compare with the S-shape baseline.

    Raises:
        ServiceError: 422 if there are too many orders or an order is invalid.
    """
    if len(orders) > MAX_ORDERS:
        raise ServiceError(422, f"at most {MAX_ORDERS} orders per request")
    out = []
    for oid, cells in orders:
        if not cells:
            continue
        try:
            inst = wh.instance(cells)
        except LayoutError as e:
            raise ServiceError(422, f"order {oid}: {e}") from e
        order = run_method(method, inst)
        baseline = inst.length(S.snake_order(inst))
        length = inst.length(order)
        out.append(
            {
                "order_id": oid,
                "n_items": inst.k,
                "length": length,
                "baseline_length": baseline,
                "saving_pct": round(100.0 * (1 - length / baseline), 2)
                if baseline
                else 0.0,
                "sequence": [list(inst.cells[n]) for n in order],
            }
        )
    total = sum(o["length"] for o in out)
    base = sum(o["baseline_length"] for o in out)
    return {
        "method": method,
        "orders": out,
        "total_length": total,
        "total_baseline_length": base,
        "total_saving_pct": round(100.0 * (1 - total / base), 2) if base else 0.0,
    }


def parse_orders_csv(raw: bytes, wh: Warehouse) -> list[tuple[str, list[Cell]]]:
    """Parse an orders CSV (columns ``order_id,x,y``) into validated orders.

    Raises:
        ServiceError: 413 if the file is too large; 422 for bad encoding,
            missing columns, non-integer coordinates or invalid items.
    """
    if len(raw) > MAX_UPLOAD_BYTES:
        raise ServiceError(413, "CSV is larger than 1 MB")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise ServiceError(422, "CSV must be UTF-8 text") from e
    reader = csv.DictReader(io.StringIO(text))
    fields = {f.strip() for f in (reader.fieldnames or [])}
    if not {"order_id", "x", "y"} <= fields:
        raise ServiceError(422, "CSV needs the columns: order_id,x,y")
    grouped: dict[str, list[list[int]]] = {}
    for line, row in enumerate(reader, start=2):
        clean = {k.strip(): (v or "").strip() for k, v in row.items() if k}
        try:
            xy = [int(clean["x"]), int(clean["y"])]
        except (ValueError, KeyError) as e:
            raise ServiceError(
                422, f"line {line}: x and y must be whole numbers"
            ) from e
        grouped.setdefault(clean["order_id"], []).append(xy)
    return [(oid, check_items(wh, items)) for oid, items in grouped.items()]


def render_routes_csv(result: dict[str, object]) -> str:
    """Render an :func:`optimize_orders` result as CSV, one row per stop."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["order_id", "step", "x", "y", "order_length", "baseline_length"])
    for o in result["orders"]:  # type: ignore[attr-defined]
        for step, (x, y) in enumerate(o["sequence"], start=1):
            w.writerow(
                [
                    csv_safe(str(o["order_id"])),
                    step,
                    x,
                    y,
                    o["length"],
                    o["baseline_length"],
                ]
            )
    return buf.getvalue()


def sample_orders_csv(n_orders: int = 8, seed: int = 2026) -> str:
    """Return a deterministic sample orders CSV for the default warehouse.

    The coordinates fit the generated 8-aisle, 22-row layout, so the file works
    with the default settings of the UI and the CSV endpoint.
    """
    wh = Warehouse.generate(8, 22)
    picks = wh.pick_cells()
    rng = np.random.default_rng(seed)
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["order_id", "x", "y"])
    for i in range(1, n_orders + 1):
        k = int(rng.integers(5, 16))
        for j in rng.choice(len(picks), size=k, replace=False):
            w.writerow([f"ORD-{i:03d}", *picks[j]])
    return buf.getvalue()
