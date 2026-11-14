"""Routing heuristics and an exact solver for pick trips.

Every solver takes an :class:`~src.environment.warehouse.Instance` and returns
the visiting order as a list of node ids ``1..k``. Node 0 is the depot, which is
implicit at both ends of the trip.
"""

from __future__ import annotations

from src.environment.warehouse import Instance


def snake_order(inst: Instance) -> list[int]:
    """Return the S-shape sweep: walk aisle by aisle, alternating direction.

    This is the textbook baseline for block-and-aisle warehouses.
    """
    cols: dict[int, list[int]] = {}
    for node in range(1, inst.k + 1):
        cols.setdefault(inst.cells[node][0], []).append(node)
    order: list[int] = []
    for rank, x in enumerate(sorted(cols)):
        nodes = cols[x]
        # the depot sits at the bottom, so the first aisle is walked upwards
        nodes.sort(key=lambda n: inst.cells[n][1], reverse=(rank % 2 == 0))
        order.extend(nodes)
    return order


def nearest_neighbor(inst: Instance, start: int = 0) -> list[int]:
    """Return the greedy route: always walk to the closest unpicked item."""
    remaining = set(range(1, inst.k + 1))
    cur = start
    order: list[int] = []
    while remaining:
        nxt = min(remaining, key=lambda j: (inst.D[cur, j], j))
        order.append(nxt)
        remaining.remove(nxt)
        cur = nxt
    return order


def hilbert_order(inst: Instance) -> list[int]:
    """Return items sorted along the Hilbert curve, in the shorter direction.

    This ignores shelves, so it is a weak heuristic on block-and-aisle layouts.
    """
    nodes = sorted(range(1, inst.k + 1), key=lambda n: (inst.hilbert[n], n))
    forward = inst.length(nodes)
    backward = inst.length(nodes[::-1])
    return nodes if forward <= backward else nodes[::-1]


def two_opt(
    inst: Instance, order: list[int] | None = None, max_passes: int = 100
) -> list[int]:
    """Improve a route with 2-opt local search on the round trip.

    Args:
        inst: The routing instance.
        order: Starting route; defaults to nearest neighbour.
        max_passes: Cap on full improvement sweeps.
    """
    if order is None:
        order = nearest_neighbor(inst)
    tour = [0] + list(order) + [0]
    dist = inst.D
    n = len(tour)
    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                a, b = tour[i - 1], tour[i]
                c, d = tour[j], tour[j + 1]
                delta = dist[a, c] + dist[b, d] - dist[a, b] - dist[c, d]
                if delta < -1e-9:
                    tour[i : j + 1] = reversed(tour[i : j + 1])
                    improved = True
    return tour[1:-1]


def best_heuristic(inst: Instance) -> list[int]:
    """Return the better of 2-opt started from nearest neighbour and from Hilbert."""
    if inst.k == 0:
        return []
    candidates = [
        two_opt(inst, nearest_neighbor(inst)),
        two_opt(inst, hilbert_order(inst)),
    ]
    return min(candidates, key=inst.length)
