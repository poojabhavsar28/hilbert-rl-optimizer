"""Google OR-Tools routing solver, used as an industrial-strength baseline.

OR-Tools is an optional dependency (``pip install hilbert-rl-optimizer[solvers]``
or ``pip install ortools``). Importing this module never fails; calling
:func:`ortools_order` without the package raises :class:`OrToolsUnavailable`.
"""

from __future__ import annotations

from src.environment.warehouse import Instance

DEFAULT_TIME_LIMIT_S = 0.3
_SCALE = 1000  # OR-Tools needs integer arc costs


class OrToolsUnavailable(RuntimeError):
    """Raised when the optional ``ortools`` package is not installed."""


def ortools_available() -> bool:
    """Return True when the ``ortools`` package can be imported."""
    try:
        import ortools  # noqa: F401
    except ImportError:
        return False
    return True


def ortools_order(
    inst: Instance, time_limit_s: float = DEFAULT_TIME_LIMIT_S
) -> list[int]:
    """Return a pick order from OR-Tools (cheapest-arc start + guided local search).

    The route is a closed tour through the depot (node 0), so the returned list
    holds node ids ``1..k`` exactly like the other solvers.

    Args:
        inst: The routing instance.
        time_limit_s: Local-search budget in seconds.

    Raises:
        OrToolsUnavailable: If ``ortools`` is not installed.
        RuntimeError: If OR-Tools finds no solution within the time limit.
    """
    try:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    except ImportError as e:
        raise OrToolsUnavailable(
            "OR-Tools is not installed; run `pip install ortools`"
        ) from e

    if inst.k <= 1:
        return list(range(1, inst.k + 1))

    n = inst.k + 1
    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def cost(i: int, j: int) -> int:
        return int(
            round(inst.D[manager.IndexToNode(i), manager.IndexToNode(j)] * _SCALE)
        )

    transit = routing.RegisterTransitCallback(cost)
    routing.SetArcCostEvaluatorOfAllVehicles(transit)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    params.time_limit.nanos = int(time_limit_s * 1e9) % 1_000_000_000
    params.time_limit.seconds = int(time_limit_s)

    solution = routing.SolveWithParameters(params)
    if solution is None:
        raise RuntimeError("OR-Tools found no solution within the time limit")

    order: list[int] = []
    index = routing.Start(0)
    index = solution.Value(routing.NextVar(index))
    while not routing.IsEnd(index):
        order.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))
    return order
