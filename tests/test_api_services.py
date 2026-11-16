"""Tests for the API service layer (no HTTP involved)."""

import pytest

from src.api import services as svc
from src.api.config import MAX_ITEMS, MAX_ORDERS
from src.api.services import ServiceError
from src.environment.ortools_solver import OrToolsUnavailable
from src.environment.warehouse import Warehouse


@pytest.fixture(scope="module")
def wh() -> Warehouse:
    """A small generated warehouse."""
    return svc.load_warehouse(None, 6, 14)


def code(exc_info: pytest.ExceptionInfo) -> int:
    """Return the HTTP status carried by a ``ServiceError``."""
    return exc_info.value.status_code


def test_load_warehouse_generates_or_parses_and_limits() -> None:
    """Generated and custom maps load; invalid or oversized ones give 422."""
    assert svc.load_warehouse(None, 6, 14).width == 19
    assert svc.load_warehouse("D.\n..", 6, 14).width == 2
    assert svc.load_warehouse("  ", 6, 14).width == 19  # blank means generated
    with pytest.raises(ServiceError) as e:
        svc.load_warehouse("xx\nxD", 6, 14)
    assert code(e) == 422
    with pytest.raises(ServiceError) as e:
        svc.load_warehouse("\n".join(["." * 200] * 61 + ["D" + "." * 199]), 6, 14)
    assert code(e) == 422 and "too large" in e.value.detail


def test_check_items_validation(wh: Warehouse) -> None:
    """Malformed, blocked, depot and excess items are rejected; duplicates merge."""
    assert svc.check_items(wh, [[0, 5], [0, 5], [3, 7]]) == [(0, 5), (3, 7)]
    bad = [
        [[1, 1]],
        [[99, 99]],
        [list(wh.depot)],
        [[1]],
        [[0, i % 14] for i in range(61)],
    ]
    for items in bad:
        with pytest.raises(ServiceError) as e:
            svc.check_items(wh, items)
        assert code(e) == 422
    assert MAX_ITEMS == 60


def test_run_method_dispatch(wh: Warehouse) -> None:
    """Every method returns a permutation; unknown or oversized ones fail clearly."""
    items = svc.random_items(wh, 8, seed=1)
    inst = svc.build_instance(wh, items)
    for name in svc.METHODS:
        assert sorted(svc.run_method(name, inst)) == list(range(1, 9))
    with pytest.raises(ServiceError) as e:
        svc.run_method("nope", inst)
    assert code(e) == 422
    big = svc.build_instance(wh, svc.random_items(wh, 14, seed=1))
    with pytest.raises(ServiceError) as e:
        svc.run_method("optimal", big)
    assert code(e) == 422


def test_rl_methods_need_weights(wh: Warehouse, monkeypatch, tmp_path) -> None:
    """Without a weights file RL methods report 503 instead of crashing."""
    monkeypatch.setenv("HILBERT_RL_WEIGHTS", str(tmp_path / "missing.json"))
    svc.get_agent.cache_clear()
    try:
        inst = svc.build_instance(wh, svc.random_items(wh, 5, seed=1))
        with pytest.raises(ServiceError) as e:
            svc.run_method("rl_greedy", inst)
        assert code(e) == 503
        assert svc.run_method("snake", inst)  # classical methods still work
    finally:
        monkeypatch.undo()
        svc.get_agent.cache_clear()


def test_ortools_reports_503_when_not_installed(
    wh: Warehouse, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing optional package is a clear 503, not a crash."""

    def missing(_inst):
        raise OrToolsUnavailable("OR-Tools is not installed; run `pip install ortools`")

    monkeypatch.setattr(svc, "ortools_order", missing)
    inst = svc.build_instance(wh, svc.random_items(wh, 5, seed=1))
    with pytest.raises(ServiceError) as e:
        svc.run_method("ortools", inst)
    assert code(e) == 503
    assert "pip install ortools" in e.value.detail


def test_random_items_reproducible_and_limited(wh: Warehouse) -> None:
    """Seeds give the same items; too many raises 422."""
    assert svc.random_items(wh, 6, 3) == svc.random_items(wh, 6, 3)
    with pytest.raises(ServiceError) as e:
        svc.random_items(wh, 10_000, 3)
    assert code(e) == 422


def test_solve_order_compares_methods(wh: Warehouse) -> None:
    """Results include routes whose cell count equals the reported length."""
    items = svc.random_items(wh, 7, seed=2)
    res = svc.solve_order(wh, items, None)
    by = {r["method"]: r for r in res["results"]}
    assert "optimal" in by
    for r in res["results"]:
        assert len(r["route"]) - 1 == r["length"]
        assert r["length"] >= by["optimal"]["length"] - 1e-9
    assert min(r["pct_over_best"] for r in res["results"]) == 0.0
    with pytest.raises(ServiceError):
        svc.solve_order(wh, [], None)


def test_optimize_orders_and_limits(wh: Warehouse) -> None:
    """Bulk optimisation reports savings versus the baseline and enforces limits."""
    orders = [(f"A{i}", svc.random_items(wh, 6, seed=i)) for i in range(4)]
    res = svc.optimize_orders(wh, orders, "two_opt")
    assert (
        len(res["orders"]) == 4 and res["total_length"] <= res["total_baseline_length"]
    )
    assert svc.optimize_orders(wh, [("empty", [])], "two_opt")["orders"] == []
    with pytest.raises(ServiceError):
        svc.optimize_orders(wh, [("x", [(0, 5)])] * (MAX_ORDERS + 1), "two_opt")


def test_parse_orders_csv(wh: Warehouse) -> None:
    """Valid CSVs group rows by order; every malformed input is a clear 422/413."""
    ok = svc.parse_orders_csv(b"order_id,x,y\nA,0,5\nA,3,7\nB,0,9\n", wh)
    assert [(oid, len(c)) for oid, c in ok] == [("A", 2), ("B", 1)]
    assert svc.parse_orders_csv(
        b"\xef\xbb\xbforder_id, x ,y\nA,0,5\n", wh
    )  # BOM, spaces
    for raw, status in [
        (b"a,b\n1,2\n", 422),
        (b"order_id,x,y\nA,x,1\n", 422),
        (b"order_id,x,y\nA,1,1\n", 422),
        (b"\xff\xfe", 422),
        (b"order_id,x,y\n" + b"A,0,5\n" * 200_000, 413),
    ]:
        with pytest.raises(ServiceError) as e:
            svc.parse_orders_csv(raw, wh)
        assert code(e) == status


def test_csv_rendering_neutralises_formulas(wh: Warehouse) -> None:
    """Order IDs that would execute as spreadsheet formulas are prefixed."""
    res = svc.optimize_orders(wh, [("=EVIL", [(0, 5), (3, 7)])], "two_opt")
    text = svc.render_routes_csv(res)
    assert text.splitlines()[0].startswith("order_id,step")
    assert text.splitlines()[1].startswith("'=EVIL")
    assert svc.csv_safe("fine") == "fine" and svc.csv_safe("@x") == "'@x"
