"""HTTP tests for the API endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)
BASE = {"aisles": 6, "height": 14}


def test_health_reports_agent_and_methods() -> None:
    """The health endpoint confirms the service is up and weights are loaded."""
    body = client.get("/api/health").json()
    assert body["status"] == "ok" and body["agent_loaded"] is True
    assert "rl_best_of_16" in body["methods"]
    assert "ortools" in body["methods"]


def test_layout_response() -> None:
    """A generated layout reports its size, depot, pick cells and curve."""
    body = client.post("/api/layout", json=BASE).json()
    assert (body["width"], body["height"], body["depot"]) == (19, 14, [0, 13])
    assert body["hilbert_side"] == 32 and body["pick_cells"]
    cells = [tuple(p) for seg in body["hilbert_curve"] for p in seg]
    assert len(cells) == len(set(cells)) == 19 * 14


def test_layout_accepts_custom_ascii_and_rejects_bad_maps() -> None:
    """Custom maps are parsed; invalid ones return a 422 with the reason."""
    ok = client.post("/api/layout", json={"ascii": "D.#\n..."})
    assert ok.status_code == 200 and ok.json()["width"] == 3
    bad = client.post("/api/layout", json={"ascii": "xx\nxD"})
    assert bad.status_code == 422 and "unknown layout character" in bad.json()["detail"]


def test_instance_is_reproducible() -> None:
    """The same seed returns the same items."""
    body = {**BASE, "k": 8, "seed": 5}
    a = client.post("/api/instance", json=body).json()
    assert len(a["items"]) == 8
    assert a == client.post("/api/instance", json=body).json()


@pytest.mark.parametrize(
    "body", [{**BASE, "k": 0}, {**BASE, "k": 999}, {"aisles": 1}, {"height": 3}]
)
def test_request_validation_returns_422(body: dict) -> None:
    """Out-of-range parameters are rejected by the schema."""
    assert client.post("/api/instance", json=body).status_code == 422


def test_too_many_items_for_the_layout() -> None:
    """Asking for more items than the layout has is a clear 422."""
    r = client.post("/api/instance", json={"aisles": 2, "height": 7, "k": 60})
    assert r.status_code == 422 and "pick locations" in r.json()["detail"]


def solve_body(k: int = 7, seed: int = 1, **extra: object) -> dict:
    """Build a ``/api/solve`` request for a random order."""
    items = client.post("/api/instance", json={**BASE, "k": k, "seed": seed}).json()[
        "items"
    ]
    return {**BASE, "items": items, **extra}


def test_solve_routes_are_consistent() -> None:
    """Walked cells equal the reported length; nothing beats the optimum."""
    res = client.post("/api/solve", json=solve_body()).json()["results"]
    by = {r["method"]: r for r in res}
    assert "optimal" in by
    for r in res:
        assert r["length"] >= by["optimal"]["length"] - 1e-9
        assert len(r["route"]) - 1 == r["length"]
        assert r["route"][0] == r["route"][-1] == [0, 13]
        assert sorted(r["order"]) == list(range(1, 8))


def test_solve_respects_the_methods_filter() -> None:
    """Only the requested methods are run."""
    res = client.post(
        "/api/solve", json=solve_body(methods=["snake", "two_opt"])
    ).json()
    assert [r["method"] for r in res["results"]] == ["snake", "two_opt"]


@pytest.mark.parametrize(
    "extra",
    [
        {"items": []},
        {"items": [[1, 1]]},  # a shelf
        {"items": [[0, 13]]},  # the depot
        {"items": [[99, 99]]},
        {"items": [[0, 5]], "methods": ["nope"]},
        {"items": [[0, i % 14] for i in range(61)] + [[3, 3]]},
    ],
)
def test_solve_validation(extra: dict) -> None:
    """Invalid orders are rejected with 422."""
    assert client.post("/api/solve", json={**BASE, **extra}).status_code == 422


def test_optimal_is_limited_to_twelve_items() -> None:
    """Large orders skip ``optimal`` by default and reject it when forced."""
    big = solve_body(k=14)
    assert (
        client.post("/api/solve", json={**big, "methods": ["optimal"]}).status_code
        == 422
    )
    ok = client.post("/api/solve", json=big).json()["results"]
    assert "optimal" not in {r["method"] for r in ok}


def test_duplicate_items_are_collapsed() -> None:
    """Repeated pick locations are visited once."""
    body = {**BASE, "items": [[0, 5], [0, 5], [3, 7]], "methods": ["nearest_neighbor"]}
    assert len(client.post("/api/solve", json=body).json()["items"]) == 2


def test_optimize_json() -> None:
    """Bulk JSON optimisation returns per-order results and totals."""
    orders = [
        {"id": "A", "items": [[0, 5], [3, 7]]},
        {"id": "B", "items": [[0, 9], [6, 4], [9, 7]]},
    ]
    r = client.post(
        "/api/optimize", json={**BASE, "orders": orders, "method": "two_opt"}
    )
    body = r.json()
    assert r.status_code == 200 and [o["order_id"] for o in body["orders"]] == [
        "A",
        "B",
    ]
    assert body["total_length"] <= body["total_baseline_length"]


def upload(text: str, **form: object):
    """POST ``text`` as an orders CSV to ``/api/optimize/csv``."""
    files = {"file": ("orders.csv", text.encode(), "text/csv")}
    return client.post("/api/optimize/csv", files=files, data={**BASE, **form})


def test_optimize_csv_returns_csv_and_neutralises_formulas() -> None:
    """CSV in, CSV out; dangerous order ids are prefixed with a quote."""
    r = upload("order_id,x,y\nA,0,5\nA,3,7\n=EVIL,0,9\n", method="two_opt")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    lines = r.text.strip().splitlines()
    assert lines[0].startswith("order_id,step") and any(
        ln.startswith("'=EVIL") for ln in lines
    )


def test_optimize_csv_as_json_and_custom_layout() -> None:
    """The same endpoint can return JSON and accept a custom layout."""
    r = upload("order_id,x,y\nA,0,5\nA,3,7\n", method="two_opt", as_json="true")
    assert len(r.json()["orders"]) == 1
    custom = upload(
        "order_id,x,y\nA,0,0\n",
        layout_ascii="..#\n.D.\n...",
        method="snake",
        as_json="true",
    )
    assert custom.status_code == 200


@pytest.mark.parametrize(
    "text", ["a,b\n1,2\n", "order_id,x,y\nA,x,1\n", "order_id,x,y\nA,1,1\n"]
)
def test_optimize_csv_errors(text: str) -> None:
    """Malformed CSVs give 422 with a reason."""
    assert upload(text).status_code == 422


def test_optimize_csv_too_large() -> None:
    """Uploads over 1 MB are refused with 413."""
    assert upload("order_id,x,y\n" + "A,0,5\n" * 200_000).status_code == 413


def test_openapi_schema_lists_every_endpoint() -> None:
    """The interactive docs depend on a valid OpenAPI document."""
    schema = client.get("/openapi.json").json()
    paths = set(schema["paths"])
    expected = {"/api/health", "/api/layout", "/api/instance", "/api/solve"}
    assert expected | {"/api/optimize", "/api/optimize/csv"} <= paths
    assert client.get("/docs").status_code == 200


def test_sample_orders_csv_downloads_and_round_trips() -> None:
    """The sample file is valid for the default warehouse and optimises cleanly."""
    r = client.get("/api/sample-orders.csv")
    assert r.status_code == 200 and "attachment" in r.headers["content-disposition"]
    assert r.text.startswith("order_id,x,y\n")
    assert client.get("/api/sample-orders.csv").text == r.text  # deterministic
    files = {"file": ("o.csv", r.content, "text/csv")}
    out = client.post("/api/optimize/csv", files=files, data={"as_json": "true"})
    body = out.json()
    assert out.status_code == 200 and len(body["orders"]) == 8
    assert body["total_length"] < body["total_baseline_length"]


def test_web_ui_is_served_at_the_root() -> None:
    """The root URL serves the single-page UI, which calls the documented API."""
    r = client.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    for needle in ("Hilbert-RL Spatial Optimizer", "/api/solve", "/api/optimize/csv"):
        assert needle in r.text
    assert client.get("/static/index.html").status_code == 200
    assert client.get("/static/missing.js").status_code == 404


def test_service_error_handler_rejects_other_exceptions() -> None:
    """The handler is only for ServiceError; anything else is re-raised."""
    import asyncio

    from src.api.main import handle_service_error

    with pytest.raises(ValueError, match="boom"):
        asyncio.run(handle_service_error(None, ValueError("boom")))  # type: ignore[arg-type]
