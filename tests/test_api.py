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
