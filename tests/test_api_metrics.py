"""Tests for the request metrics and the /metrics endpoint."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.metrics import BUCKETS, METRICS, UNMATCHED, Metrics

client = TestClient(app)


@pytest.fixture(autouse=True)
def fresh_metrics() -> None:
    METRICS.reset()


def test_observe_counts_requests_and_fills_cumulative_buckets() -> None:
    m = Metrics()
    m.observe("GET", "/a", 200, 0.004)  # below every bound
    m.observe("GET", "/a", 200, 0.3)
    m.observe("GET", "/a", 500, 99.0)  # above every bound: only +Inf
    text = m.render()
    assert 'http_requests_total{method="GET",path="/a",status="200"} 2' in text
    assert 'http_requests_total{method="GET",path="/a",status="500"} 1' in text
    assert f'le="{BUCKETS[0]}"}} 1' in text  # only the 4 ms request
    assert 'le="0.5"} 2' in text  # 4 ms and 300 ms
    assert 'le="+Inf"} 3' in text
    assert 'http_request_duration_seconds_count{path="/a"} 3' in text


def test_reset_clears_everything() -> None:
    m = Metrics()
    m.observe("GET", "/a", 200, 0.1)
    m.reset()
    assert "/a" not in m.render()


def test_metrics_endpoint_reports_traffic_by_route_template() -> None:
    client.get("/api/health")
    client.post("/api/instance", json={"aisles": 6, "height": 14, "k": 3, "seed": 1})
    res = client.get("/metrics")
    assert res.status_code == 200
    assert res.headers["content-type"].startswith("text/plain")
    assert 'path="/api/health",status="200"} 1' in res.text
    assert 'path="/api/instance",status="200"} 1' in res.text


def test_unknown_urls_share_one_label_so_memory_stays_bounded() -> None:
    for i in range(25):
        client.get(f"/random/{i}")
    text = client.get("/metrics").text
    assert f'path="{UNMATCHED}",status="404"}} 25' in text
    assert "/random/" not in text


def test_metrics_endpoint_is_hidden_from_the_openapi_schema() -> None:
    assert "/metrics" not in client.get("/openapi.json").json()["paths"]
