"""Tests for optional API-key auth and rate limiting."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.metrics import METRICS
from src.api.security import (
    API_KEY_ENV,
    RATE_LIMIT_ENV,
    Settings,
    SlidingWindowLimiter,
    is_api_path,
    is_protected_path,
)

BODY = {"aisles": 6, "height": 14, "k": 3, "seed": 1}


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


# -- settings ---------------------------------------------------------------
def test_settings_default_to_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    monkeypatch.delenv(RATE_LIMIT_ENV, raising=False)
    assert Settings.from_env() == Settings(api_key=None, rate_limit_per_min=0)


def test_settings_read_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "s3cret")
    monkeypatch.setenv(RATE_LIMIT_ENV, " 30 ")
    assert Settings.from_env() == Settings(api_key="s3cret", rate_limit_per_min=30)
    monkeypatch.setenv(API_KEY_ENV, "")
    assert Settings.from_env().api_key is None  # empty means no key


@pytest.mark.parametrize("bad", ["many", "-5", "1.5"])
def test_settings_reject_a_bad_rate_limit(
    monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    monkeypatch.setenv(RATE_LIMIT_ENV, bad)
    with pytest.raises(ValueError, match=RATE_LIMIT_ENV):
        Settings.from_env()


def test_path_classification() -> None:
    assert is_api_path("/api/solve") and not is_api_path("/api/health")
    assert not is_api_path("/") and not is_api_path("/static/x.js")
    assert is_protected_path("/metrics") and is_protected_path("/api/layout")
    assert not is_protected_path("/api/health") and not is_protected_path("/")


# -- limiter ----------------------------------------------------------------
def test_limiter_allows_the_budget_then_blocks_with_a_retry_time() -> None:
    clock = FakeClock()
    lim = SlidingWindowLimiter(3, window=60, clock=clock)
    assert [lim.allow("a")[0] for _ in range(3)] == [True, True, True]
    allowed, retry = lim.allow("a")
    assert not allowed and retry == pytest.approx(60.0)
    clock.now += 45
    assert lim.allow("a") == (False, pytest.approx(15.0))
    clock.now += 15  # the first hits have left the window
    assert lim.allow("a")[0]


def test_limiter_tracks_clients_independently() -> None:
    lim = SlidingWindowLimiter(1, clock=FakeClock())
    assert lim.allow("a")[0] and not lim.allow("a")[0]
    assert lim.allow("b")[0]


def test_limiter_memory_is_bounded() -> None:
    lim = SlidingWindowLimiter(5, max_clients=100, clock=FakeClock())
    for i in range(1000):
        lim.allow(f"ip-{i}")
    assert len(lim._hits) == 100


# -- API key ----------------------------------------------------------------
@pytest.fixture
def secured() -> TestClient:
    return TestClient(create_app(Settings(api_key="s3cret")))


def test_without_a_key_setting_everything_is_open() -> None:
    open_client = TestClient(create_app(Settings()))
    assert open_client.post("/api/instance", json=BODY).status_code == 200
    assert open_client.get("/metrics").status_code == 200


def test_missing_or_wrong_key_is_rejected(secured: TestClient) -> None:
    for headers in ({}, {"X-API-Key": "wrong"}, {"X-API-Key": ""}):
        res = secured.post("/api/instance", json=BODY, headers=headers)
        assert res.status_code == 401
        assert res.headers["www-authenticate"] == "ApiKey"
        assert "X-API-Key" in res.json()["detail"]
    assert secured.get("/metrics").status_code == 401


def test_correct_key_is_accepted(secured: TestClient) -> None:
    res = secured.post("/api/instance", json=BODY, headers={"X-API-Key": "s3cret"})
    assert res.status_code == 200
    assert secured.get("/metrics", headers={"X-API-Key": "s3cret"}).status_code == 200


def test_health_ui_and_docs_stay_open(secured: TestClient) -> None:
    assert secured.get("/api/health").status_code == 200
    assert secured.get("/").status_code == 200
    assert secured.get("/docs").status_code == 200


def test_rejected_requests_are_still_counted_in_metrics(secured: TestClient) -> None:
    METRICS.reset()
    secured.post("/api/instance", json=BODY)
    assert 'status="401"' in METRICS.render()


# -- rate limit -------------------------------------------------------------
def test_rate_limit_returns_429_with_retry_after_but_spares_health() -> None:
    client = TestClient(create_app(Settings(rate_limit_per_min=3)))
    codes = [client.post("/api/instance", json=BODY).status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]
    res = client.post("/api/instance", json=BODY)
    assert int(res.headers["retry-after"]) >= 1
    assert "rate limit" in res.json()["detail"]
    assert client.get("/api/health").status_code == 200
    assert client.get("/").status_code == 200


def test_rate_limit_runs_before_the_key_check() -> None:
    """Wrong keys cannot be guessed at full speed: they count against the budget."""
    client = TestClient(create_app(Settings(api_key="k", rate_limit_per_min=2)))
    codes = [client.post("/api/instance", json=BODY).status_code for _ in range(4)]
    assert codes == [401, 401, 429, 429]
