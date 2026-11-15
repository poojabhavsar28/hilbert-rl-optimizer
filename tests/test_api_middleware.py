"""Tests for request ids and structured logging."""

import json
import logging

from fastapi.testclient import TestClient

from src.api.main import app, create_app
from src.api.middleware import (
    LOGGER_NAME,
    REQUEST_ID_HEADER,
    JsonFormatter,
    configure_logging,
)

client = TestClient(app)


def test_response_carries_a_generated_request_id() -> None:
    """Every response has a fresh correlation id."""
    a = client.get("/api/health").headers[REQUEST_ID_HEADER]
    b = client.get("/api/health").headers[REQUEST_ID_HEADER]
    assert a and b and a != b


def test_valid_client_request_id_is_echoed() -> None:
    """A well-formed id supplied by the caller is propagated unchanged."""
    r = client.get("/api/health", headers={REQUEST_ID_HEADER: "trace-123.abc"})
    assert r.headers[REQUEST_ID_HEADER] == "trace-123.abc"


def test_unsafe_client_request_id_is_replaced() -> None:
    """Ids with spaces or quotes (log-forging attempts) are replaced."""
    r = client.get("/api/health", headers={REQUEST_ID_HEADER: 'x" , "status": 200'})
    assert r.headers[REQUEST_ID_HEADER] != 'x" , "status": 200'
    assert " " not in r.headers[REQUEST_ID_HEADER]


def test_each_request_is_logged_with_its_id_and_status(caplog) -> None:
    """One structured record per request, including failures."""
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        client.get("/api/health", headers={REQUEST_ID_HEADER: "rid-1"})
        client.post("/api/layout", json={"ascii": "xx\nxD"})
    fields = [r.fields for r in caplog.records if hasattr(r, "fields")]
    assert fields[0]["request_id"] == "rid-1" and fields[0]["status"] == 200
    assert fields[1]["path"] == "/api/layout" and fields[1]["status"] == 422
    assert fields[0]["duration_ms"] >= 0


def test_unhandled_errors_are_logged_as_500(caplog) -> None:
    """A crashing route is logged with status 500 and still returns an error."""
    broken = create_app()

    @broken.get("/boom")
    def boom() -> None:
        """Raise to simulate an unexpected failure."""
        raise RuntimeError("boom")

    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        r = TestClient(broken, raise_server_exceptions=False).get("/boom")
    assert r.status_code == 500
    assert any(
        getattr(rec, "fields", {}).get("status") == 500 for rec in caplog.records
    )


def test_json_formatter_emits_valid_json() -> None:
    """The formatter outputs a JSON object containing the extra fields."""
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "request", None, None)
    record.fields = {"request_id": "r1", "status": 200}
    out = json.loads(JsonFormatter().format(record))
    assert out["message"] == "request" and out["request_id"] == "r1"
    assert out["level"] == "INFO"


def test_configure_logging_is_idempotent() -> None:
    """Calling it twice does not duplicate handlers."""
    configure_logging()
    before = len(logging.getLogger(LOGGER_NAME).handlers)
    configure_logging()
    assert len(logging.getLogger(LOGGER_NAME).handlers) == before
