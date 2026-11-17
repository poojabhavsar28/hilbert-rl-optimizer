"""Tests for the API request and response schemas."""

import pytest
from pydantic import ValidationError

from src.api.config import MAX_ITEMS
from src.api.schemas import InstanceRequest, LayoutRequest, SolveRequest


def test_layout_request_defaults() -> None:
    """Defaults describe the standard generated warehouse."""
    req = LayoutRequest()
    assert (req.aisles, req.height, req.ascii) == (8, 22, None)


@pytest.mark.parametrize(
    "field,value", [("aisles", 1), ("aisles", 31), ("height", 6), ("height", 101)]
)
def test_layout_request_bounds(field: str, value: int) -> None:
    """Out-of-range generator settings are rejected."""
    with pytest.raises(ValidationError):
        LayoutRequest(**{field: value})


def test_instance_request_bounds() -> None:
    """``k`` is limited to ``MAX_ITEMS``."""
    assert InstanceRequest(k=MAX_ITEMS).k == MAX_ITEMS
    for bad in (0, MAX_ITEMS + 1):
        with pytest.raises(ValidationError):
            InstanceRequest(k=bad)


def test_solve_request_requires_items() -> None:
    """``items`` is mandatory; ``methods`` is optional."""
    with pytest.raises(ValidationError):
        SolveRequest()
    assert SolveRequest(items=[[0, 1]]).methods is None
