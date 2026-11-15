"""Pydantic request and response models for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.api.config import MAX_ITEMS


class LayoutRequest(BaseModel):
    """Describe a warehouse: either an ASCII map or generator settings."""

    ascii: str | None = Field(
        None, description="Custom layout: '#' shelf, '.' floor, 'D' depot"
    )
    aisles: int = Field(8, ge=2, le=30, description="Aisles of a generated layout")
    height: int = Field(22, ge=7, le=100, description="Rows of a generated layout")


class InstanceRequest(LayoutRequest):
    """Ask for a random order on a warehouse."""

    k: int = Field(12, ge=1, le=MAX_ITEMS, description="Number of items")
    seed: int = Field(0, description="Seed, so the order is reproducible")


class SolveRequest(LayoutRequest):
    """Route one order with one or more methods."""

    items: list[list[int]] = Field(description="Pick cells as [x, y] pairs")
    methods: list[str] | None = Field(
        None, description="Methods to run; default is all (plus 'optimal' if small)"
    )


class OrderIn(BaseModel):
    """One order in a bulk request."""

    id: str
    items: list[list[int]]


class OptimizeRequest(LayoutRequest):
    """Route many orders with one method."""

    orders: list[OrderIn]
    method: str = "rl_best_of_16"


class HealthResponse(BaseModel):
    """Service status."""

    status: str
    agent_loaded: bool
    methods: list[str]


class LayoutResponse(BaseModel):
    """A parsed warehouse and its Hilbert curve."""

    ascii: str
    width: int
    height: int
    depot: list[int]
    pick_cells: list[list[int]]
    hilbert_side: int
    hilbert_curve: list[list[list[int]]]


class InstanceResponse(BaseModel):
    """A random order."""

    items: list[list[int]]


class MethodResult(BaseModel):
    """One method's route for an order."""

    method: str
    label: str
    order: list[int]
    length: float
    ms: float
    visit: list[list[int]]
    route: list[list[int]]
    pct_over_best: float


class SolveResponse(BaseModel):
    """Routes for one order from several methods."""

    depot: list[int]
    items: list[list[int]]
    results: list[MethodResult]


class OrderResult(BaseModel):
    """The optimised route for one order of a bulk request."""

    order_id: str
    n_items: int
    length: float
    baseline_length: float
    saving_pct: float
    sequence: list[list[int]]


class OptimizeResponse(BaseModel):
    """Optimised routes for many orders."""

    method: str
    orders: list[OrderResult]
    total_length: float
    total_baseline_length: float
    total_saving_pct: float
