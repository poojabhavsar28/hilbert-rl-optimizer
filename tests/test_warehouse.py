"""Tests for the warehouse layout model."""

import pytest

from src.environment.warehouse import LayoutError, Warehouse, generate_layout


def test_generated_layout_shape_and_depot() -> None:
    """A generated map has the documented size and a depot in the corner."""
    wh = Warehouse.generate(aisles=6, height=14)
    assert (wh.width, wh.height) == (19, 14)
    assert wh.depot == (0, 13)
    assert wh.walkable(wh.depot)


def test_ascii_round_trip() -> None:
    """Parsing then rendering a layout reproduces it exactly."""
    text = generate_layout(5, 11)
    assert Warehouse.from_ascii(text).to_ascii() == text


def test_pick_cells_touch_shelves_and_exclude_depot() -> None:
    """Pick locations are floor cells next to a shelf, never the depot."""
    wh = Warehouse.generate(4, 10)
    picks = wh.pick_cells()
    assert picks and wh.depot not in picks
    for x, y in picks:
        assert wh.walkable((x, y))
        neighbours = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        assert any(
            0 <= nx < wh.width and 0 <= ny < wh.height and wh.grid[ny, nx]
            for nx, ny in neighbours
        )


def test_walkable_rejects_shelves_and_outside() -> None:
    """Shelves and cells outside the map are not walkable."""
    wh = Warehouse.generate(4, 10)
    assert not wh.walkable((1, 1))
    assert not wh.walkable((-1, 0))
    assert not wh.walkable((wh.width, 0))


@pytest.mark.parametrize(
    "text",
    ["", "..\n.", "..\n..", "D.\n.D", "D?\n.."],
    ids=["empty", "ragged", "no-depot", "two-depots", "bad-char"],
)
def test_invalid_layouts_are_rejected(text: str) -> None:
    """Malformed maps raise ``LayoutError`` with a message."""
    with pytest.raises(LayoutError):
        Warehouse.from_ascii(text)


@pytest.mark.parametrize(("aisles", "height"), [(1, 10), (41, 10), (5, 6), (5, 201)])
def test_generate_layout_validates_dimensions(aisles: int, height: int) -> None:
    """Out-of-range dimensions raise ``LayoutError``."""
    with pytest.raises(LayoutError):
        generate_layout(aisles, height)


def test_oversized_layout_is_rejected() -> None:
    """Maps larger than ``MAX_SIDE`` are refused."""
    with pytest.raises(LayoutError):
        Warehouse.from_ascii("D" + "." * 200)
