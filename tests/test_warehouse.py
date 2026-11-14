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


def test_distance_is_symmetric_and_matches_path_length() -> None:
    """Walking distance is symmetric and equals the length of the returned path."""
    wh = Warehouse.generate(6, 14)
    picks = wh.pick_cells()
    a, b = picks[3], picks[-5]
    assert wh.distance(a, b) == wh.distance(b, a)
    path = wh.path(a, b)
    assert path[0] == a and path[-1] == b
    assert len(path) - 1 == wh.distance(a, b)
    assert all(wh.walkable(c) for c in path)


def test_path_steps_are_single_moves() -> None:
    """Every step of a path moves exactly one cell."""
    wh = Warehouse.generate(5, 12)
    path = wh.path(wh.depot, wh.pick_cells()[-1])
    for a, b in zip(path, path[1:], strict=False):
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def test_distance_goes_around_shelves() -> None:
    """Walking distance can exceed the Manhattan distance when shelves block."""
    wh = Warehouse.from_ascii("D#.\n.#.\n...")
    assert wh.distance((0, 0), (2, 0)) == 6  # down, across the bottom, up


def test_unreachable_cells_report_minus_one_and_have_no_path() -> None:
    """A walled-off cell has distance -1 and ``path`` raises."""
    wh = Warehouse.from_ascii("D#.\n.#.\n.#.")
    assert wh.distance((0, 0), (2, 0)) == -1
    with pytest.raises(LayoutError):
        wh.path((0, 0), (2, 0))


def test_route_path_visits_waypoints_in_order() -> None:
    """The full route passes through each waypoint, in order."""
    wh = Warehouse.generate(5, 12)
    picks = wh.pick_cells()
    way = [wh.depot, picks[2], picks[10], wh.depot]
    route = wh.route_path(way)
    assert route[0] == route[-1] == wh.depot
    positions = [route.index(w) for w in way[:-1]]
    assert positions == sorted(positions)
    expected = sum(wh.distance(a, b) for a, b in zip(way, way[1:], strict=False))
    assert len(route) - 1 == expected
