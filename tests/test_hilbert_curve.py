"""Tests for the 2D Hilbert curve encoder and decoder."""

import numpy as np
import pytest

from src.hilbert import (
    curve_segments,
    d2xy,
    hilbert_index,
    hilbert_side,
    next_pow2,
    xy2d,
    xy2d_batch,
)


@pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32])
def test_round_trip_and_bijection(n: int) -> None:
    """Every index decodes to a distinct cell that encodes back to the index."""
    seen = set()
    for d in range(n * n):
        x, y = d2xy(n, d)
        assert xy2d(n, x, y) == d
        seen.add((x, y))
    assert len(seen) == n * n


@pytest.mark.parametrize("n", [2, 4, 16, 32])
def test_consecutive_cells_are_adjacent(n: int) -> None:
    """The defining property: neighbours on the curve are neighbours on the grid."""
    pts = [d2xy(n, d) for d in range(n * n)]
    for a, b in zip(pts, pts[1:], strict=False):
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def test_curve_starts_at_origin() -> None:
    """Index 0 is the origin cell for every grid size."""
    for n in (1, 2, 8, 64):
        assert d2xy(n, 0) == (0, 0)
        assert xy2d(n, 0, 0) == 0


@pytest.mark.parametrize(
    ("n", "x", "y"),
    [(6, 0, 0), (0, 0, 0), (4, 4, 0), (4, 0, -1)],
)
def test_xy2d_rejects_bad_input(n: int, x: int, y: int) -> None:
    """Non power-of-two sides and out-of-grid cells raise ``ValueError``."""
    with pytest.raises(ValueError):
        xy2d(n, x, y)


@pytest.mark.parametrize(("n", "d"), [(4, 16), (4, -1), (3, 0)])
def test_d2xy_rejects_bad_input(n: int, d: int) -> None:
    """Out-of-range indices and invalid sides raise ``ValueError``."""
    with pytest.raises(ValueError):
        d2xy(n, d)


def test_pow2_helpers() -> None:
    """``next_pow2`` and ``hilbert_side`` round up to a power of two."""
    assert [next_pow2(n) for n in (1, 2, 3, 17, 32)] == [1, 2, 4, 32, 32]
    assert hilbert_side(25, 22) == 32
    assert hilbert_side(8, 3) == 8


def test_rectangular_index_uses_enclosing_square() -> None:
    """Indices on a rectangle equal those on its enclosing power-of-two square."""
    assert hilbert_index(25, 22, 0, 0) == 0
    assert hilbert_index(25, 22, 7, 3) == xy2d(32, 7, 3)


def test_clipped_curve_covers_every_cell_once() -> None:
    """Clipping to a non power-of-two rectangle keeps each cell exactly once."""
    cells = [c for seg in curve_segments(25, 22) for c in seg]
    assert len(cells) == len(set(cells)) == 25 * 22


def test_clipped_curve_segments_are_connected_paths() -> None:
    """Inside one segment, consecutive points are grid neighbours."""
    for seg in curve_segments(13, 12):
        for a, b in zip(seg, seg[1:], strict=False):
            assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def test_clipped_curve_on_square_power_of_two_is_one_segment() -> None:
    """A power-of-two square needs no clipping, so there is a single polyline."""
    segs = curve_segments(8, 8)
    assert len(segs) == 1 and len(segs[0]) == 64


@pytest.mark.parametrize("n", [1, 2, 8, 32])
def test_batch_encoding_matches_scalar(n: int) -> None:
    """The vectorised encoder agrees with the scalar one on every cell."""
    xs, ys = np.meshgrid(np.arange(n), np.arange(n))
    batch = xy2d_batch(n, xs, ys)
    assert batch.shape == (n, n)
    expected = [[xy2d(n, int(x), int(y)) for x in range(n)] for y in range(n)]
    assert batch.tolist() == expected


def test_batch_encoding_handles_empty_input() -> None:
    """An empty batch returns an empty array rather than failing."""
    out = xy2d_batch(8, np.array([], dtype=int), np.array([], dtype=int))
    assert out.shape == (0,)


def test_batch_encoding_rejects_bad_input() -> None:
    """Shape mismatches, bad sides and out-of-grid cells raise ``ValueError``."""
    with pytest.raises(ValueError):
        xy2d_batch(8, np.array([1, 2]), np.array([1]))
    with pytest.raises(ValueError):
        xy2d_batch(6, np.array([1]), np.array([1]))
    with pytest.raises(ValueError):
        xy2d_batch(8, np.array([8]), np.array([0]))
