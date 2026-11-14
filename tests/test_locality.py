"""Tests for the locality comparison between Hilbert and Z-order curves."""

import pytest

from src.hilbert import consecutive_jumps, d2xy, morton_decode


@pytest.mark.parametrize("n", [2, 8, 32])
def test_hilbert_has_no_discontinuities(n: int) -> None:
    """Every Hilbert step moves exactly one cell."""
    mean, worst = consecutive_jumps(d2xy, n)
    assert mean == 1.0 and worst == 1


@pytest.mark.parametrize("n", [4, 8, 32])
def test_morton_has_discontinuities(n: int) -> None:
    """Z-order jumps across quadrant boundaries, so it is strictly worse."""
    h_mean, h_worst = consecutive_jumps(d2xy, n)
    m_mean, m_worst = consecutive_jumps(morton_decode, n)
    assert m_worst > h_worst
    assert m_mean > h_mean


def test_morton_worst_jump_grows_with_grid_size() -> None:
    """The Z-order worst case scales with the grid; Hilbert stays at 1."""
    assert (
        consecutive_jumps(morton_decode, 32)[1] > consecutive_jumps(morton_decode, 8)[1]
    )


def test_single_cell_grid_has_no_jumps() -> None:
    """A 1x1 grid has no consecutive pairs."""
    assert consecutive_jumps(d2xy, 1) == (0.0, 0)
