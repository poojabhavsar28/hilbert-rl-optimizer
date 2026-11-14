"""Tests for the Morton (Z-order) index."""

import pytest

from src.hilbert import morton_decode, morton_index


@pytest.mark.parametrize("n", [1, 2, 4, 16])
def test_morton_round_trip_and_bijection(n: int) -> None:
    """Every Z-order index decodes to a distinct cell and encodes back."""
    cells = set()
    for d in range(n * n):
        x, y = morton_decode(n, d)
        assert morton_index(n, x, y) == d
        cells.add((x, y))
    assert len(cells) == n * n


def test_morton_interleaves_bits() -> None:
    """x occupies the even bits and y the odd bits of the index."""
    assert morton_index(4, 1, 0) == 0b0001
    assert morton_index(4, 0, 1) == 0b0010
    assert morton_index(4, 3, 3) == 0b1111


def test_morton_rejects_bad_input() -> None:
    """Invalid sides, cells and indices raise ``ValueError``."""
    with pytest.raises(ValueError):
        morton_index(5, 0, 0)
    with pytest.raises(ValueError):
        morton_index(4, 4, 0)
    with pytest.raises(ValueError):
        morton_decode(4, 16)
    with pytest.raises(ValueError):
        morton_decode(3, 0)
