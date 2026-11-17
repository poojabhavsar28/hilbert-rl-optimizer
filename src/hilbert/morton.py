"""Morton (Z-order) curve, included as a locality baseline for the Hilbert curve.

A Z-order index interleaves the bits of ``x`` and ``y``. It is cheaper to
compute than a Hilbert index but has discontinuities: two consecutive indices
can be far apart on the grid (see :mod:`src.hilbert.locality`).
"""

from __future__ import annotations

__all__ = ["morton_decode", "morton_index"]


def _check_side(n: int) -> None:
    """Raise ``ValueError`` unless ``n`` is a positive power of two."""
    if n < 1 or n & (n - 1):
        raise ValueError("n must be a power of two")


def morton_index(n: int, x: int, y: int) -> int:
    """Return the Z-order index of cell ``(x, y)`` on an ``n x n`` grid.

    Raises:
        ValueError: If ``n`` is not a power of two or the cell is outside the grid.
    """
    _check_side(n)
    if not (0 <= x < n and 0 <= y < n):
        raise ValueError("cell outside the grid")
    d = 0
    for bit in range(n.bit_length() - 1):
        d |= ((x >> bit) & 1) << (2 * bit)
        d |= ((y >> bit) & 1) << (2 * bit + 1)
    return d


def morton_decode(n: int, d: int) -> tuple[int, int]:
    """Return the cell ``(x, y)`` at Z-order index ``d``; inverse of :func:`morton_index`.

    Raises:
        ValueError: If ``n`` is not a power of two or ``d`` is out of range.
    """
    _check_side(n)
    if not (0 <= d < n * n):
        raise ValueError("index outside the curve")
    x = y = 0
    for bit in range(n.bit_length() - 1):
        x |= ((d >> (2 * bit)) & 1) << bit
        y |= ((d >> (2 * bit + 1)) & 1) << bit
    return x, y
