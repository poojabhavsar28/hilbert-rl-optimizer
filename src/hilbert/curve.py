"""2D Hilbert space-filling curve primitives.

A Hilbert curve visits every cell of a ``2**k x 2**k`` grid exactly once, and
consecutive cells along the curve are always grid neighbours. That gives a
locality-preserving map from 2D positions to a 1D index: cells that are close
on the grid tend to get close indices.

Only two dimensions are implemented here.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "curve_segments",
    "d2xy",
    "hilbert_index",
    "hilbert_side",
    "encode_unit",
    "next_pow2",
    "spatial_sort",
    "xy2d",
    "xy2d_batch",
]


def next_pow2(n: int) -> int:
    """Return the smallest power of two that is >= ``n`` (and at least 1)."""
    p = 1
    while p < n:
        p *= 2
    return p


def _check_side(n: int) -> None:
    """Raise ``ValueError`` unless ``n`` is a positive power of two."""
    if n < 1 or n & (n - 1):
        raise ValueError("n must be a power of two")


def _rot(n: int, x: int, y: int, rx: int, ry: int) -> tuple[int, int]:
    """Rotate/flip a quadrant so the sub-curve has the right orientation."""
    if ry == 0:
        if rx == 1:
            x = n - 1 - x
            y = n - 1 - y
        x, y = y, x
    return x, y


def xy2d(n: int, x: int, y: int) -> int:
    """Return the distance along the Hilbert curve of cell ``(x, y)``.

    Args:
        n: Side length of the square grid; must be a power of two.
        x: Column of the cell, ``0 <= x < n``.
        y: Row of the cell, ``0 <= y < n``.

    Returns:
        The cell's index along the curve, in ``[0, n * n)``.

    Raises:
        ValueError: If ``n`` is not a power of two or the cell is outside the grid.
    """
    _check_side(n)
    if not (0 <= x < n and 0 <= y < n):
        raise ValueError("cell outside the grid")
    d = 0
    s = n // 2
    while s > 0:
        rx = 1 if (x & s) > 0 else 0
        ry = 1 if (y & s) > 0 else 0
        d += s * s * ((3 * rx) ^ ry)
        x, y = _rot(n, x, y, rx, ry)
        s //= 2
    return d


def d2xy(n: int, d: int) -> tuple[int, int]:
    """Return the cell ``(x, y)`` at distance ``d`` along the Hilbert curve.

    This is the inverse of :func:`xy2d`.

    Args:
        n: Side length of the square grid; must be a power of two.
        d: Index along the curve, ``0 <= d < n * n``.

    Raises:
        ValueError: If ``n`` is not a power of two or ``d`` is out of range.
    """
    _check_side(n)
    if not (0 <= d < n * n):
        raise ValueError("index outside the curve")
    x = y = 0
    t = d
    s = 1
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        x, y = _rot(s, x, y, rx, ry)
        x += s * rx
        y += s * ry
        t //= 4
        s *= 2
    return x, y


def hilbert_side(width: int, height: int) -> int:
    """Return the side of the power-of-two square that encloses the grid."""
    return next_pow2(max(width, height))


def hilbert_index(width: int, height: int, x: int, y: int) -> int:
    """Return the Hilbert index of a cell in a rectangular grid.

    The rectangle is embedded in its enclosing power-of-two square, so indices
    are comparable across every cell of the same grid.
    """
    return xy2d(hilbert_side(width, height), x, y)


def curve_segments(width: int, height: int) -> list[list[tuple[int, int]]]:
    """Return the Hilbert curve clipped to a ``width x height`` rectangle.

    The curve leaves and re-enters the rectangle when it is not a power-of-two
    square, so the result is a list of polylines, split wherever the curve
    steps outside. Every cell of the rectangle appears exactly once.
    """
    n = hilbert_side(width, height)
    segments: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []
    for d in range(n * n):
        x, y = d2xy(n, d)
        if x >= width or y >= height:
            if current:
                segments.append(current)
                current = []
            continue
        current.append((x, y))
    if current:
        segments.append(current)
    return segments


def xy2d_batch(
    n: int, x: NDArray[np.integer], y: NDArray[np.integer]
) -> NDArray[np.int64]:
    """Vectorised :func:`xy2d` for arrays of cells.

    Args:
        n: Side length of the square grid; must be a power of two.
        x: Integer columns, any shape.
        y: Integer rows, same shape as ``x``.

    Returns:
        An ``int64`` array of curve indices with the same shape as the inputs.

    Raises:
        ValueError: If ``n`` is not a power of two, the shapes differ, or a
            cell lies outside the grid.
    """
    _check_side(n)
    xs = np.asarray(x, dtype=np.int64).copy()
    ys = np.asarray(y, dtype=np.int64).copy()
    if xs.shape != ys.shape:
        raise ValueError("x and y must have the same shape")
    if xs.size and (xs.min() < 0 or ys.min() < 0 or xs.max() >= n or ys.max() >= n):
        raise ValueError("cell outside the grid")
    d = np.zeros_like(xs)
    s = n // 2
    while s > 0:
        rx = ((xs & s) > 0).astype(np.int64)
        ry = ((ys & s) > 0).astype(np.int64)
        d += s * s * ((3 * rx) ^ ry)
        flip = (ry == 0) & (rx == 1)
        xs = np.where(flip, n - 1 - xs, xs)
        ys = np.where(flip, n - 1 - ys, ys)
        swap = ry == 0
        xs, ys = np.where(swap, ys, xs), np.where(swap, xs, ys)
        s //= 2
    return d


def spatial_sort(cells: Iterable[tuple[int, int]], n: int) -> list[tuple[int, int]]:
    """Sort grid cells along the Hilbert curve.

    The sort is stable, so cells with equal indices (duplicates) keep their
    input order. Visiting cells in this order is a cheap locality-preserving
    tour, though it ignores obstacles.

    Args:
        cells: ``(x, y)`` cells, each inside the ``n x n`` grid.
        n: Side length of the square grid; must be a power of two.
    """
    return sorted(cells, key=lambda c: xy2d(n, c[0], c[1]))


def encode_unit(x: float, y: float, order: int = 10) -> float:
    """Encode a point of the unit square as a normalised Hilbert index.

    The square is divided into a ``2**order x 2**order`` grid and the point
    mapped to its cell, so nearby points get nearby values in ``[0, 1)``.

    Args:
        x: Horizontal coordinate in ``[0, 1]``.
        y: Vertical coordinate in ``[0, 1]``.
        order: Resolution; the grid has ``2**order`` cells per side (1 to 16).

    Raises:
        ValueError: If a coordinate is outside ``[0, 1]`` or ``order`` is invalid.
    """
    if not 1 <= order <= 16:
        raise ValueError("order must be between 1 and 16")
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        raise ValueError("coordinates must lie in [0, 1]")
    n = 1 << order
    cx = min(int(x * n), n - 1)
    cy = min(int(y * n), n - 1)
    return xy2d(n, cx, cy) / (n * n)
