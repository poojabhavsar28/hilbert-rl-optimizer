"""2D Hilbert space-filling curve primitives.

A Hilbert curve visits every cell of a ``2**k x 2**k`` grid exactly once, and
consecutive cells along the curve are always grid neighbours. That gives a
locality-preserving map from 2D positions to a 1D index: cells that are close
on the grid tend to get close indices.

Only two dimensions are implemented here.
"""

from __future__ import annotations

__all__ = [
    "curve_segments",
    "d2xy",
    "hilbert_index",
    "hilbert_side",
    "next_pow2",
    "xy2d",
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
