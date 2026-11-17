"""Measure how well a space-filling curve preserves locality."""

from __future__ import annotations

from collections.abc import Callable

__all__ = ["consecutive_jumps"]


def consecutive_jumps(
    decode: Callable[[int, int], tuple[int, int]], n: int
) -> tuple[float, int]:
    """Measure the grid distance between cells with consecutive curve indices.

    A perfectly continuous curve (the Hilbert curve) has every jump equal to 1.

    Args:
        decode: Maps ``(n, index)`` to a cell, e.g. ``d2xy`` or ``morton_decode``.
        n: Side length of the square grid; must be a power of two.

    Returns:
        ``(mean, max)`` Manhattan distance between cells ``d`` and ``d + 1``.
    """
    total = 0
    worst = 0
    prev = decode(n, 0)
    for d in range(1, n * n):
        cur = decode(n, d)
        step = abs(cur[0] - prev[0]) + abs(cur[1] - prev[1])
        total += step
        worst = max(worst, step)
        prev = cur
    count = n * n - 1
    return (total / count if count else 0.0), worst
