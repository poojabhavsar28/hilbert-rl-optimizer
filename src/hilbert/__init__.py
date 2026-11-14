"""Hilbert curve encoding: locality-preserving 2D to 1D mapping."""

from src.hilbert.curve import (
    curve_segments,
    d2xy,
    hilbert_index,
    hilbert_side,
    next_pow2,
    xy2d,
)

__all__ = [
    "curve_segments",
    "d2xy",
    "hilbert_index",
    "hilbert_side",
    "next_pow2",
    "xy2d",
]
