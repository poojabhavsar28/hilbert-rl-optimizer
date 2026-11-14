"""Hilbert curve encoding: locality-preserving 2D to 1D mapping."""

from src.hilbert.curve import (
    curve_segments,
    d2xy,
    encode_unit,
    hilbert_index,
    hilbert_side,
    next_pow2,
    spatial_sort,
    xy2d,
    xy2d_batch,
)
from src.hilbert.morton import (
    morton_decode,
    morton_index,
)

__all__ = [
    "curve_segments",
    "d2xy",
    "encode_unit",
    "hilbert_index",
    "hilbert_side",
    "morton_decode",
    "morton_index",
    "next_pow2",
    "spatial_sort",
    "xy2d",
    "xy2d_batch",
]
