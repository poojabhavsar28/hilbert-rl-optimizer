"""Tests for the uniform and Hilbert-indexed replay buffers."""

import numpy as np
import pytest

from src.rl.replay import HilbertReplay, UniformReplay


def fill(buf: UniformReplay, n: int = 500, seed: int = 0) -> None:
    """Add ``n`` random transitions on a 13x12 grid."""
    rng = np.random.default_rng(seed)
    for _ in range(n):
        x, y = int(rng.integers(0, 13)), int(rng.integers(0, 12))
        buf.add(x, y, int(rng.integers(4)), -1.0, x, y, False)


def test_uniform_buffer_stores_and_wraps() -> None:
    """The ring buffer never exceeds its capacity and overwrites the oldest."""
    buf = UniformReplay(10)
    fill(buf, 25)
    assert len(buf) == 10
    x, y, a, r, nx, ny, done = buf.sample(6, np.random.default_rng(0))
    assert len(x) == len(y) == len(a) == len(r) == len(nx) == len(ny) == len(done) == 6


@pytest.mark.parametrize("mode", ["stratified", "local"])
def test_hilbert_buffer_samples_valid_cells_and_evicts(mode: str) -> None:
    """Index and storage stay in sync through evictions; samples are in range."""
    buf = HilbertReplay(100, 13, 12, mode=mode)
    fill(buf, 350)
    assert len(buf) == 100 and len(buf._index) == 100
    keys = [k for k, _ in buf._index]
    assert keys == sorted(keys)
    x, y, *_ = buf.sample(32, np.random.default_rng(0), anchor=(3, 3))
    assert len(x) == 32
    assert (x >= 0).all() and (x < 13).all() and (y >= 0).all() and (y < 12).all()


def test_local_sampling_without_anchor_works() -> None:
    """``local`` mode picks a random centre when no anchor is given."""
    buf = HilbertReplay(200, 13, 12, mode="local")
    fill(buf, 200)
    x, *_ = buf.sample(16, np.random.default_rng(1))
    assert len(x) == 16


def test_local_sampling_stays_near_the_anchor() -> None:
    """Local batches come from cells close to the anchor on the Hilbert curve."""
    buf = HilbertReplay(5000, 16, 16, mode="local", window=8)
    rng = np.random.default_rng(0)
    for _ in range(3000):
        x, y = int(rng.integers(0, 16)), int(rng.integers(0, 16))
        buf.add(x, y, 0, -1.0, x, y, False)
    xs, ys, *_ = buf.sample(200, rng, anchor=(2, 2))
    spread_local = np.mean(np.hypot(xs - 2, ys - 2))
    uni = UniformReplay(5000)
    uni._data[:] = buf._data
    uni.size = buf.size
    xu, yu, *_ = uni.sample(200, rng)
    assert spread_local < np.mean(np.hypot(xu - 2, yu - 2))


def test_stratified_is_more_spatially_even_than_uniform() -> None:
    """With 90% of transitions in one corner, stratified still reaches the rest."""
    rng = np.random.default_rng(0)

    def build(buf: UniformReplay) -> UniformReplay:
        """Fill ``buf`` with a heavily skewed visitation pattern."""
        for _ in range(2000):
            if rng.random() < 0.9:
                x, y = int(rng.integers(0, 3)), int(rng.integers(0, 3))
            else:
                x, y = int(rng.integers(0, 13)), int(rng.integers(0, 12))
            buf.add(x, y, 0, -1.0, x, y, False)
        return buf

    uni = build(UniformReplay(5000))
    hil = build(HilbertReplay(5000, 13, 12, "stratified"))

    def far_fraction(b: UniformReplay) -> float:
        """Share of sampled transitions that start outside the busy corner."""
        draws = [b.sample(64, np.random.default_rng(s))[0] >= 3 for s in range(20)]
        return float(np.mean(draws))

    assert far_fraction(hil) > far_fraction(uni) + 0.1


def test_empty_buffers_and_bad_arguments() -> None:
    """Sampling an empty buffer and bad constructor arguments raise ``ValueError``."""
    with pytest.raises(ValueError):
        UniformReplay(10).sample(4, np.random.default_rng(0))
    with pytest.raises(ValueError):
        HilbertReplay(10, 8, 8).sample(4, np.random.default_rng(0))
    with pytest.raises(ValueError):
        HilbertReplay(10, 8, 8, mode="nope")
    with pytest.raises(ValueError):
        UniformReplay(0)
