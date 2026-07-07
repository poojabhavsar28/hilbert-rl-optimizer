"""
Shared pytest fixtures for the Hilbert-RL test suite.

Fixtures defined here are automatically available to ALL test files
without any import needed — pytest picks them up automatically.
"""

import pytest
import numpy as np


@pytest.fixture(scope="session")
def seed() -> int:
    """Fixed random seed for reproducible tests."""
    return 42


@pytest.fixture(scope="session")
def rng(seed: int) -> np.random.Generator:
    """Seeded numpy random generator — use this in all tests instead of random."""
    return np.random.default_rng(seed)


@pytest.fixture
def sample_2d_coords() -> list[tuple[float, float]]:
    """A small set of known 2D coordinates for encoder tests."""
    return [
        (0.0, 0.0),   # origin
        (1.0, 1.0),   # far corner
        (0.5, 0.5),   # center
        (0.25, 0.75), # upper-left quadrant
        (0.75, 0.25), # lower-right quadrant
    ]


@pytest.fixture
def sample_3d_coords() -> list[tuple[float, float, float]]:
    """A small set of known 3D coordinates for ND encoder tests."""
    return [
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        (0.5, 0.5, 0.5),
        (0.1, 0.9, 0.5),
    ]
