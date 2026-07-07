"""
Placeholder tests for Phase 1 — ensures CI passes immediately.

Real tests are added in Phase 2 (Hilbert encoder tests)
and Phase 3 (environment tests).
"""


def test_project_imports() -> None:
    """Verify the src package is importable — catches install issues early."""
    import src  # noqa: F401


def test_python_version() -> None:
    """Ensure we are running the correct Python version."""
    import sys

    assert sys.version_info >= (3, 10), (
        f"Python 3.10+ required, got {sys.version_info}"
    )


def test_core_dependencies_importable() -> None:
    """Verify all critical dependencies installed correctly."""
    import numpy  # noqa: F401
    import gymnasium  # noqa: F401
    import torch  # noqa: F401
    import pydantic  # noqa: F401
    import yaml  # noqa: F401
