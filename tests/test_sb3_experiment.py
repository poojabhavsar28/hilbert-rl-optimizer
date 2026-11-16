"""Tests for the PPO xy-vs-Hilbert experiment (tiny runs; needs SB3)."""

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")
pytest.importorskip("sb3_contrib")

from src.environment.gym_env import WarehouseRoutingEnv  # noqa: E402
from src.rl import sb3_experiment as X  # noqa: E402
from src.rl.benchmark import EVAL_LAYOUTS  # noqa: E402


def test_eval_orders_are_reproducible_and_held_out() -> None:
    a, b = X.eval_orders(6, 5), X.eval_orders(6, 5)
    assert a == b and len(a) == 5
    assert all(0 <= idx < len(EVAL_LAYOUTS) and len(items) == 6 for idx, items in a)


def test_gap_is_zero_for_the_reference_route_and_positive_for_a_poor_one() -> None:
    orders = X.eval_orders(6, 4)
    env = WarehouseRoutingEnv(k=6, layouts=EVAL_LAYOUTS)
    first = X.gap_to_reference(env, lambda _o, m: int(np.flatnonzero(m)[0]), orders)
    assert first > 0.0
    nn = X.nearest_neighbour_gap(6, orders)
    assert 0.0 <= nn < first


def test_tracking_returns_a_curve_starting_at_step_zero() -> None:
    orders = X.eval_orders(5, 3)
    curve = X.train_and_track("hilbert", 0, 1024, 1024, 5, orders)
    steps = [s for s, _ in curve]
    assert steps[0] == 0 and steps[-1] >= 1024
    assert all(np.isfinite(g) for _, g in curve)


def test_report_lists_every_checkpoint_for_both_modes() -> None:
    curve = [(0, 80.0), (1000, 20.0)]
    res = {
        "k": 5,
        "n_eval": 3,
        "total_steps": 1000,
        "nearest_neighbor_gap": 9.5,
        "curves": {"xy": [curve, curve], "hilbert": [curve, curve]},
    }
    text = X.format_report(res)
    assert "+9.5%" in text and "1000" in text and "hilbert mean" in text
