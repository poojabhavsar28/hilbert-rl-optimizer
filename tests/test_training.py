"""Tests for REINFORCE training."""

import numpy as np
import pytest

from src.environment.warehouse import Warehouse
from src.rl.agent import FEATURE_NAMES, Agent
from src.rl.training import train


def test_training_improves_over_untrained_agent() -> None:
    """A short training run beats the all-zero policy on unseen orders."""
    agent, history = train(episodes=300, seed=0, verbose=False)
    assert len(history) == 3
    wh = Warehouse.generate(8, 22)
    trained = untrained = 0.0
    for offset in range(15):
        inst = wh.random_instance(14, np.random.default_rng(100 + offset))
        trained += inst.length(agent.solve(inst, "greedy"))
        untrained += inst.length(Agent().solve(inst, "greedy"))
    assert trained < untrained


def test_training_is_deterministic_for_a_seed() -> None:
    """The same seed reproduces the same weights."""
    a, _ = train(episodes=40, seed=3, verbose=False)
    b, _ = train(episodes=40, seed=3, verbose=False)
    assert np.array_equal(a.theta, b.theta)


def test_disabled_features_stay_at_zero() -> None:
    """Ablated features are never given weight."""
    off = ("hilbert_forward_gap", "hilbert_abs_gap")
    agent, _ = train(episodes=60, seed=0, verbose=False, disable=off)
    for name, weight in zip(FEATURE_NAMES, agent.theta, strict=True):
        if name in off:
            assert weight == 0.0
    assert np.any(agent.theta != 0.0)


def test_unknown_disabled_feature_is_rejected() -> None:
    """Typos in ``disable`` fail loudly instead of silently training everything."""
    with pytest.raises(ValueError):
        train(episodes=1, disable=("not_a_feature",), verbose=False)


def test_verbose_training_prints_progress(capsys) -> None:
    """Progress lines are printed when ``verbose`` is set."""
    train(episodes=20, log_every=10, seed=0, verbose=True)
    assert "episode" in capsys.readouterr().out
