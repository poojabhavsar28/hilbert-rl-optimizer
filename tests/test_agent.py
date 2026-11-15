"""Tests for the REINFORCE routing agent."""

import numpy as np
import pytest

from src.environment.warehouse import Warehouse
from src.rl.agent import (
    N_FEATURES,
    WEIGHTS_ENV_VAR,
    Agent,
    _features,
    default_weights_path,
)


@pytest.fixture(scope="module")
def wh() -> Warehouse:
    """A small generated warehouse shared by the agent tests."""
    return Warehouse.generate(6, 14)


@pytest.mark.parametrize("mode", ["greedy", "best_of_n"])
def test_routes_are_valid_permutations(wh: Warehouse, mode: str, seed: int) -> None:
    """The agent visits every item exactly once, in both modes."""
    inst = wh.random_instance(12, np.random.default_rng(seed))
    assert sorted(Agent().solve(inst, mode)) == list(range(1, 13))


def test_solve_rejects_unknown_mode_and_handles_empty(wh: Warehouse, seed: int) -> None:
    """Unknown modes raise; an empty order gives an empty route."""
    inst = wh.random_instance(4, np.random.default_rng(seed))
    with pytest.raises(ValueError):
        Agent().solve(inst, "bogus")
    assert Agent().solve(wh.instance([])) == []


def test_best_of_n_is_never_worse_than_greedy(wh: Warehouse, seed: int) -> None:
    """Sampling can only improve on the greedy route, which is always included."""
    agent = Agent(theta=np.array([-3.0, 1.0, 0.0, -1.0, 2.0, 0.5]))
    for offset in range(5):
        inst = wh.random_instance(10, np.random.default_rng(seed + offset))
        sampled = inst.length(agent.solve(inst, "best_of_n", 8))
        assert sampled <= inst.length(agent.solve(inst, "greedy")) + 1e-9


def test_rollout_shapes_and_gradient_guard(wh: Warehouse, seed: int) -> None:
    """Rollouts return per-route orders, lengths and gradients."""
    inst = wh.random_instance(6, np.random.default_rng(seed))
    rng = np.random.default_rng(seed)
    orders, lengths, grad = Agent().rollout(inst, 5, rng, want_grad=True)
    assert orders.shape == (5, 6) and lengths.shape == (5,)
    assert grad.shape == (5, N_FEATURES)
    for row, length in zip(orders, lengths, strict=True):
        assert inst.length(row.tolist()) == pytest.approx(length)
    with pytest.raises(ValueError):
        Agent().rollout(inst, 2, rng, greedy=True, want_grad=True)


def test_policy_gradient_matches_finite_difference(seed: int) -> None:
    """The analytic grad-log-prob used by REINFORCE equals a numerical derivative."""
    inst = Warehouse.generate(4, 10).random_instance(5, np.random.default_rng(3))
    theta = np.random.default_rng(seed).normal(size=N_FEATURES)
    agent = Agent(theta=theta.copy())
    orders, _, grad = agent.rollout(inst, 1, np.random.default_rng(7), want_grad=True)
    route = orders[0].tolist()
    dn = inst.D / inst.D.max()

    def route_logp(th: np.ndarray) -> float:
        """Log-probability of the sampled route under weights ``th``."""
        rem = np.ones((1, inst.k + 1), dtype=bool)
        rem[0, 0] = False
        cur = np.array([0])
        total = 0.0
        for node in route:
            feats = _features(dn, inst.hilbert, inst.hilbert_n, cur, rem)
            z = np.where(rem, feats @ th, -np.inf)[0]
            z = z - z.max()
            total += z[node] - np.log(np.exp(z).sum())
            rem[0, node] = False
            cur = np.array([node])
        return float(total)

    numeric = np.zeros(N_FEATURES)
    for i in range(N_FEATURES):
        e = np.zeros(N_FEATURES)
        e[i] = 1e-5
        numeric[i] = (route_logp(theta + e) - route_logp(theta - e)) / 2e-5
    assert np.allclose(grad[0], numeric, atol=1e-4)


def test_weights_round_trip(tmp_path) -> None:
    """Saved weights load back unchanged; a mismatched feature set is rejected."""
    a = Agent(theta=np.arange(N_FEATURES, dtype=float))
    path = tmp_path / "w.json"
    a.save(path, meta={"note": "test"})
    assert np.allclose(Agent.load(path).theta, a.theta)
    path.write_text('{"features": ["other"], "theta": [1.0]}')
    with pytest.raises(ValueError):
        Agent.load(path)


def test_default_weights_path_honours_env_var(monkeypatch, tmp_path) -> None:
    """``HILBERT_RL_WEIGHTS`` overrides the default weights location."""
    monkeypatch.delenv(WEIGHTS_ENV_VAR, raising=False)
    assert default_weights_path().name == "routing_agent.json"
    monkeypatch.setenv(WEIGHTS_ENV_VAR, str(tmp_path / "x.json"))
    assert default_weights_path() == tmp_path / "x.json"


def test_shipped_weights_beat_nearest_neighbor_on_held_out_layout() -> None:
    """The committed weights beat nearest neighbour on a layout never trained on."""
    from src.environment import solvers as S

    agent = Agent.load()
    wh = Warehouse.generate(5, 16)  # held out from training
    rl = nn = 0.0
    for offset in range(25):
        inst = wh.random_instance(15, np.random.default_rng(500 + offset))
        rl += inst.length(agent.solve(inst, "greedy"))
        nn += inst.length(S.nearest_neighbor(inst))
    assert rl < nn
