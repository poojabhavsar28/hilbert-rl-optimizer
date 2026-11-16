"""Tests for the Gymnasium routing environment."""

import warnings

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from src.environment import solvers as S
from src.environment.gym_env import ENV_ID, WarehouseRoutingEnv


def rollout(env: WarehouseRoutingEnv, policy) -> tuple[float, dict]:
    """Run one episode with ``policy(env) -> action``; return (return, last info)."""
    total, done, info = 0.0, False, {}
    while not done:
        _, r, term, trunc, info = env.step(policy(env))
        total += r
        done = term or trunc
    return total, info


def first_valid(env: WarehouseRoutingEnv) -> int:
    return int(np.flatnonzero(env.action_masks())[0])


@pytest.mark.parametrize("mode", ["xy", "hilbert"])
def test_passes_the_gymnasium_checker(mode: str) -> None:
    env = gym.make(ENV_ID, obs_mode=mode, render_mode="ansi")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        check_env(env.unwrapped)


def test_observation_sizes_differ_by_mode() -> None:
    xy = WarehouseRoutingEnv(k=8, obs_mode="xy")
    hil = WarehouseRoutingEnv(k=8, obs_mode="hilbert")
    assert xy.observation_space.shape == (2 + 3 * 8,)
    assert hil.observation_space.shape == (1 + 2 * 8,)
    obs, _ = hil.reset(seed=0)
    assert hil.observation_space.contains(obs)


def test_same_seed_gives_the_same_order_and_observation() -> None:
    a, b = WarehouseRoutingEnv(), WarehouseRoutingEnv()
    oa, _ = a.reset(seed=7)
    ob, _ = b.reset(seed=7)
    assert np.array_equal(oa, ob)
    assert a.instance.cells == b.instance.cells


def test_return_is_the_negative_scaled_route_length() -> None:
    """Whatever the policy, the episode return matches the walked distance."""
    env = WarehouseRoutingEnv(k=6)
    env.reset(seed=3)
    total, info = rollout(env, first_valid)
    expected = -env.instance.length(list(range(1, 7))) / (
        env._wh.width + env._wh.height
    )
    assert total == pytest.approx(expected)
    assert info["route_length"] == pytest.approx(env.instance.length(list(range(1, 7))))


def test_following_a_solver_route_reproduces_its_length() -> None:
    env = WarehouseRoutingEnv(k=8, obs_mode="hilbert")
    env.reset(seed=11)
    order = S.nearest_neighbor(env.instance)
    it = iter(order)
    _, info = rollout(env, lambda e: next(it) - 1)
    assert info["route_length"] == pytest.approx(env.instance.length(order))


def test_mask_tracks_picked_items_and_ends_the_episode() -> None:
    env = WarehouseRoutingEnv(k=4)
    _, info = env.reset(seed=0)
    assert info["action_mask"].all()
    _, _, term, _, info = env.step(2)
    assert not info["action_mask"][2] and info["action_mask"].sum() == 3 and not term
    for a in (0, 1, 3):
        _, _, term, trunc, _ = env.step(a)
    assert term and not trunc


def test_invalid_action_is_penalised_and_changes_nothing() -> None:
    env = WarehouseRoutingEnv(k=4, invalid_penalty=2.0)
    env.reset(seed=0)
    env.step(1)
    before = env._obs().copy()
    _, reward, term, trunc, info = env.step(1)
    assert info["invalid"] and reward == -2.0 and not term and not trunc
    assert np.array_equal(before, env._obs())


def test_endless_invalid_actions_are_truncated() -> None:
    env = WarehouseRoutingEnv(k=3)
    env.reset(seed=0)
    env.step(0)
    trunc = False
    for _ in range(10):
        _, _, term, trunc, _ = env.step(0)
        if trunc:
            break
    assert trunc and not term


def test_reset_options_choose_the_order_and_layout() -> None:
    env = WarehouseRoutingEnv(k=3, layouts=[(4, 10), (6, 14)])
    items = env._warehouses[1].pick_cells()[:3]
    env.reset(seed=0, options={"layout": 1, "items": items})
    assert env.instance.cells[1:] == items
    assert env._wh is env._warehouses[1]
    with pytest.raises(ValueError, match="expected 3 items"):
        env.reset(options={"items": items[:2]})


def test_multiple_layouts_keep_a_fixed_observation_size() -> None:
    env = WarehouseRoutingEnv(k=5, layouts=[(4, 10), (8, 20)])
    seen = set()
    for seed in range(12):
        obs, _ = env.reset(seed=seed)
        assert env.observation_space.contains(obs)
        seen.add((env._wh.width, env._wh.height))
    assert len(seen) == 2


def test_step_before_reset_and_bad_action_raise() -> None:
    env = WarehouseRoutingEnv()
    with pytest.raises(RuntimeError, match="reset"):
        env.step(0)
    env.reset(seed=0)
    with pytest.raises(ValueError, match="not in"):
        env.step(99)


@pytest.mark.parametrize(
    "kwargs",
    [{"obs_mode": "polar"}, {"k": 0}, {"k": 10_000}, {"render_mode": "human"}],
)
def test_bad_constructor_arguments(kwargs) -> None:
    with pytest.raises(ValueError):
        WarehouseRoutingEnv(**kwargs)


def test_render_draws_the_depot_items_and_picked_marks() -> None:
    env = WarehouseRoutingEnv(k=3, render_mode="ansi")
    env.reset(seed=0)
    env.step(0)
    text = env.render()
    assert "D" in text and "x" in text and "1" in text
    assert WarehouseRoutingEnv().render() is None
