"""Tests for the tabular Q-learning navigation testbed."""

import numpy as np
import pytest

from src.environment.warehouse import Warehouse
from src.rl.navigation import (
    ACTIONS,
    GOAL_REWARD,
    GridNav,
    compare_replay,
    format_replay_report,
    greedy_steps,
    run,
)


@pytest.fixture(scope="module")
def env() -> GridNav:
    """A small navigation task with a far goal."""
    wh = Warehouse.generate(3, 9)
    goal = max(wh.pick_cells(), key=lambda c: wh.distance(wh.depot, c))
    return GridNav(wh, goal)


def test_step_moves_blocks_and_rewards(env: GridNav) -> None:
    """Moves into shelves or edges leave the agent in place; the goal ends the episode."""
    wh = env.wh
    pos, reward, done = env.step(wh.depot, ACTIONS.index((0, 1)))  # off the map
    assert pos == wh.depot and reward < 0 and not done
    pos, reward, done = env.step(wh.depot, ACTIONS.index((0, -1)))
    assert pos == (wh.depot[0], wh.depot[1] - 1) and not done
    gx, gy = env.goal
    prev = env.wh.path(wh.depot, env.goal)[-2]
    action = ACTIONS.index((gx - prev[0], gy - prev[1]))
    _, reward, done = env.step(prev, action)
    assert done and reward == GOAL_REWARD


def test_greedy_steps_detects_success_and_loops(env: GridNav) -> None:
    """A zero Q-table walks into a wall forever (None); a BFS-shaped table succeeds."""
    q = np.zeros((env.wh.height, env.wh.width, len(ACTIONS)))
    assert greedy_steps(env, q) is None
    path = env.wh.path(env.wh.depot, env.goal)
    for a, b in zip(path, path[1:], strict=False):
        q[a[1], a[0], ACTIONS.index((b[0] - a[0], b[1] - a[1]))] = 1.0
    assert greedy_steps(env, q) == env.optimal


@pytest.mark.parametrize("strategy", ["uniform", "stratified", "local"])
def test_run_reports_consistent_results(strategy: str) -> None:
    """Each strategy returns well-formed results for a short run."""
    r = run(strategy, seed=0, episodes=12, aisles=3, height=9)
    assert r["strategy"] == strategy and r["updates"] > 0
    assert len(r["curve"]) == 4
    assert r["final_steps"] is None or r["final_steps"] >= r["optimal_steps"]


def test_uniform_replay_learns_the_optimal_path() -> None:
    """With enough episodes the uniform learner finds a shortest path."""
    r = run("uniform", seed=0, episodes=80)
    assert r["final_steps"] == r["optimal_steps"]
    assert r["solved_at_episode"] is not None


def test_unknown_strategy_is_rejected() -> None:
    """A misspelt strategy fails loudly."""
    with pytest.raises(ValueError):
        run("nope", seed=0, episodes=1)


def test_compare_and_report() -> None:
    """The comparison summarises every strategy and renders a text table."""
    rep = compare_replay(seeds=1, episodes=15, strategies=("uniform", "local"))
    assert set(rep["summary"]) == {"uniform", "local"}
    text = format_replay_report(rep)
    assert "uniform" in text and "seeds solved" in text
