"""Tabular Q-learning on a warehouse grid: a testbed for the replay buffers.

The agent starts at the depot and must walk to one pick location. Reward is -1
per step and +20 on arrival, so the optimal return follows from the BFS
distance. We compare how quickly the greedy policy becomes optimal under three
replay strategies (uniform, Hilbert-stratified, Hilbert-local) with everything
else held equal, across several seeds.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from src.environment.warehouse import Cell, Warehouse
from src.rl.replay import HilbertReplay, UniformReplay

ACTIONS = ((1, 0), (-1, 0), (0, 1), (0, -1))
GOAL_REWARD = 20.0
STEP_REWARD = -1.0
STRATEGIES = ("uniform", "stratified", "local")


class GridNav:
    """Navigate from the depot to a goal cell, one grid step at a time."""

    def __init__(self, wh: Warehouse, goal: Cell, max_steps: int = 200) -> None:
        """Create the task.

        Args:
            wh: The warehouse to walk in.
            goal: Cell to reach.
            max_steps: Episode length cap.
        """
        self.wh = wh
        self.goal = goal
        self.max_steps = max_steps
        self.optimal = wh.distance(wh.depot, goal)

    def step(self, pos: Cell, action: int) -> tuple[Cell, float, bool]:
        """Apply ``action`` at ``pos``; walls and edges leave the agent in place.

        Returns:
            ``(next_position, reward, done)``.
        """
        dx, dy = ACTIONS[action]
        nxt = (pos[0] + dx, pos[1] + dy)
        if not self.wh.walkable(nxt):
            nxt = pos
        done = nxt == self.goal
        return nxt, (GOAL_REWARD if done else STEP_REWARD), done


def greedy_steps(env: GridNav, q: NDArray[np.float64]) -> int | None:
    """Return the steps the greedy policy needs to reach the goal (None if never)."""
    pos = env.wh.depot
    seen: set[Cell] = set()
    for t in range(env.max_steps):
        if pos == env.goal:
            return t
        if pos in seen:  # a loop means the policy will never arrive
            return None
        seen.add(pos)
        pos, _, _ = env.step(pos, int(np.argmax(q[pos[1], pos[0]])))
    return env.max_steps if pos == env.goal else None


def _make_buffer(strategy: str, capacity: int, wh: Warehouse) -> UniformReplay:
    """Create the replay buffer for ``strategy``."""
    if strategy == "uniform":
        return UniformReplay(capacity)
    if strategy in ("stratified", "local"):
        return HilbertReplay(capacity, wh.width, wh.height, mode=strategy)
    raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")


def run(
    strategy: str,
    seed: int,
    episodes: int = 150,
    aisles: int = 4,
    height: int = 12,
    batch: int = 16,
    alpha: float = 0.3,
    gamma: float = 0.99,
    capacity: int = 20000,
    eval_every: int = 3,
) -> dict:
    """Train one tabular Q-learner and report how fast it becomes optimal.

    Args:
        strategy: ``"uniform"``, ``"stratified"`` or ``"local"`` replay.
        seed: Seed for the goal choice, exploration and sampling.
        episodes: Training episodes.
        aisles: Warehouse aisles.
        height: Warehouse rows.
        batch: Transitions replayed per environment step.
        alpha: Learning rate.
        gamma: Discount factor.
        capacity: Replay capacity.
        eval_every: Episodes between greedy evaluations.

    Returns:
        A dict with ``optimal_steps``, ``solved_at_episode`` (None if never),
        ``final_steps``, the evaluation ``curve`` and the ``updates`` count.
    """
    rng = np.random.default_rng(seed)
    wh = Warehouse.generate(aisles, height)
    picks = wh.pick_cells()
    # a pick location far from the depot, so exploration actually matters
    far = sorted(picks, key=lambda c: -wh.distance(wh.depot, c))
    far = far[: max(3, len(picks) // 10)]
    env = GridNav(wh, far[int(rng.integers(len(far)))])
    buf = _make_buffer(strategy, capacity, wh)

    q = np.zeros((wh.height, wh.width, len(ACTIONS)))
    solved_at: int | None = None
    curve: list[tuple[int, int | None]] = []
    updates = 0
    for ep in range(1, episodes + 1):
        eps = max(0.05, 1.0 - ep / (0.6 * episodes))
        pos = wh.depot
        for _ in range(env.max_steps):
            if rng.random() < eps:
                a = int(rng.integers(len(ACTIONS)))
            else:
                row = q[pos[1], pos[0]]
                a = int(rng.choice(np.flatnonzero(row == row.max())))
            nxt, r, done = env.step(pos, a)
            buf.add(pos[0], pos[1], a, r, nxt[0], nxt[1], done)
            x, y, aa, rr, nx, ny, dd = buf.sample(batch, rng, anchor=pos)
            target = rr + gamma * (1.0 - dd) * q[ny, nx].max(axis=1)
            q[y, x, aa] += alpha * (target - q[y, x, aa])
            updates += 1
            pos = nxt
            if done:
                break
        if ep % eval_every == 0:
            steps = greedy_steps(env, q)
            curve.append((ep, steps))
            if solved_at is None and steps == env.optimal:
                solved_at = ep
    return {
        "strategy": strategy,
        "seed": seed,
        "optimal_steps": env.optimal,
        "solved_at_episode": solved_at,
        "final_steps": greedy_steps(env, q),
        "curve": curve,
        "updates": updates,
    }


def compare_replay(
    seeds: int = 5,
    episodes: int = 150,
    strategies: tuple[str, ...] = STRATEGIES,
) -> dict:
    """Run every strategy on the same seeds and summarise the results.

    Returns:
        ``{"episodes", "summary", "runs"}`` where ``summary`` maps each strategy to
        how many seeds reached the optimal path, the median/mean episode they did
        so, and the mean final gap to optimal (unsolved runs count as 3x optimal).
    """
    results = {s: [run(s, seed, episodes) for seed in range(seeds)] for s in strategies}
    summary = {}
    for s, runs in results.items():
        solved = [r["solved_at_episode"] for r in runs if r["solved_at_episode"]]
        gaps = [
            (
                r["final_steps"]
                if r["final_steps"] is not None
                else 3 * r["optimal_steps"]
            )
            - r["optimal_steps"]
            for r in runs
        ]
        summary[s] = {
            "seeds": len(runs),
            "solved_optimal": len(solved),
            "median_episodes_to_optimal": float(np.median(solved)) if solved else None,
            "mean_episodes_to_optimal": float(np.mean(solved)) if solved else None,
            "final_gap_to_optimal_mean": float(np.mean(gaps)),
        }
    return {"episodes": episodes, "summary": summary, "runs": results}


def format_replay_report(report: dict) -> str:
    """Render a :func:`compare_replay` result as a text table."""
    lines = [
        f"Tabular Q-learning, {report['episodes']} training episodes per run",
        "",
        f"{'replay':<12}{'seeds solved':>14}{'median ep. to optimal':>24}"
        f"{'final gap (steps)':>20}",
    ]
    for s, v in report["summary"].items():
        med = v["median_episodes_to_optimal"]
        med_s = "-" if med is None else f"{med:.0f}"
        solved = f"{v['solved_optimal']}/{v['seeds']}"
        gap = v["final_gap_to_optimal_mean"]
        lines.append(f"{s:<12}{solved:>14}{med_s:>24}{gap:>20.1f}")
    return "\n".join(lines)
