"""Does a Hilbert observation speed up PPO? A controlled comparison.

Trains maskable PPO on :class:`~src.environment.gym_env.WarehouseRoutingEnv`
with three observation encodings (plain ``xy``, a Hilbert index alone, or both together), several
seeds each, and evaluates the greedy policy on *held-out* warehouses at regular
intervals. Everything else (network, hyper-parameters, orders, seeds) is the
same, so any difference comes from the observation.

Requires ``stable-baselines3`` and ``sb3-contrib`` (not part of the API image).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from src.environment import solvers as S
from src.environment.gym_env import WarehouseRoutingEnv
from src.rl.benchmark import EVAL_LAYOUTS
from src.rl.training import TRAIN_LAYOUTS

OBS_MODES = ("xy", "hilbert", "xy_hilbert")


def eval_orders(k: int, n: int, seed: int = 999) -> list[tuple[int, list]]:
    """Return ``n`` reproducible held-out orders as ``(layout index, items)``."""
    from src.environment.warehouse import Warehouse

    rng = np.random.default_rng(seed)
    whs = [Warehouse.generate(a, h) for a, h in EVAL_LAYOUTS]
    out = []
    for _ in range(n):
        idx = int(rng.integers(len(whs)))
        out.append((idx, whs[idx].random_instance(k, rng).cells[1:]))
    return out


def gap_to_reference(
    env: WarehouseRoutingEnv,
    act: Callable[[np.ndarray, np.ndarray], int],
    orders: list[tuple[int, list]],
) -> float:
    """Return the mean % by which a policy's routes exceed the 2-opt reference.

    ``act(obs, mask)`` returns the action. Lower is better; 0 matches 2-opt.
    """
    gaps = []
    for layout, items in orders:
        obs, info = env.reset(options={"layout": layout, "items": items})
        ref = env.instance.length(S.best_heuristic(env.instance))
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(act(obs, info["action_mask"]))
            done = term or trunc
        gaps.append(100.0 * (info["route_length"] / ref - 1.0))
    return float(np.mean(gaps))


def nearest_neighbour_gap(k: int, orders: list[tuple[int, list]]) -> float:
    """Return the nearest-neighbour heuristic's gap on the same held-out orders."""
    env = WarehouseRoutingEnv(k=k, layouts=EVAL_LAYOUTS)

    def act(_obs: np.ndarray, mask: np.ndarray) -> int:
        inst = env.instance
        cur = env._cur
        valid = np.flatnonzero(mask)
        return int(valid[np.argmin(inst.D[cur, valid + 1])])

    return gap_to_reference(env, act, orders)


def train_and_track(
    obs_mode: str,
    seed: int,
    total_steps: int,
    eval_every: int,
    k: int,
    orders: list[tuple[int, list]],
) -> list[tuple[int, float]]:
    """Train one PPO run; return ``[(steps, held-out gap %)]`` at each checkpoint."""
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    from stable_baselines3.common.callbacks import BaseCallback

    env = ActionMasker(
        WarehouseRoutingEnv(k=k, obs_mode=obs_mode, layouts=TRAIN_LAYOUTS),
        lambda e: e.unwrapped.action_masks(),
    )
    eval_env = WarehouseRoutingEnv(k=k, obs_mode=obs_mode, layouts=EVAL_LAYOUTS)
    curve: list[tuple[int, float]] = []
    model = MaskablePPO(
        "MlpPolicy", env, seed=seed, verbose=0, device="cpu", n_steps=1024
    )

    def measure(steps: int) -> None:
        def act(obs: np.ndarray, mask: np.ndarray) -> int:
            action, _ = model.predict(obs, action_masks=mask, deterministic=True)
            return int(action)

        curve.append((steps, gap_to_reference(eval_env, act, orders)))

    class Track(BaseCallback):
        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self) -> None:
            if self.num_timesteps % eval_every < model.n_steps:
                measure(self.num_timesteps)

    measure(0)
    model.learn(total_timesteps=total_steps, callback=Track())
    if curve[-1][0] != model.num_timesteps:
        measure(model.num_timesteps)
    return curve


def compare(
    seeds: int = 3,
    total_steps: int = 100_000,
    eval_every: int = 10_240,
    k: int = 10,
    n_eval: int = 30,
) -> dict:
    """Run both observation modes over several seeds on identical held-out orders."""
    orders = eval_orders(k, n_eval)
    result: dict = {
        "k": k,
        "n_eval": n_eval,
        "total_steps": total_steps,
        "nearest_neighbor_gap": nearest_neighbour_gap(k, orders),
        "curves": {m: [] for m in OBS_MODES},
    }
    for mode in OBS_MODES:
        for seed in range(seeds):
            result["curves"][mode].append(
                train_and_track(mode, seed, total_steps, eval_every, k, orders)
            )
    return result


def format_report(res: dict) -> str:
    """Render :func:`compare` output: mean gap (and spread) per checkpoint."""
    modes = [m for m in OBS_MODES if m in res["curves"]]
    n_seeds = len(res["curves"][modes[0]])
    lines = [
        f"PPO, k={res['k']} items, {res['n_eval']} held-out orders, "
        f"{n_seeds} seeds. Gap = % longer than 2-opt (lower is better).",
        f"nearest neighbour on the same orders: {res['nearest_neighbor_gap']:+.1f}%",
        "",
        f"{'steps':>9}  " + "".join(f"{m + ' mean (min..max)':>30}" for m in modes),
    ]
    arrays = {
        m: np.array([[g for _, g in run] for run in res["curves"][m]]) for m in modes
    }
    steps = [s for s, _ in res["curves"][modes[0]][0]]
    for i, step in enumerate(steps):
        cells = []
        for m in modes:
            col = arrays[m][:, i]
            cells.append(f"{col.mean():+7.1f}% ({col.min():+.1f}..{col.max():+.1f})")
        lines.append(f"{step:>9}  " + "".join(f"{c:>30}" for c in cells))
    return "\n".join(lines)
