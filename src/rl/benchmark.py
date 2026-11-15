"""Benchmark every routing method on held-out warehouses, and ablate features."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from src.environment import solvers as S
from src.environment.warehouse import Instance, Warehouse
from src.rl.agent import Agent
from src.rl.training import train

# Held-out layouts: none of these appear in the training set (TRAIN_LAYOUTS).
EVAL_LAYOUTS = [(5, 16), (11, 28), (12, 30)]
HILBERT_FEATURES = ("hilbert_forward_gap", "hilbert_abs_gap")


def benchmark(
    agent: Agent,
    n_instances: int = 40,
    ks: tuple[int, ...] = (8, 12, 20, 30),
    seed: int = 12345,
) -> dict[str, dict]:
    """Compare all methods on random orders from held-out warehouses.

    The reference is the exact optimum when ``k <= HELD_KARP_MAX_K``, otherwise
    the 2-opt heuristic.

    Args:
        agent: The trained agent to evaluate.
        n_instances: Random orders per size.
        ks: Order sizes (items per order).
        seed: Seed for the orders.

    Returns:
        ``{str(k): row}`` where each row holds the mean route length per method,
        ``_reference``, ``_pct_over_reference`` per method, and the share of
        orders on which the greedy agent beat nearest neighbour.
    """
    rng = np.random.default_rng(seed)
    warehouses = [Warehouse.generate(a, h) for a, h in EVAL_LAYOUTS]
    methods: dict[str, Callable[[Instance], list[int]]] = {
        "snake": S.snake_order,
        "nearest_neighbor": S.nearest_neighbor,
        "hilbert_sort": S.hilbert_order,
        "rl_greedy": lambda i: agent.solve(i, "greedy"),
        "rl_best_of_16": lambda i: agent.solve(
            i, "best_of_n", 16, np.random.default_rng(0)
        ),
        "two_opt": S.best_heuristic,
    }
    out: dict[str, dict] = {}
    for k in ks:
        lens: dict[str, list[float]] = {m: [] for m in methods}
        opt: list[float] = []
        for _ in range(n_instances):
            inst = warehouses[int(rng.integers(len(warehouses)))].random_instance(
                k, rng
            )
            for name, fn in methods.items():
                lens[name].append(inst.length(fn(inst)))
            if k <= S.HELD_KARP_MAX_K:
                opt.append(inst.length(S.held_karp(inst)))
        row: dict = {m: float(np.mean(v)) for m, v in lens.items()}
        if opt:
            row["optimal"] = float(np.mean(opt))
        ref = row.get("optimal", row["two_opt"])
        row["_reference"] = "optimal" if opt else "two_opt"
        row["_pct_over_reference"] = {m: 100.0 * (row[m] / ref - 1.0) for m in methods}
        wins = [
            a < b
            for a, b in zip(lens["rl_greedy"], lens["nearest_neighbor"], strict=True)
        ]
        row["_rl_beats_nearest_neighbor_pct_of_instances"] = float(
            100.0 * np.mean(wins)
        )
        out[str(k)] = row
    return out


def format_benchmark(res: dict[str, dict]) -> str:
    """Render a :func:`benchmark` result as a text table."""
    lines = [f"{'k':>4}  {'method':<18}{'mean length':>12}{'vs ref':>9}"]
    for k, row in res.items():
        for m, pct in row["_pct_over_reference"].items():
            lines.append(f"{k:>4}  {m:<18}{row[m]:>12.1f}{pct:>+8.1f}%")
        if "optimal" in row:
            lines.append(f"{k:>4}  {'optimal':<18}{row['optimal']:>12.1f}{0.0:>+8.1f}%")
        share = row["_rl_beats_nearest_neighbor_pct_of_instances"]
        lines.append(
            f"      (reference = {row['_reference']}; RL beats nearest-neighbour "
            f"on {share:.0f}% of instances)"
        )
    return "\n".join(lines)


def ablation(
    seeds: int = 5,
    episodes: int = 1500,
    n_instances: int = 40,
    ks: tuple[int, ...] = (8, 12, 20, 30),
) -> dict[str, dict]:
    """Train with and without the Hilbert features and compare on identical orders.

    Returns:
        For each variant, the mean (and std across training seeds) of the
        percentage by which greedy and best-of-16 routes exceed 2-opt.
    """
    rng = np.random.default_rng(777)
    warehouses = [Warehouse.generate(a, h) for a, h in EVAL_LAYOUTS]
    insts = [
        warehouses[int(rng.integers(len(warehouses)))].random_instance(k, rng)
        for k in ks
        for _ in range(n_instances)
    ]
    ref = np.array([i.length(S.best_heuristic(i)) for i in insts])
    out: dict[str, dict] = {}
    variants = (
        ("with_hilbert_features", ()),
        ("without_hilbert_features", HILBERT_FEATURES),
    )
    for label, disable in variants:
        greedy: list[float] = []
        best16: list[float] = []
        for seed in range(seeds):
            agent, _ = train(
                episodes=episodes, seed=seed, verbose=False, disable=disable
            )
            g = np.array([i.length(agent.solve(i, "greedy")) for i in insts])
            b = np.array(
                [
                    i.length(agent.solve(i, "best_of_n", 16, np.random.default_rng(0)))
                    for i in insts
                ]
            )
            greedy.append(float(100 * (g / ref - 1).mean()))
            best16.append(float(100 * (b / ref - 1).mean()))
        out[label] = {
            "greedy_pct_over_2opt_mean": float(np.mean(greedy)),
            "greedy_std": float(np.std(greedy)),
            "best_of_16_pct_over_2opt_mean": float(np.mean(best16)),
            "best_of_16_std": float(np.std(best16)),
            "greedy_per_seed": greedy,
        }
    return out


def format_ablation(res: dict[str, dict], seeds: int) -> str:
    """Render an :func:`ablation` result as text."""
    lines = [
        f"Mean % over 2-opt on identical held-out orders "
        f"({seeds} training seeds, +/- std across seeds)",
        "",
    ]
    for label, v in res.items():
        lines.append(
            f"{label:<28} greedy {v['greedy_pct_over_2opt_mean']:+.2f}% "
            f"(+/-{v['greedy_std']:.2f})   best-of-16 "
            f"{v['best_of_16_pct_over_2opt_mean']:+.2f}% (+/-{v['best_of_16_std']:.2f})"
        )
    return "\n".join(lines)
