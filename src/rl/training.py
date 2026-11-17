"""REINFORCE training for the routing agent."""

from __future__ import annotations

import numpy as np

from src.environment.solvers import nearest_neighbor
from src.environment.warehouse import Warehouse
from src.rl.agent import FEATURE_NAMES, N_FEATURES, Agent

# (aisles, height) of the layouts used for training. Evaluation uses different ones.
TRAIN_LAYOUTS = [(8, 22), (6, 18), (10, 24), (7, 20), (9, 26)]


def train(
    episodes: int = 1500,
    batch: int = 16,
    lr: float = 0.05,
    k_range: tuple[int, int] = (8, 20),
    seed: int = 0,
    log_every: int = 100,
    verbose: bool = True,
    disable: tuple[str, ...] = (),
) -> tuple[Agent, list[dict[str, float]]]:
    """Train an agent with REINFORCE and a per-order baseline (Adam updates).

    Each episode draws a warehouse and a random order, samples ``batch`` routes
    on it, and rewards each route relative to the others (standardised
    advantage), which removes the need for a learned value function.

    Args:
        episodes: Number of orders to train on.
        batch: Routes sampled per order.
        lr: Adam learning rate.
        k_range: Inclusive range of items per order.
        seed: Seed for orders and sampling.
        log_every: Episodes between history entries.
        verbose: Print progress.
        disable: Names of features to hold at zero weight (for ablations).

    Returns:
        ``(agent, history)`` where ``history`` records the mean sampled route
        length relative to nearest neighbour every ``log_every`` episodes.

    Raises:
        ValueError: If ``disable`` names an unknown feature.
    """
    unknown = set(disable) - set(FEATURE_NAMES)
    if unknown:
        raise ValueError(f"unknown features: {sorted(unknown)}")
    off = np.array([name in disable for name in FEATURE_NAMES])
    rng = np.random.default_rng(seed)
    warehouses = [Warehouse.generate(a, h) for a, h in TRAIN_LAYOUTS]
    agent = Agent()
    m = np.zeros(N_FEATURES)
    v = np.zeros(N_FEATURES)
    b1, b2, eps = 0.9, 0.999, 1e-8
    history: list[dict[str, float]] = []
    window: list[float] = []
    for ep in range(1, episodes + 1):
        wh = warehouses[int(rng.integers(len(warehouses)))]
        k = int(rng.integers(k_range[0], k_range[1] + 1))
        inst = wh.random_instance(k, rng)
        _, lengths, grad = agent.rollout(inst, batch, rng, want_grad=True)
        std = lengths.std()
        if std > 1e-9:
            adv = (lengths.mean() - lengths) / (std + 1e-8)
        else:
            adv = np.zeros(batch)
        g = (adv[:, None] * grad).mean(axis=0)
        g[off] = 0.0
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mhat = m / (1 - b1**ep)
        vhat = v / (1 - b2**ep)
        step = lr * mhat / (np.sqrt(vhat) + eps)
        agent.theta = np.clip(agent.theta + step, -50.0, 50.0)
        nn_len = inst.length(nearest_neighbor(inst))
        window.append(float(lengths.mean() / nn_len))
        if ep % log_every == 0:
            entry = {
                "episode": float(ep),
                "sampled_vs_nearest_neighbor": float(np.mean(window)),
            }
            history.append(entry)
            window = []
            if verbose:
                ratio = entry["sampled_vs_nearest_neighbor"]
                print(
                    f"episode {ep:5d}  sampled route / nearest-neighbour = {ratio:.3f}"
                )
    return agent, history
