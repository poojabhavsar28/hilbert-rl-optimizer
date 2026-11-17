"""A small reinforcement-learning agent for pick routing.

The agent builds a route one stop at a time. At each step it scores every
unpicked item with a linear function of a few features and samples the next
stop from a softmax over those scores (a stochastic policy). Training is in
:mod:`src.rl.training`.

Two features come from the Hilbert curve (the forward and absolute gap between
the current and candidate item along the curve); the rest are walking-distance
features. An ablation (:func:`src.rl.benchmark.ablation`) shows the agent does
not need the Hilbert features.

Pure NumPy: no torch required.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.environment.warehouse import Instance

FEATURE_NAMES = [
    "dist_from_current",
    "dist_to_depot",
    "hilbert_forward_gap",
    "hilbert_abs_gap",
    "mean_dist_to_remaining",
    "dist_to_nearest_remaining",
]
N_FEATURES = len(FEATURE_NAMES)
WEIGHTS_ENV_VAR = "HILBERT_RL_WEIGHTS"
_DEFAULT_WEIGHTS = Path(__file__).resolve().parents[2] / "models" / "routing_agent.json"


def default_weights_path() -> Path:
    """Return where trained weights are read from.

    Honours the ``HILBERT_RL_WEIGHTS`` environment variable, otherwise
    ``models/routing_agent.json`` in the repository root.
    """
    override = os.environ.get(WEIGHTS_ENV_VAR)
    return Path(override) if override else _DEFAULT_WEIGHTS


def _features(
    dn: NDArray[np.float64],
    hil: NDArray[np.float64],
    hil_n: float,
    cur: NDArray[np.int64],
    rem: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Compute the features of every candidate item.

    Args:
        dn: ``(n, n)`` distance matrix scaled to ``[0, 1]``.
        hil: ``(n,)`` Hilbert index of each node.
        hil_n: Number of cells on the Hilbert square.
        cur: ``(B,)`` current node of each of ``B`` parallel routes.
        rem: ``(B, n)`` mask of items still to pick.

    Returns:
        An array of shape ``(B, n, N_FEATURES)``.
    """
    batch, n = rem.shape
    f_cur = dn[cur]
    f_dep = np.broadcast_to(dn[0], (batch, n))
    gap = ((hil[None, :] - hil[cur][:, None]) % hil_n) / hil_n
    gap_abs = np.minimum(gap, 1.0 - gap)
    remf = rem.astype(np.float64)
    cnt = np.maximum(remf.sum(axis=1, keepdims=True) - 1.0, 1.0)
    f_cent = (remf @ dn) / cnt
    eye = np.eye(n, dtype=bool)[None]
    masked = np.where(rem[:, None, :] & ~eye, dn[None], np.inf)
    f_near = masked.min(axis=2)
    f_near[~np.isfinite(f_near)] = 0.0
    return np.stack([f_cur, f_dep, gap, gap_abs, f_cent, f_near], axis=-1)


@dataclass
class Agent:
    """A linear softmax routing policy.

    Attributes:
        theta: One weight per entry of ``FEATURE_NAMES``.
    """

    theta: NDArray[np.float64] = field(default_factory=lambda: np.zeros(N_FEATURES))

    def rollout(
        self,
        inst: Instance,
        batch: int,
        rng: np.random.Generator,
        greedy: bool = False,
        want_grad: bool = False,
    ) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
        """Build ``batch`` routes on one instance in parallel.

        Args:
            inst: The routing instance.
            batch: Number of routes to build.
            rng: Random generator used when sampling stops.
            greedy: Take the highest-scoring stop instead of sampling.
            want_grad: Also accumulate the gradient of each route's log-probability.

        Returns:
            ``(orders, lengths, grad_logp)`` with shapes ``(B, k)``, ``(B,)`` and
            ``(B, N_FEATURES)``. ``grad_logp`` is zero unless ``want_grad``.

        Raises:
            ValueError: If gradients are requested for a greedy rollout.
        """
        if want_grad and greedy:
            raise ValueError("gradients need a stochastic rollout")
        k = inst.k
        n = k + 1
        dist = inst.D
        scale = float(dist.max()) or 1.0
        dn = dist / scale
        rem = np.ones((batch, n), dtype=bool)
        rem[:, 0] = False
        cur = np.zeros(batch, dtype=np.int64)
        orders = np.zeros((batch, k), dtype=np.int64)
        lengths = np.zeros(batch)
        grad = np.zeros((batch, N_FEATURES))
        rows = np.arange(batch)
        for t in range(k):
            feats = _features(dn, inst.hilbert, inst.hilbert_n, cur, rem)
            logits = np.where(rem, feats @ self.theta, -np.inf)
            if greedy:
                a = logits.argmax(axis=1)
            else:
                logits = logits - logits.max(axis=1, keepdims=True)
                p = np.exp(logits)
                p /= p.sum(axis=1, keepdims=True)
                u = rng.random((batch, 1))
                a = np.minimum((p.cumsum(axis=1) > u).argmax(axis=1), n - 1)
                if want_grad:
                    grad += feats[rows, a] - (p[..., None] * feats).sum(axis=1)
            lengths += dist[cur, a]
            cur = a
            rem[rows, a] = False
            orders[:, t] = a
        lengths += dist[cur, 0]
        return orders, lengths, grad

    def solve(
        self,
        inst: Instance,
        mode: str = "greedy",
        samples: int = 16,
        rng: np.random.Generator | None = None,
    ) -> list[int]:
        """Route one instance.

        Args:
            inst: The routing instance.
            mode: ``"greedy"`` follows the policy's top choice; ``"best_of_n"``
                also samples ``samples`` stochastic routes and keeps the shortest.
            samples: Number of sampled routes for ``"best_of_n"``.
            rng: Random generator for sampling; defaults to a fixed seed.

        Raises:
            ValueError: If ``mode`` is not recognised.
        """
        if mode not in ("greedy", "best_of_n"):
            raise ValueError("mode must be 'greedy' or 'best_of_n'")
        if inst.k == 0:
            return []
        orders, lengths, _ = self.rollout(
            inst, 1, np.random.default_rng(0), greedy=True
        )
        best_order: list[int] = orders[0].tolist()
        best_len = float(lengths[0])
        if mode == "best_of_n":
            rng = rng or np.random.default_rng(0)
            orders, lengths, _ = self.rollout(inst, max(1, samples), rng)
            i = int(lengths.argmin())
            if lengths[i] < best_len:
                best_order = orders[i].tolist()
        return best_order

    def save(self, path: Path | None = None, meta: dict | None = None) -> None:
        """Write the weights (and optional metadata) as JSON.

        Args:
            path: Destination; defaults to :func:`default_weights_path`.
            meta: Extra information to store alongside the weights.
        """
        payload = {
            "features": FEATURE_NAMES,
            "theta": [float(v) for v in self.theta],
            "meta": meta or {},
        }
        Path(path or default_weights_path()).write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: Path | None = None) -> Agent:
        """Load weights saved by :meth:`save`.

        Raises:
            OSError: If the file cannot be read.
            ValueError: If it was trained with a different feature set.
        """
        data = json.loads(Path(path or default_weights_path()).read_text())
        if data.get("features") != FEATURE_NAMES:
            raise ValueError("weights file was trained with a different feature set")
        return cls(theta=np.array(data["theta"], dtype=np.float64))
