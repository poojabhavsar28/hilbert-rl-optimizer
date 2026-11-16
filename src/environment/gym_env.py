"""A Gymnasium environment for pick routing.

The agent builds a pick route one stop at a time: each action chooses which
unpicked item to walk to next. The episode ends after the last item, when the
walk back to the depot is added to the final reward, so the undiscounted return
is exactly ``-(route length) / (width + height)``.

Two observation modes let the central hypothesis be tested with the same agent:

* ``"xy"``: the current position and each item's ``(x, y)``, normalised to
  ``[0, 1]``, plus a picked flag per item.
* ``"hilbert"``: the same, but every position is a single Hilbert-curve index
  normalised to ``[0, 1]``, so the observation is about half as wide.

Import this module only where Gymnasium is installed. The API never imports it.
Passing an ``action_mask`` (see :meth:`WarehouseRoutingEnv.action_masks`) is
supported by ``sb3_contrib.MaskablePPO``.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from src.environment.warehouse import Cell, Instance, Warehouse

OBS_MODES = ("xy", "hilbert")
ENV_ID = "HilbertWarehouse-v0"


class WarehouseRoutingEnv(gym.Env):
    """Choose the next item to pick; reward is the negative walking distance.

    Args:
        aisles: Number of aisles of the generated warehouse.
        height: Height of the generated warehouse.
        k: Items per order (fixed, so the observation has a fixed size).
        obs_mode: ``"xy"`` or ``"hilbert"``.
        layouts: Optional list of ``(aisles, height)`` pairs; one is chosen at
            random on every reset. Overrides ``aisles`` and ``height``.
        invalid_penalty: Reward for choosing an already picked item (the state
            does not change). The episode is truncated after ``2 * k`` steps.
        render_mode: ``None`` or ``"ansi"``.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 1}

    def __init__(
        self,
        aisles: int = 6,
        height: int = 14,
        k: int = 8,
        obs_mode: str = "xy",
        layouts: list[tuple[int, int]] | None = None,
        invalid_penalty: float = 1.0,
        render_mode: str | None = None,
    ) -> None:
        """Build the warehouses and the action and observation spaces."""
        super().__init__()
        if obs_mode not in OBS_MODES:
            raise ValueError(f"obs_mode must be one of {OBS_MODES}, got {obs_mode!r}")
        if k < 1:
            raise ValueError("k must be at least 1")
        if render_mode not in (None, "ansi"):
            raise ValueError(f"unsupported render_mode {render_mode!r}")
        self.k = k
        self.obs_mode = obs_mode
        self.invalid_penalty = float(invalid_penalty)
        self.render_mode = render_mode
        self._warehouses = [
            Warehouse.generate(a, h) for a, h in (layouts or [(aisles, height)])
        ]
        picks = min(len(w.pick_cells()) for w in self._warehouses)
        if k > picks:
            raise ValueError(
                f"k={k} exceeds the {picks} pick cells of the smallest layout"
            )
        self.action_space = spaces.Discrete(k)
        width = 2 + 3 * k if obs_mode == "xy" else 1 + 2 * k
        self.observation_space = spaces.Box(0.0, 1.0, shape=(width,), dtype=np.float32)
        self._wh: Warehouse = self._warehouses[0]
        self._inst: Instance | None = None
        self._cur = 0
        self._picked = np.zeros(k, dtype=bool)
        self._steps = 0
        self._walked = 0.0

    # -- helpers -----------------------------------------------------------
    @property
    def instance(self) -> Instance:
        """The current order (call :meth:`reset` first)."""
        if self._inst is None:
            raise RuntimeError("call reset() before using the environment")
        return self._inst

    @property
    def _scale(self) -> float:
        return float(self._wh.width + self._wh.height)

    def _position(self, node: int) -> NDArray[np.float32]:
        inst = self.instance
        if self.obs_mode == "hilbert":
            return np.array([inst.hilbert[node] / inst.hilbert_n], dtype=np.float32)
        x, y = inst.cells[node]
        return np.array(
            [x / max(self._wh.width - 1, 1), y / max(self._wh.height - 1, 1)],
            dtype=np.float32,
        )

    def _obs(self) -> NDArray[np.float32]:
        parts = [self._position(self._cur)]
        parts += [self._position(i + 1) for i in range(self.k)]
        parts.append(self._picked.astype(np.float32))
        return np.concatenate(parts)

    def action_masks(self) -> NDArray[np.bool_]:
        """Return which actions are valid (the items not yet picked)."""
        return ~self._picked

    # -- Gymnasium API -----------------------------------------------------
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        """Start a new order.

        ``options`` may hold ``"layout"`` (index into ``layouts``) and
        ``"items"`` (``k`` distinct pick cells) to evaluate a specific order.
        """
        super().reset(seed=seed)
        options = options or {}
        layout = options.get("layout")
        if layout is None:
            layout = int(self.np_random.integers(len(self._warehouses)))
        self._wh = self._warehouses[layout]
        items: list[Cell] | None = options.get("items")
        if items is not None:
            if len(items) != self.k:
                raise ValueError(f"expected {self.k} items, got {len(items)}")
            self._inst = self._wh.instance(items)
        else:
            self._inst = self._wh.random_instance(self.k, self.np_random)
        self._cur = 0
        self._picked = np.zeros(self.k, dtype=bool)
        self._steps = 0
        self._walked = 0.0
        return self._obs(), {"action_mask": self.action_masks()}

    def step(
        self, action: int
    ) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        """Walk to item ``action``; finishing the order adds the walk home."""
        inst = self.instance
        if not self.action_space.contains(int(action)):
            raise ValueError(f"action {action!r} is not in {self.action_space}")
        self._steps += 1
        info: dict[str, Any] = {"invalid": False}
        terminated = False
        if self._picked[action]:
            reward = -self.invalid_penalty
            info["invalid"] = True
        else:
            node = int(action) + 1
            dist = float(inst.D[self._cur, node])
            self._walked += dist
            self._cur = node
            self._picked[action] = True
            reward = -dist / self._scale
            if self._picked.all():
                home = float(inst.D[node, 0])
                self._walked += home
                reward -= home / self._scale
                terminated = True
        truncated = not terminated and self._steps >= 2 * self.k
        info["route_length"] = self._walked
        info["action_mask"] = self.action_masks()
        return self._obs(), float(reward), terminated, truncated, info

    def render(self) -> str | None:
        """Draw the warehouse: ``D`` depot, ``#`` shelf, digits for items.

        Picked items show as ``x``. Only the ``"ansi"`` mode is supported.
        """
        if self.render_mode != "ansi":
            return None
        rows = [list(r) for r in self._wh.to_ascii().splitlines()]
        for i in range(self.k):
            x, y = self.instance.cells[i + 1]
            rows[y][x] = "x" if self._picked[i] else str(i % 10)
        return "\n".join("".join(r) for r in rows)


if ENV_ID not in gym.registry:
    gym.register(id=ENV_ID, entry_point="src.environment.gym_env:WarehouseRoutingEnv")
