"""Replay buffers for grid-world reinforcement learning.

:class:`UniformReplay` is the standard ring buffer, sampled uniformly.

:class:`HilbertReplay` is the **Hilbert-indexed replay buffer**. Every
transition is keyed by the Hilbert index of the cell it started in. That turns
a 2D position into a 1D key that preserves locality, so keeping transitions
sorted by key gives two cheap ways to sample:

``stratified``
    Draw targets uniformly along the *curve* (one per equal-width slice, so
    uniformly over space) and take the nearest stored transition. Densely
    revisited areas no longer dominate a batch and rarely visited areas are
    replayed more often.

``local``
    Pick a spatial anchor (the agent's current cell) and sample transitions
    from the neighbourhood of that key on the curve, so an update batch is
    spatially coherent.

Whether either helps learning is an empirical question; see
``docs/results/replay.txt``: in the tabular testbed neither beats uniform replay.
"""

from __future__ import annotations

from bisect import bisect_left, insort

import numpy as np
from numpy.typing import NDArray

from src.hilbert import hilbert_index, hilbert_side

# x, y, action, reward, next x, next y, done
Batch = tuple[
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.int64],
    NDArray[np.float64],
]


class UniformReplay:
    """Fixed-size ring buffer of ``(x, y, action, reward, x', y', done)``."""

    def __init__(self, capacity: int) -> None:
        """Create an empty buffer holding at most ``capacity`` transitions."""
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.size = 0
        self._next = 0
        self._data = np.zeros((capacity, 7), dtype=np.float64)

    def __len__(self) -> int:
        """Return the number of stored transitions."""
        return self.size

    def _store(self, row: tuple[float, ...]) -> int:
        """Write ``row`` into the next slot (overwriting the oldest) and return it."""
        slot = self._next
        self._data[slot] = row
        self._next = (self._next + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return slot

    def add(
        self,
        x: int,
        y: int,
        action: int,
        reward: float,
        nx: int,
        ny: int,
        done: bool,
    ) -> None:
        """Store one transition."""
        self._store((x, y, action, reward, nx, ny, float(done)))

    def _unpack(self, slots: NDArray[np.int64]) -> Batch:
        """Return the transitions stored at ``slots`` as column arrays."""
        d = self._data[slots]
        return (
            d[:, 0].astype(np.int64),
            d[:, 1].astype(np.int64),
            d[:, 2].astype(np.int64),
            d[:, 3],
            d[:, 4].astype(np.int64),
            d[:, 5].astype(np.int64),
            d[:, 6],
        )

    def sample(
        self,
        batch: int,
        rng: np.random.Generator,
        anchor: tuple[int, int] | None = None,
    ) -> Batch:
        """Sample ``batch`` transitions uniformly (``anchor`` is ignored).

        Raises:
            ValueError: If the buffer is empty.
        """
        if self.size == 0:
            raise ValueError("cannot sample from an empty buffer")
        return self._unpack(rng.integers(0, self.size, size=batch))


class HilbertReplay(UniformReplay):
    """Replay buffer indexed by the Hilbert index of each transition's start cell."""

    MODES = ("stratified", "local")

    def __init__(
        self,
        capacity: int,
        width: int,
        height: int,
        mode: str = "stratified",
        window: int = 64,
    ) -> None:
        """Create a buffer for a ``width x height`` grid.

        Args:
            capacity: Maximum number of stored transitions.
            width: Grid width.
            height: Grid height.
            mode: ``"stratified"`` or ``"local"`` (see the module docstring).
            window: For ``"local"``, how many curve neighbours on each side of
                the anchor to sample from.
        """
        super().__init__(capacity)
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        self.mode = mode
        self.window = window
        self.width = width
        self.height = height
        self.side = hilbert_side(width, height)
        self.n_cells = self.side * self.side
        self._index: list[tuple[int, int]] = []  # sorted (hilbert key, slot)
        self._slot_key = np.full(capacity, -1, dtype=np.int64)

    def key(self, x: int, y: int) -> int:
        """Return the Hilbert key of cell ``(x, y)``."""
        return hilbert_index(self.width, self.height, int(x), int(y))

    def add(
        self,
        x: int,
        y: int,
        action: int,
        reward: float,
        nx: int,
        ny: int,
        done: bool,
    ) -> None:
        """Store one transition, evicting the oldest one when full."""
        slot = self._next
        if self.size == self.capacity:  # overwriting: drop the evicted key
            old = (int(self._slot_key[slot]), slot)
            pos = bisect_left(self._index, old)
            if pos < len(self._index) and self._index[pos] == old:
                del self._index[pos]
        self._store((x, y, action, reward, nx, ny, float(done)))
        k = self.key(x, y)
        self._slot_key[slot] = k
        insort(self._index, (k, slot))

    def sample(
        self,
        batch: int,
        rng: np.random.Generator,
        anchor: tuple[int, int] | None = None,
    ) -> Batch:
        """Sample ``batch`` transitions using the buffer's Hilbert sampling mode.

        Args:
            batch: Number of transitions.
            rng: Random generator.
            anchor: Cell to centre ``"local"`` sampling on (ignored otherwise).

        Raises:
            ValueError: If the buffer is empty.
        """
        if self.size == 0:
            raise ValueError("cannot sample from an empty buffer")
        n = len(self._index)
        ranks: list[int] = []
        if self.mode == "stratified":
            targets = (np.arange(batch) + rng.random(batch)) / batch * self.n_cells
            for t in targets:
                pos = bisect_left(self._index, (int(t), -1))
                # take whichever neighbour is closer on the curve
                if pos >= n:
                    pos = n - 1
                elif pos > 0 and t - self._index[pos - 1][0] < self._index[pos][0] - t:
                    pos -= 1
                # many transitions share one key (same start cell): pick uniformly
                # within that group instead of always returning its first element
                key = self._index[pos][0]
                lo = bisect_left(self._index, (key, -1))
                hi = bisect_left(self._index, (key + 1, -1))
                ranks.append(int(rng.integers(lo, hi)))
        else:  # local
            if anchor is None:
                centre = int(rng.integers(0, n))
            else:
                centre = min(bisect_left(self._index, (self.key(*anchor), -1)), n - 1)
            lo = max(0, centre - self.window)
            hi = min(n, centre + self.window + 1)
            ranks = rng.integers(lo, hi, size=batch).tolist()
        slots = np.fromiter(
            (self._index[r][1] for r in ranks), dtype=np.int64, count=len(ranks)
        )
        return self._unpack(slots)
