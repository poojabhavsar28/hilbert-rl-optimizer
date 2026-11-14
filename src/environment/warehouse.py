"""Warehouse model: a grid with shelves, a depot, walking distances and routes."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from src.hilbert import hilbert_index
from src.hilbert import hilbert_side as enclosing_side

Cell = tuple[int, int]  # (x, y): column, row counted from the top-left

MAX_SIDE = 200
_NEIGHBOURS = ((1, 0), (-1, 0), (0, 1), (0, -1))


class LayoutError(ValueError):
    """Raised when a warehouse layout or an order is invalid."""


def generate_layout(aisles: int = 8, height: int = 22) -> str:
    """Build a classic block-and-aisle warehouse as an ASCII map.

    ``#`` is a shelf, ``.`` is walkable floor and ``D`` is the depot. Shelf
    blocks are two cells wide and separated by one-cell vertical aisles; there
    are cross-aisles along the top and bottom and one through the middle. The
    depot is the bottom-left corner.

    Args:
        aisles: Number of aisle columns (2 to 40); the map is ``3 * aisles + 1`` wide.
        height: Number of rows (7 to ``MAX_SIDE``).

    Raises:
        LayoutError: If a dimension is out of range.
    """
    if not (2 <= aisles <= 40):
        raise LayoutError("aisles must be between 2 and 40")
    if not (7 <= height <= MAX_SIDE):
        raise LayoutError(f"height must be between 7 and {MAX_SIDE}")
    width = aisles * 3 + 1
    if width > MAX_SIDE:
        raise LayoutError(f"layout would be wider than {MAX_SIDE} cells")
    mid = height // 2
    rows = []
    for y in range(height):
        row = []
        for x in range(width):
            in_block = x % 3 != 0
            cross = y in (0, height - 1, mid)
            row.append("#" if in_block and not cross else ".")
        rows.append(row)
    rows[height - 1][0] = "D"
    return "\n".join("".join(r) for r in rows)


@dataclass(eq=False)
class Warehouse:
    """A grid warehouse with one depot.

    Attributes:
        grid: Boolean array of shape ``(height, width)``; ``True`` marks a shelf.
        depot: Cell where every trip starts and ends.
    """

    grid: np.ndarray
    depot: Cell
    _dist_cache: dict[Cell, np.ndarray] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ build
    @classmethod
    def from_ascii(cls, text: str) -> Warehouse:
        """Parse an ASCII map (``#`` shelf, ``.`` floor, one ``D`` depot).

        Raises:
            LayoutError: If the map is empty, ragged, too large, has an unknown
                character, or does not contain exactly one depot.
        """
        lines = [ln.rstrip() for ln in text.strip("\n").splitlines()]
        lines = [ln for ln in lines if ln != ""]
        if not lines:
            raise LayoutError("layout is empty")
        width = len(lines[0])
        if any(len(ln) != width for ln in lines):
            raise LayoutError("layout rows must all have the same length")
        if width > MAX_SIDE or len(lines) > MAX_SIDE:
            raise LayoutError(f"layout must be at most {MAX_SIDE} x {MAX_SIDE}")
        depot: Cell | None = None
        grid = np.zeros((len(lines), width), dtype=bool)
        for y, ln in enumerate(lines):
            for x, ch in enumerate(ln):
                if ch == "#":
                    grid[y, x] = True
                elif ch == "D":
                    if depot is not None:
                        raise LayoutError("layout must contain exactly one depot 'D'")
                    depot = (x, y)
                elif ch != ".":
                    raise LayoutError(
                        f"unknown layout character {ch!r}; use '#', '.', 'D'"
                    )
        if depot is None:
            raise LayoutError("layout must contain exactly one depot 'D'")
        return cls(grid=grid, depot=depot)

    @classmethod
    def generate(cls, aisles: int = 8, height: int = 22) -> Warehouse:
        """Build a generated block-and-aisle warehouse (see :func:`generate_layout`)."""
        return cls.from_ascii(generate_layout(aisles, height))

    def to_ascii(self) -> str:
        """Render the warehouse back to its ASCII map."""
        rows = []
        for y in range(self.height):
            rows.append(
                "".join(
                    "D" if (x, y) == self.depot else ("#" if self.grid[y, x] else ".")
                    for x in range(self.width)
                )
            )
        return "\n".join(rows)

    # ------------------------------------------------------------- properties
    @property
    def height(self) -> int:
        """Number of rows."""
        return int(self.grid.shape[0])

    @property
    def width(self) -> int:
        """Number of columns."""
        return int(self.grid.shape[1])

    def walkable(self, cell: Cell) -> bool:
        """Return whether ``cell`` is inside the map and not a shelf."""
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height and not self.grid[y, x]

    def pick_cells(self) -> list[Cell]:
        """Return walkable cells (other than the depot) that touch a shelf."""
        out = []
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y, x] or (x, y) == self.depot:
                    continue
                for dx, dy in _NEIGHBOURS:
                    nx, ny = x + dx, y + dy
                    in_map = 0 <= nx < self.width and 0 <= ny < self.height
                    if in_map and self.grid[ny, nx]:
                        out.append((x, y))
                        break
        return out

    # --------------------------------------------------------------- distance
    def _bfs(self, src: Cell) -> np.ndarray:
        """Return walking distances from ``src`` to every cell (-1 = unreachable).

        Results are cached per source cell.
        """
        cached = self._dist_cache.get(src)
        if cached is not None:
            return cached
        dist = np.full((self.height, self.width), -1, dtype=np.int32)
        dist[src[1], src[0]] = 0
        queue = deque([src])
        grid = self.grid
        w, h = self.width, self.height
        while queue:
            x, y = queue.popleft()
            d = dist[y, x] + 1
            for dx, dy in _NEIGHBOURS:
                nx, ny = x + dx, y + dy
                if (
                    0 <= nx < w
                    and 0 <= ny < h
                    and not grid[ny, nx]
                    and dist[ny, nx] < 0
                ):
                    dist[ny, nx] = d
                    queue.append((nx, ny))
        self._dist_cache[src] = dist
        return dist

    def distance(self, a: Cell, b: Cell) -> int:
        """Return the shortest walking distance from ``a`` to ``b`` (-1 if unreachable)."""
        return int(self._bfs(a)[b[1], b[0]])

    def path(self, a: Cell, b: Cell) -> list[Cell]:
        """Return one shortest walking path from ``a`` to ``b``, both ends included.

        Raises:
            LayoutError: If there is no walking path between the cells.
        """
        dist = self._bfs(b)  # distances *to* b, so walk downhill from a
        if dist[a[1], a[0]] < 0:
            raise LayoutError(f"no walking path between {a} and {b}")
        cur = a
        out = [cur]
        while cur != b:
            x, y = cur
            for dx, dy in _NEIGHBOURS:
                nx, ny = x + dx, y + dy
                in_map = 0 <= nx < self.width and 0 <= ny < self.height
                if in_map and dist[ny, nx] >= 0 and dist[ny, nx] == dist[y, x] - 1:
                    cur = (nx, ny)
                    break
            else:  # pragma: no cover - a BFS distance map always has a downhill step
                raise LayoutError(f"no walking path between {a} and {b}")
            out.append(cur)
        return out

    def route_path(self, cells: list[Cell]) -> list[Cell]:
        """Return the full cell-by-cell walk that visits ``cells`` in order."""
        out: list[Cell] = [cells[0]]
        for a, b in zip(cells, cells[1:], strict=False):
            out.extend(self.path(a, b)[1:])
        return out

    # --------------------------------------------------------------- instance
    def hilbert(self, cell: Cell) -> int:
        """Return the Hilbert index of ``cell`` on this warehouse's grid."""
        return hilbert_index(self.width, self.height, cell[0], cell[1])

    def hilbert_side(self) -> int:
        """Return the side of the power-of-two square the Hilbert curve covers."""
        return enclosing_side(self.width, self.height)

    def instance(self, items: list[Cell]) -> Instance:
        """Build a routing instance for the given pick cells (the depot is node 0).

        Raises:
            LayoutError: If an item is not a walkable cell or cannot be reached.
        """
        for c in items:
            if not self.walkable(tuple(c)):
                raise LayoutError(f"pick location {tuple(c)} is not a walkable cell")
        cells = [self.depot] + [tuple(c) for c in items]
        n = len(cells)
        dist_matrix = np.zeros((n, n), dtype=np.float64)
        for i, ci in enumerate(cells):
            dist = self._bfs(ci)
            for j, cj in enumerate(cells):
                v = dist[cj[1], cj[0]]
                if v < 0:
                    raise LayoutError(f"pick location {cj} cannot be reached from {ci}")
                dist_matrix[i, j] = v
        hil = np.array([self.hilbert(c) for c in cells], dtype=np.float64)
        return Instance(
            cells=cells,
            D=dist_matrix,
            hilbert=hil,
            hilbert_n=float(self.hilbert_side() ** 2),
        )

    def random_instance(self, k: int, rng: np.random.Generator) -> Instance:
        """Build an instance of ``k`` distinct random pick locations.

        Raises:
            LayoutError: If the warehouse has fewer than ``k`` pick locations.
        """
        picks = self.pick_cells()
        if k > len(picks):
            raise LayoutError(f"layout only has {len(picks)} pick locations")
        idx = rng.choice(len(picks), size=k, replace=False)
        return self.instance([picks[i] for i in idx])


@dataclass
class Instance:
    """One picking trip: node 0 is the depot, nodes 1..k are the items.

    Attributes:
        cells: Cell of each node.
        D: ``(k + 1, k + 1)`` matrix of walking distances between nodes.
        hilbert: Hilbert index of each node.
        hilbert_n: Number of cells on the Hilbert square (to normalise indices).
    """

    cells: list[Cell]
    D: np.ndarray
    hilbert: np.ndarray
    hilbert_n: float

    @property
    def k(self) -> int:
        """Number of items to collect."""
        return len(self.cells) - 1

    def length(self, order: list[int]) -> float:
        """Return the round-trip length ``depot -> order... -> depot``."""
        total = 0.0
        prev = 0
        for node in order:
            total += self.D[prev, node]
            prev = node
        return float(total + self.D[prev, 0])
