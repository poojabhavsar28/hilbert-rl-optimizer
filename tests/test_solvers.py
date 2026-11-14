"""Tests for the routing heuristics and the exact solver."""

import numpy as np
import pytest

from src.environment import solvers as S
from src.environment.warehouse import Warehouse


@pytest.fixture(scope="module")
def wh() -> Warehouse:
    """A small generated warehouse shared by the solver tests."""
    return Warehouse.generate(6, 14)


@pytest.mark.parametrize("fn", [S.snake_order, S.nearest_neighbor, S.hilbert_order])
def test_baselines_return_permutations(wh: Warehouse, fn, seed: int) -> None:
    """Each baseline visits every item exactly once."""
    inst = wh.random_instance(9, np.random.default_rng(seed))
    assert sorted(fn(inst)) == list(range(1, 10))


def test_snake_groups_items_by_aisle(wh: Warehouse, seed: int) -> None:
    """The S-shape sweep finishes one aisle column before starting the next."""
    inst = wh.random_instance(12, np.random.default_rng(seed))
    columns = [inst.cells[n][0] for n in S.snake_order(inst)]
    seen: list[int] = []
    for c in columns:
        if not seen or seen[-1] != c:
            seen.append(c)
    assert len(seen) == len(set(columns))
    assert seen == sorted(seen)


def test_nearest_neighbor_first_stop_is_closest(wh: Warehouse, seed: int) -> None:
    """The first stop is the item nearest to the depot."""
    inst = wh.random_instance(8, np.random.default_rng(seed))
    first = S.nearest_neighbor(inst)[0]
    assert inst.D[0, first] == inst.D[0, 1:].min()


def test_hilbert_order_picks_the_shorter_direction(wh: Warehouse, seed: int) -> None:
    """The returned order is no longer than its own reverse."""
    inst = wh.random_instance(10, np.random.default_rng(seed))
    order = S.hilbert_order(inst)
    assert inst.length(order) <= inst.length(order[::-1])


@pytest.mark.parametrize("fn", [S.snake_order, S.nearest_neighbor, S.hilbert_order])
def test_baselines_handle_no_items(wh: Warehouse, fn) -> None:
    """An empty order produces an empty route."""
    assert fn(wh.instance([])) == []
