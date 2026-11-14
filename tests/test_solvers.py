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


def test_two_opt_never_lengthens_its_start(wh: Warehouse, seed: int) -> None:
    """2-opt returns a permutation no longer than the route it started from."""
    for offset in range(6):
        inst = wh.random_instance(11, np.random.default_rng(seed + offset))
        start = S.snake_order(inst)
        improved = S.two_opt(inst, start)
        assert sorted(improved) == list(range(1, 12))
        assert inst.length(improved) <= inst.length(start) + 1e-9


def test_best_heuristic_beats_or_ties_nearest_neighbor(
    wh: Warehouse, seed: int
) -> None:
    """The combined heuristic is at least as good as plain nearest neighbour."""
    for offset in range(6):
        inst = wh.random_instance(12, np.random.default_rng(seed + offset))
        best = inst.length(S.best_heuristic(inst))
        assert best <= inst.length(S.nearest_neighbor(inst)) + 1e-9


def test_best_heuristic_handles_empty_and_single_item(wh: Warehouse, seed: int) -> None:
    """Degenerate orders are handled without error."""
    assert S.best_heuristic(wh.instance([])) == []
    one = wh.random_instance(1, np.random.default_rng(seed))
    assert S.best_heuristic(one) == [1]


def test_held_karp_matches_brute_force(wh: Warehouse, seed: int) -> None:
    """The DP optimum equals the exhaustive optimum on small instances."""
    for offset in range(6):
        inst = wh.random_instance(6, np.random.default_rng(seed + offset))
        dp = inst.length(S.held_karp(inst))
        assert dp == pytest.approx(inst.length(S.brute_force(inst)))


def test_no_method_beats_the_optimum(wh: Warehouse, seed: int) -> None:
    """Every heuristic is at least as long as the exact optimum."""
    for offset in range(8):
        inst = wh.random_instance(10, np.random.default_rng(seed + offset))
        opt = inst.length(S.held_karp(inst))
        for fn in (
            S.snake_order,
            S.nearest_neighbor,
            S.hilbert_order,
            S.best_heuristic,
        ):
            assert inst.length(fn(inst)) >= opt - 1e-9


def test_held_karp_returns_a_permutation(wh: Warehouse, seed: int) -> None:
    """The exact route visits each item once."""
    inst = wh.random_instance(9, np.random.default_rng(seed))
    assert sorted(S.held_karp(inst)) == list(range(1, 10))


def test_held_karp_edge_cases(wh: Warehouse, seed: int) -> None:
    """Empty and one-item orders work; too many items raise ``ValueError``."""
    assert S.held_karp(wh.instance([])) == []
    one = wh.random_instance(1, np.random.default_rng(seed))
    assert S.held_karp(one) == [1]
    assert S.brute_force(wh.instance([])) == []
    big = wh.random_instance(S.HELD_KARP_MAX_K + 1, np.random.default_rng(seed))
    with pytest.raises(ValueError):
        S.held_karp(big)
