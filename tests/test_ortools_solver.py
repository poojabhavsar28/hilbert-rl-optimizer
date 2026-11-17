"""OR-Tools baseline: valid tours, optimal on small orders, graceful when missing."""

import builtins

import numpy as np
import pytest

from src.environment import solvers as S
from src.environment.ortools_solver import (
    OrToolsUnavailable,
    ortools_available,
    ortools_order,
)
from src.environment.warehouse import Warehouse

pytest.importorskip("ortools")


@pytest.fixture(scope="module")
def wh():
    return Warehouse.generate(5, 16)


@pytest.mark.parametrize("k", [1, 2, 8, 20])
def test_returns_a_permutation_of_all_items(wh, k):
    inst = wh.random_instance(k, np.random.default_rng(k))
    assert sorted(ortools_order(inst)) == list(range(1, k + 1))


def test_matches_the_exact_optimum_on_small_orders(wh):
    rng = np.random.default_rng(0)
    for _ in range(5):
        inst = wh.random_instance(8, rng)
        assert inst.length(ortools_order(inst)) == pytest.approx(
            inst.length(S.held_karp(inst))
        )


def test_never_worse_than_s_shape_on_large_orders(wh):
    rng = np.random.default_rng(3)
    for _ in range(3):
        inst = wh.random_instance(25, rng)
        assert inst.length(ortools_order(inst)) <= inst.length(S.snake_order(inst))


def test_single_item_order_is_returned_as_is(wh):
    inst = wh.random_instance(1, np.random.default_rng(0))
    assert ortools_order(inst) == [1]


def test_available_flag_is_true_when_installed():
    assert ortools_available() is True


def test_raises_a_clear_error_when_ortools_is_missing(wh, monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("ortools"):
            raise ImportError("no ortools")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    inst = wh.random_instance(5, np.random.default_rng(0))
    assert ortools_available() is False
    with pytest.raises(OrToolsUnavailable, match="pip install ortools"):
        ortools_order(inst)
