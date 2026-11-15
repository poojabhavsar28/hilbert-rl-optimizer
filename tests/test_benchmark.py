"""Tests for the benchmark and ablation utilities (kept tiny for speed)."""

import numpy as np

from src.rl.agent import Agent
from src.rl.benchmark import (
    HILBERT_FEATURES,
    ablation,
    benchmark,
    format_ablation,
    format_benchmark,
)
from src.rl.training import train


def test_benchmark_structure_and_references() -> None:
    """Rows report every method; small orders are measured against the optimum."""
    agent, _ = train(episodes=30, seed=0, verbose=False)
    res = benchmark(agent, n_instances=3, ks=(6, 14))
    assert set(res) == {"6", "14"}
    assert res["6"]["_reference"] == "optimal" and "optimal" in res["6"]
    assert res["14"]["_reference"] == "two_opt" and "optimal" not in res["14"]
    for row in res.values():
        pct = row["_pct_over_reference"]
        assert set(pct) >= {"snake", "nearest_neighbor", "rl_greedy", "two_opt"}
        assert 0.0 <= row["_rl_beats_nearest_neighbor_pct_of_instances"] <= 100.0
    assert res["6"]["_pct_over_reference"]["snake"] >= -1e-9  # nothing beats optimal


def test_benchmark_is_reproducible() -> None:
    """Fixed seeds give identical results."""
    agent = Agent(theta=np.array([-3.0, 1.0, 0.0, -1.0, 2.0, 0.5]))
    a = benchmark(agent, n_instances=2, ks=(6,))
    b = benchmark(agent, n_instances=2, ks=(6,))
    assert a == b


def test_format_benchmark_lists_methods() -> None:
    """The text table names each method and the reference."""
    agent = Agent(theta=np.array([-3.0, 1.0, 0.0, -1.0, 2.0, 0.5]))
    text = format_benchmark(benchmark(agent, n_instances=2, ks=(6, 14)))
    assert "rl_greedy" in text and "reference = optimal" in text


def test_ablation_compares_both_variants() -> None:
    """The ablation reports numbers for runs with and without Hilbert features."""
    res = ablation(seeds=1, episodes=20, n_instances=2, ks=(6,))
    assert set(res) == {"with_hilbert_features", "without_hilbert_features"}
    for v in res.values():
        assert len(v["greedy_per_seed"]) == 1
    text = format_ablation(res, seeds=1)
    assert "with_hilbert_features" in text
    assert len(HILBERT_FEATURES) == 2
