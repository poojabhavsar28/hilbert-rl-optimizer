"""Tests for the command line interface."""

import json

import pytest

from src.rl.agent import WEIGHTS_ENV_VAR, Agent
from src.rl.cli import build_parser, main


def test_train_writes_loadable_weights(monkeypatch, tmp_path, capsys) -> None:
    """``train`` creates the weights file (and its folder) and reports it."""
    path = tmp_path / "nested" / "agent.json"
    monkeypatch.setenv(WEIGHTS_ENV_VAR, str(path))
    main(["train", "--episodes", "20", "--seed", "1"])
    assert "saved" in capsys.readouterr().out
    assert Agent.load(path).theta.shape == (6,)
    assert json.loads(path.read_text())["meta"]["episodes"] == 20


def test_train_without_hilbert_features_zeroes_them(monkeypatch, tmp_path) -> None:
    """The ``--no-hilbert-features`` flag leaves those two weights at zero."""
    path = tmp_path / "agent.json"
    monkeypatch.setenv(WEIGHTS_ENV_VAR, str(path))
    main(["train", "--episodes", "20", "--no-hilbert-features"])
    theta = Agent.load(path).theta
    assert theta[2] == 0.0 and theta[3] == 0.0


def test_benchmark_text_and_json(monkeypatch, tmp_path, capsys) -> None:
    """``benchmark`` prints a table, or JSON with ``--json``."""
    path = tmp_path / "agent.json"
    monkeypatch.setenv(WEIGHTS_ENV_VAR, str(path))
    main(["train", "--episodes", "20"])
    capsys.readouterr()
    main(["benchmark", "--instances", "1"])
    assert "rl_greedy" in capsys.readouterr().out
    main(["benchmark", "--instances", "1", "--json"])
    assert "_reference" in json.loads(capsys.readouterr().out)["8"]


def test_replay_command_prints_report(capsys) -> None:
    """``replay`` prints a comparison of the three strategies."""
    main(["replay", "--seeds", "1", "--episodes", "6"])
    out = capsys.readouterr().out
    assert "uniform" in out and "stratified" in out and "local" in out


def test_parser_requires_a_command() -> None:
    """Running with no sub-command is an error."""
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
