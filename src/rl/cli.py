"""Command line: train the routing agent, benchmark it, run the experiments.

python -m src.rl.cli train
python -m src.rl.cli benchmark
python -m src.rl.cli ablation
python -m src.rl.cli replay
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence

import numpy as np

from src.rl.agent import FEATURE_NAMES, Agent, default_weights_path
from src.rl.benchmark import (
    HILBERT_FEATURES,
    ablation,
    benchmark,
    format_ablation,
    format_benchmark,
    format_latency,
    latency,
)
from src.rl.navigation import compare_replay, format_replay_report
from src.rl.training import train


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with its five sub-commands."""
    p = argparse.ArgumentParser(prog="hilbert-rl")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="train the routing agent and save its weights")
    t.add_argument("--episodes", type=int, default=1500)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument(
        "--no-hilbert-features",
        action="store_true",
        help="hold the Hilbert features at zero weight (see the ablation)",
    )
    b = sub.add_parser("benchmark", help="compare all methods on held-out warehouses")
    b.add_argument("--instances", type=int, default=40)
    b.add_argument("--json", action="store_true", help="print raw JSON")
    lt = sub.add_parser("latency", help="mean milliseconds per solve for each method")
    lt.add_argument("--instances", type=int, default=20)
    a = sub.add_parser("ablation", help="train with vs without the Hilbert features")
    a.add_argument("--seeds", type=int, default=5)
    r = sub.add_parser("replay", help="uniform vs Hilbert-indexed replay on navigation")
    r.add_argument("--seeds", type=int, default=5)
    r.add_argument("--episodes", type=int, default=150)
    return p


def main(argv: Sequence[str] | None = None) -> None:
    """Run the command line interface."""
    args = build_parser().parse_args(argv)
    if args.cmd == "train":
        t0 = time.time()
        disable = HILBERT_FEATURES if args.no_hilbert_features else ()
        agent, history = train(episodes=args.episodes, seed=args.seed, disable=disable)
        meta = {"episodes": args.episodes, "seed": args.seed, "history": history}
        path = default_weights_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        agent.save(path, meta=meta)
        print(f"saved {path} in {time.time() - t0:.0f}s")
        print(
            "weights:",
            dict(zip(FEATURE_NAMES, np.round(agent.theta, 2).tolist(), strict=True)),
        )
    elif args.cmd == "benchmark":
        res = benchmark(Agent.load(), n_instances=args.instances)
        print(json.dumps(res, indent=2) if args.json else format_benchmark(res))
    elif args.cmd == "latency":
        print(format_latency(latency(Agent.load(), n_instances=args.instances)))
    elif args.cmd == "ablation":
        print(format_ablation(ablation(seeds=args.seeds), args.seeds))
    elif args.cmd == "replay":
        report = compare_replay(seeds=args.seeds, episodes=args.episodes)
        print(format_replay_report(report))


if __name__ == "__main__":
    main()
