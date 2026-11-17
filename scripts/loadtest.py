"""Minimal load test for a running API (needs ``httpx``).

Start the server first, e.g. ``uvicorn src.api.main:app --port 8000``, then::

    python scripts/loadtest.py --url http://127.0.0.1:8000 --concurrency 8 --seconds 20

It drives a realistic mix (route one order with all methods, and a 20-order bulk
optimisation) from ``--concurrency`` concurrent clients and prints throughput and
latency percentiles per endpoint. Pass ``--api-key`` for a keyed server.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import httpx

SOLVE_METHODS = ["snake", "nearest_neighbor", "rl_greedy", "two_opt"]
N_BULK_ORDERS = 20


async def worker(
    client: httpx.AsyncClient, deadline: float, mix: list[tuple[str, dict]], out: dict
) -> None:
    """Send requests round-robin from ``mix`` until the deadline."""
    i = 0
    while time.perf_counter() < deadline:
        path, body = mix[i % len(mix)]
        i += 1
        t0 = time.perf_counter()
        try:
            res = await client.post(path, json=body)
            status = res.status_code
        except httpx.HTTPError:
            status = 0
        out.setdefault(path, []).append((time.perf_counter() - t0, status))


def report(out: dict, elapsed: float) -> str:
    """Format throughput and latency percentiles per endpoint."""
    lines = [
        f"{'endpoint':<20}{'requests':>9}{'req/s':>8}{'p50 ms':>9}{'p95 ms':>9}{'p99 ms':>9}{'errors':>8}"
    ]
    for path, rows in sorted(out.items()):
        lat = sorted(s * 1000 for s, _ in rows)
        errors = sum(1 for _, st in rows if st != 200)
        q = statistics.quantiles(lat, n=100) if len(lat) >= 2 else lat * 99
        lines.append(
            f"{path:<20}{len(rows):>9}{len(rows) / elapsed:>8.1f}"
            f"{q[49]:>9.1f}{q[94]:>9.1f}{q[98]:>9.1f}{errors:>8}"
        )
    total = sum(len(r) for r in out.values())
    lines.append(f"{'total':<20}{total:>9}{total / elapsed:>8.1f}")
    return "\n".join(lines)


async def main_async(args: argparse.Namespace) -> None:
    """Run the load test and print the report."""
    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    out: dict[str, list[tuple[float, int]]] = {}
    limits = httpx.Limits(max_connections=args.concurrency * 2)
    async with httpx.AsyncClient(
        base_url=args.url, headers=headers, timeout=60, limits=limits
    ) as client:
        health = await client.get("/api/health")
        health.raise_for_status()
        orders = []
        for seed in range(N_BULK_ORDERS):
            res = await client.post(
                "/api/instance", json={"aisles": 8, "height": 22, "k": 12, "seed": seed}
            )
            res.raise_for_status()
            orders.append({"id": f"o{seed}", "items": res.json()["items"]})
        bulk = {"aisles": 8, "height": 22, "orders": orders, "method": "rl_greedy"}
        solve = {
            "aisles": 8,
            "height": 22,
            "items": orders[0]["items"][:8],
            "methods": SOLVE_METHODS,
        }
        mix = [("/api/solve", solve), ("/api/optimize", bulk)]
        start = time.perf_counter()
        deadline = start + args.seconds
        await asyncio.gather(
            *(worker(client, deadline, mix, out) for _ in range(args.concurrency))
        )
        elapsed = time.perf_counter() - start
    print(f"{args.concurrency} concurrent clients, {args.seconds}s, {args.url}")
    print(report(out, elapsed))


def main() -> None:
    """Parse arguments and run."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--seconds", type=float, default=20)
    p.add_argument("--api-key", default=None)
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
