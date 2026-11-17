"""In-memory request metrics in the Prometheus text exposition format.

No client library is needed. Labels are bounded on purpose: the path label is
the matched route template (``/api/solve``), never the raw URL, and unmatched
requests share one label, so random URLs cannot grow memory without limit.
"""

from __future__ import annotations

import threading
from collections import defaultdict

BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
UNMATCHED = "unmatched"


class Metrics:
    """Thread-safe request counters and a latency histogram."""

    def __init__(self) -> None:
        """Create empty metrics."""
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        """Forget everything recorded so far (used by tests)."""
        with self._lock:
            self._count: dict[tuple[str, str, int], int] = defaultdict(int)
            self._bucket: dict[str, list[int]] = defaultdict(lambda: [0] * len(BUCKETS))
            self._sum: dict[str, float] = defaultdict(float)
            self._total: dict[str, int] = defaultdict(int)

    def observe(self, method: str, path: str, status: int, seconds: float) -> None:
        """Record one finished request."""
        with self._lock:
            self._count[(method, path, status)] += 1
            self._sum[path] += seconds
            self._total[path] += 1
            counts = self._bucket[path]
            for i, bound in enumerate(BUCKETS):
                if seconds <= bound:
                    counts[i] += 1

    def render(self) -> str:
        """Return the metrics as Prometheus text."""
        with self._lock:
            lines = [
                "# HELP http_requests_total Requests by method, route and status.",
                "# TYPE http_requests_total counter",
            ]
            for (method, path, status), n in sorted(self._count.items()):
                lines.append(
                    f'http_requests_total{{method="{method}",path="{path}",'
                    f'status="{status}"}} {n}'
                )
            lines += [
                "# HELP http_request_duration_seconds Request latency by route.",
                "# TYPE http_request_duration_seconds histogram",
            ]
            for path in sorted(self._total):
                for bound, n in zip(BUCKETS, self._bucket[path], strict=True):
                    lines.append(
                        f'http_request_duration_seconds_bucket{{path="{path}",'
                        f'le="{bound}"}} {n}'
                    )
                lines.append(
                    f'http_request_duration_seconds_bucket{{path="{path}",'
                    f'le="+Inf"}} {self._total[path]}'
                )
                lines.append(
                    f'http_request_duration_seconds_sum{{path="{path}"}} '
                    f"{self._sum[path]:.6f}"
                )
                lines.append(
                    f'http_request_duration_seconds_count{{path="{path}"}} '
                    f"{self._total[path]}"
                )
        return "\n".join(lines) + "\n"


METRICS = Metrics()
