# Operating the API

Everything here is optional. With no configuration the API is open, which is right
for local use and a public demo.

## Configuration

| Variable | Default | Effect |
|---|---|---|
| `HILBERT_API_KEY` | unset | When set, `/api/*` (except `/api/health`) and `/metrics` need a matching `X-API-Key` header, otherwise 401. The web UI has an API key field for this. |
| `HILBERT_RATE_LIMIT_PER_MIN` | `0` (off) | Requests per client per minute on `/api/*` (except health), sliding window. Over the limit: 429 with `Retry-After`. |
| `HILBERT_RL_WEIGHTS` | `models/routing_agent.json` | Path of the trained routing agent. |

```bash
HILBERT_API_KEY=change-me HILBERT_RATE_LIMIT_PER_MIN=120 uvicorn src.api.main:app
curl -H "X-API-Key: change-me" localhost:8000/api/sample-orders.csv
```

In Docker, pass them with `-e` or the `environment:` section of `docker-compose.yml`.

Notes and limits:

- The key is compared in constant time, but it is one shared secret over plain HTTP. Put TLS in front (a reverse proxy) before exposing it.
- The rate limit identifies clients by socket address and keeps at most 10,000 clients in memory. Behind a proxy every request appears to come from the proxy, and `X-Forwarded-For` is deliberately not trusted because callers can forge it; rate-limit at the proxy in that case.
- The rate limit runs before the key check, so guessing keys counts against the budget.
- Limits are per process. With several workers each has its own budget and counters.

## Observability

- **Logs:** one JSON line per request with a correlation id (`X-Request-ID`, accepted from the caller if it is a simple token), method, path, status and duration.
- **Metrics:** `GET /metrics` returns Prometheus text: `http_requests_total{method,path,status}` and the `http_request_duration_seconds` histogram. The path label is the route template, and every unmatched URL shares one label, so scanners cannot inflate memory. Counters are per process and reset on restart.
- **Health:** `GET /api/health` is always open (the Docker healthcheck uses it).

## Load testing

```bash
uvicorn src.api.main:app --port 8000 &
python scripts/loadtest.py --url http://127.0.0.1:8000 --concurrency 8 --seconds 20
```

It mixes single-order solves (four methods) and 20-order bulk optimisations and prints requests per second and p50, p95 and p99 latency per endpoint. Results depend heavily on the machine; see [results/loadtest.txt](results/loadtest.txt) for one recorded run and the hardware it ran on.

## Type checking

`mypy src` runs in CI. It targets Python 3.12 in `pyproject.toml` because NumPy's type stubs use syntax older targets cannot parse; the code still supports 3.10 and newer.
