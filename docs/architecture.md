# Architecture

## System Overview

The Hilbert-RL Optimizer is structured in layers. Each layer can be developed,
tested, and deployed independently.

```
┌─────────────────────────────────────────────────────────┐
│                    Data / Environment                    │
│   Raw coordinates → Hilbert encoder → 1D index          │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│               Warehouse Gymnasium Environment            │
│   10 actions · shaped reward · Hilbert observation       │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│                    RL Training Layer                     │
│   PPO / SAC · Hilbert Replay Buffer · MLflow tracking   │
└───────────┬───────────────────────────────┬─────────────┘
            │                               │
┌───────────▼────────┐         ┌────────────▼────────────┐
│   FastAPI Backend  │         │   Streamlit Dashboard   │
│   ONNX inference   │         │   Live visualization    │
│   REST API         │         │   Interactive demo      │
└───────────┬────────┘         └────────────┬────────────┘
            │                               │
┌───────────▼───────────────────────────────▼────────────┐
│                  NGINX Reverse Proxy                    │
│         Docker Compose · Cloud deployment               │
└─────────────────────────────────────────────────────────┘
```

## Module Responsibilities

### `src/hilbert/`
- Encodes N-dimensional coordinates to a 1D Hilbert index
- Preserves spatial locality: nearby points → nearby indices
- Supports 2D, 3D, and arbitrary N dimensions
- Adaptive resolution for non-uniform data density

**Status (Phase 2):** 2D only. Implemented: `xy2d`/`d2xy`, vectorised
`xy2d_batch`, `spatial_sort`, `encode_unit` (unit square to a normalised index),
`curve_segments` (clip to rectangles), a Morton (Z-order) baseline and a
locality metric. 3D / N-D encoding and adaptive resolution are not implemented
yet.

Measured locality (mean / worst grid distance between consecutive indices, 32x32):
Hilbert 1.00 / 1, Z-order 1.94 / 32. See `tests/test_locality.py`.

### `src/environment/`
- Gymnasium-compatible warehouse simulation
- Agent must pick items and deliver to dispatch station
- State space uses Hilbert indices (the core innovation)
- Shaped reward with 6 components for efficient RL learning

**Status (Phase 3):** this phase provides the *routing* model, not yet a
Gymnasium environment: block-and-aisle warehouses from ASCII maps, exact BFS
walking distances around shelves, routing instances (distance matrix plus
Hilbert indices) and solvers (S-shape, nearest neighbour, Hilbert sort, 2-opt,
exact Held-Karp up to 12 items). There is no `step`/`reset` loop, 10-action
space or shaped reward yet.

### `src/rl/`
- PPO and SAC agents via Stable-Baselines3
- **Hilbert Replay Buffer**: samples spatially-local experiences
- MLflow experiment tracking
- Training callbacks: checkpoint, early stopping, video recording

**Status (Phase 4):** Stable-Baselines3 PPO/SAC, MLflow tracking and training
callbacks are **not implemented yet**. This phase delivers a NumPy-only
REINFORCE routing agent (`src/rl/agent.py`, `training.py`), the Hilbert-indexed
replay buffer with `stratified` and `local` sampling (`replay.py`), a tabular
Q-learning testbed for it (`navigation.py`), and reproducible experiments
(`benchmark.py`, `cli.py`, results in `docs/results/`).

What the experiments found (see `docs/results/`):

- The agent beats nearest neighbour by 5 to 8 points of route length on
  held-out warehouses and matches 2-opt on short orders, but 2-opt is better at
  30 items.
- Hilbert-curve *sorting* is the worst routing method tested (+33% to +54%).
- The agent does slightly **better without** the Hilbert features (ablation).
- The Hilbert-indexed replay buffer performs the same as uniform replay in the
  tabular testbed (20 seeds). Its stratified-sampling property is verified in
  `tests/test_replay.py`, but it has not been shown to improve learning. Testing
  it where it should matter (large state spaces, function approximation, skewed
  visitation) is future work.

### `src/api/`
- FastAPI REST backend
- ONNX model for <50ms inference
- Pydantic v2 request/response validation
- Structured JSON logging with correlation IDs

**Status (Phase 5):** implemented: FastAPI app (`src/api/main.py`), thin routes
(`routes/`), pydantic v2 schemas, a service layer (`services.py`), request
correlation IDs (`X-Request-ID`) with one JSON access-log line per request, and
request limits (`config.py`). **Not implemented:** ONNX inference (there is no
torch model; the agent is a 6-weight linear policy evaluated in NumPy).

| Endpoint | Purpose |
|---|---|
| `GET /api/health` | status, whether weights loaded, available methods |
| `POST /api/layout` | generate/parse a layout; pick cells and clipped Hilbert curve |
| `POST /api/instance` | reproducible random order |
| `POST /api/solve` | route one order with several methods |
| `POST /api/optimize` | JSON bulk optimisation |
| `POST /api/optimize/csv` | CSV in, CSV (or JSON) out |

Limits: layouts up to 12,000 cells, 60 items per trip, 200 orders per request,
1 MB CSV. Warehouses are not cached between requests because each memoises BFS
maps and would grow without bound.

**Status (Phase 6):** the API serves a single-page web UI (`src/api/static/`)
at `/`, and ships as a Docker image (`Dockerfile`, `docker-compose.yml`) built
from `requirements-api.txt`, which excludes torch and the other training-only
dependencies. CI builds the image and smoke-tests it. **Not implemented:** the
Streamlit dashboard below and cloud deployment.

### `app/`
- Streamlit interactive dashboard
- Live agent simulation
- Hilbert curve explorer
- Benchmark comparison charts

## Key Design Decisions

**Why Hilbert over Z-order (Morton)?**
Hilbert curves have strictly better locality preservation.
Z-order curves have discontinuities at quadrant boundaries —
two adjacent Z-order indices can be far apart in 2D space.
Hilbert curves eliminate these discontinuities entirely.

**Why Gymnasium over custom environment?**
Gymnasium compatibility means the environment works with
any standard RL library — SB3, RLlib, CleanRL, tianshou —
with zero modification. This is production engineering practice.

**Why ONNX for serving?**
ONNX runtime is 5–10x faster than PyTorch for inference.
It also removes the PyTorch dependency from the API server,
making the Docker image significantly smaller.
