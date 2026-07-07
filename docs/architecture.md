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

### `src/environment/`
- Gymnasium-compatible warehouse simulation
- Agent must pick items and deliver to dispatch station
- State space uses Hilbert indices (the core innovation)
- Shaped reward with 6 components for efficient RL learning

### `src/rl/`
- PPO and SAC agents via Stable-Baselines3
- **Hilbert Replay Buffer**: samples spatially-local experiences
- MLflow experiment tracking
- Training callbacks: checkpoint, early stopping, video recording

### `src/api/`
- FastAPI REST backend
- ONNX model for <50ms inference
- Pydantic v2 request/response validation
- Structured JSON logging with correlation IDs

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
