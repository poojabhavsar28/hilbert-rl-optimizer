# Changelog

All notable changes to this project are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Phase 1: Repository foundation, dev environment, CI/CD pipeline
- Phase 2: 2D Hilbert curve encoder/decoder (`src/hilbert`) with vectorised batch
  encoding, `spatial_sort`, unit-square encoder, rectangular-grid clipping, a
  Morton (Z-order) baseline and a locality metric, with tests

- Phase 3: warehouse model (`src/environment`) with ASCII layouts, BFS walking
  distances and paths, routing instances, and solvers: S-shape, nearest
  neighbour, Hilbert sort, 2-opt and exact Held-Karp, with tests

- Phase 4: REINFORCE routing agent and trainer, uniform and Hilbert-indexed replay
  buffers, tabular Q-learning testbed, benchmark/ablation tooling, CLI
  (`python -m src.rl.cli`), trained weights in `models/`, and recorded results in
  `docs/results/`

- Phase 5: FastAPI service (`src/api`): health, layout, instance, solve and bulk
  optimise (JSON and CSV upload) endpoints, a service layer, request
  correlation IDs with JSON access logs, CSV formula-injection protection, and
  `python-multipart` as a dependency
- Phase 6: single-page web UI served by the API at `/` (route comparison, bulk
  CSV optimisation, sample orders endpoint `GET /api/sample-orders.csv`),
  `Dockerfile`, `docker-compose.yml`, `requirements-api.txt` and a CI job that
  builds the image and smoke-tests it
- Phase 7: Google OR-Tools solver (`src/environment/ortools_solver.py`, optional
  `solvers` extra) as a routing method in the benchmark, API and UI; a latency
  measurement (`make latency`); benchmark reference above 12 items is now the
  best known route. Result: OR-Tools is optimal up to 12 items and 3 to 6 points
  shorter than the learned agent at 20 to 30 items

### Fixed
- `pip install -e .` failed because hatchling could not locate the `src` package
- Import order and formatting in the Phase 1 test files (ruff)

---

## [0.1.0] — Phase 1 Foundation

### Added
- Project structure with `src/`, `tests/`, `scripts/`, `notebooks/`, `configs/`
- `pyproject.toml` with all dependencies and tool configuration
- `Makefile` with `setup`, `test`, `lint`, `check`, `clean` targets
- `pre-commit` hooks for ruff linting and formatting
- GitHub Actions CI pipeline running on Python 3.10 and 3.11
- PR template and issue templates (bug report, feature request)
- MIT License, README with architecture overview
- Branch protection strategy documented in CONTRIBUTING.md
