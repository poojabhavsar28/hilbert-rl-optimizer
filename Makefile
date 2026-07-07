# ── Hilbert-RL Optimizer — Developer Makefile ────────────────────────────────
# Usage:
#   make setup    → create venv and install all dependencies
#   make test     → run full pytest suite with coverage
#   make lint     → run ruff linter and formatter check
#   make check    → lint + test (run before every PR)
#   make clean    → remove all cache and build artifacts
#   make run-api  → start FastAPI dev server
#   make run-app  → start Streamlit dashboard

.PHONY: setup test lint format check clean run-api run-app docker-up docker-down help

# ── Setup ─────────────────────────────────────────────────────────────────────
setup:
	@echo "→ Creating virtual environment with Python 3.11..."
	uv venv --python 3.11
	@echo "→ Installing all dependencies..."
	uv pip install -e ".[dev]"
	@echo "→ Installing pre-commit hooks..."
	pre-commit install
	@echo "✓ Setup complete. Run: source .venv/bin/activate"

# ── Testing ───────────────────────────────────────────────────────────────────
test:
	@echo "→ Running test suite..."
	pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=xml

test-fast:
	@echo "→ Running tests (no coverage, faster)..."
	pytest tests/ -v -x

# ── Linting & Formatting ──────────────────────────────────────────────────────
lint:
	@echo "→ Running ruff linter..."
	ruff check src/ tests/ scripts/
	@echo "→ Checking code format..."
	ruff format --check src/ tests/ scripts/

format:
	@echo "→ Auto-formatting all code..."
	ruff format src/ tests/ scripts/
	ruff check --fix src/ tests/ scripts/

# ── Full CI check (run this before every PR) ──────────────────────────────────
check: lint test
	@echo "✓ All checks passed — ready to push"

# ── Running services ──────────────────────────────────────────────────────────
run-api:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

run-app:
	streamlit run app/main.py

# ── Docker ────────────────────────────────────────────────────────────────────
docker-up:
	docker compose up --build

docker-down:
	docker compose down

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	@echo "→ Removing Python cache..."
	find . -type d -name "__pycache__" | xargs rm -rf
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete
	find . -name ".coverage" -delete
	find . -name "coverage.xml" -delete
	find . -type d -name "*.egg-info" | xargs rm -rf
	find . -type d -name ".pytest_cache" | xargs rm -rf
	find . -type d -name ".mypy_cache" | xargs rm -rf
	find . -type d -name ".ruff_cache" | xargs rm -rf
	@echo "✓ Clean complete"

# ── Help ──────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "Hilbert-RL Optimizer — Available Commands"
	@echo "────────────────────────────────────────────"
	@echo "  make setup      Create venv and install dependencies"
	@echo "  make test       Run pytest with coverage"
	@echo "  make lint       Run ruff linter + format check"
	@echo "  make format     Auto-fix formatting issues"
	@echo "  make check      Run lint + test (pre-PR check)"
	@echo "  make run-api    Start FastAPI on port 8000"
	@echo "  make run-app    Start Streamlit dashboard"
	@echo "  make docker-up  Start all services with Docker Compose"
	@echo "  make clean      Remove cache and build artifacts"
	@echo ""
