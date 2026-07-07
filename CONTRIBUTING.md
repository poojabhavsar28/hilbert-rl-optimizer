# Contributing to Hilbert-RL Optimizer

Thank you for your interest in contributing!

---

## Development Setup

```bash
# 1. Fork and clone
git clone https://github.com/YOUR_USERNAME/hilbert-rl-optimizer.git
cd hilbert-rl-optimizer

# 2. Set up environment
make setup

# 3. Activate environment
source .venv/bin/activate

# 4. Verify everything works
make check
```

---

## Branch Strategy

```
main      ← stable releases only, never commit directly
dev       ← integration branch
phase/*   ← phase feature branches
fix/*     ← bug fix branches
```

**Always branch from `dev`, never from `main`:**

```bash
git checkout dev
git pull origin dev
git checkout -b fix/your-fix-name
```

---

## Commit Message Format

Use the conventional commits standard:

```
<type>: <short description in present tense>

Types:
  feat     → new feature
  fix      → bug fix
  docs     → documentation only
  test     → adding or updating tests
  refactor → code change without new feature or fix
  perf     → performance improvement
  ci       → CI/CD pipeline changes
  chore    → tooling, dependencies, config
  exp      → experiment or research result
```

**Examples:**

```bash
git commit -m "feat: implement 2D Hilbert encode and decode"
git commit -m "fix: handle edge case when order=1 in encoder"
git commit -m "test: add round-trip encode-decode tests for all orders"
git commit -m "docs: add docstring to spatial_sort utility"
git commit -m "perf: vectorise batch encoding with numpy"
```

---

## Before Opening a Pull Request

```bash
# Run full check — must pass with zero errors
make check
```

Then open a PR into `dev` (never directly into `main`).

---

## Code Style

- All functions must have **type hints** and **docstrings**
- Line length: **88 characters** (enforced by ruff)
- Imports sorted automatically by ruff
- No commented-out code in PRs

---

## Questions?

Open an issue with the label `question`.
