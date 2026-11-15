"""Guard the Docker image: its dependency list must cover everything src/ imports."""

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Provided by fastapi itself, so not listed separately.
TRANSITIVE = {"starlette"}


def third_party_imports() -> set[str]:
    """Return top-level third-party modules imported anywhere under ``src/``."""
    found: set[str] = set()
    for path in (ROOT / "src").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                found.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    stdlib = set(sys.stdlib_module_names)
    return {m for m in found if m not in stdlib and m != "src"}


def listed_packages() -> set[str]:
    """Return package names in ``requirements-api.txt`` (without extras/versions)."""
    names = set()
    for line in (ROOT / "requirements-api.txt").read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            names.add(re.split(r"[\[<>=!~ ]", line)[0].lower().replace("-", "_"))
    return names


def test_requirements_cover_every_third_party_import() -> None:
    """A missing package would crash the container at start-up."""
    missing = third_party_imports() - listed_packages() - TRANSITIVE
    assert not missing, f"add to requirements-api.txt: {sorted(missing)}"


def test_requirements_include_the_server_and_upload_support() -> None:
    """uvicorn runs the app and python-multipart enables the CSV upload form."""
    assert {"uvicorn", "python_multipart"} <= listed_packages()


def test_dockerfile_ships_code_and_weights() -> None:
    """The image copies the code and the trained weights the API loads."""
    text = (ROOT / "Dockerfile").read_text()
    assert "COPY src ./src" in text and "COPY models ./models" in text
    assert "src.api.main:app" in text
    assert (ROOT / "models" / "routing_agent.json").exists()
