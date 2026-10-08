"""Build a source-only QA archive for an isolated existing runtime container."""

import argparse
import tarfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
output = parser.parse_args().output.resolve()
output.parent.mkdir(parents=True, exist_ok=True)
directories = [
    "backend/app",
    "backend/packages",
    "backend/tests",
    "backend/scripts",
    "backend/evals",
    "backend/docs",
    "nir_core",
    "scripts",
    "contracts",
    "docs",
    "tests",
    "skills/public",
    ".github",
    "docker",
    "frontend/tests",
    ".agent/skills/smoke-test",
]
excluded = {
    "__pycache__",
    ".venv",
    ".pytest_cache",
    ".test-cache",
    "node_modules",
    ".git",
    ".deer-flow",
}
with tarfile.open(output, "w") as archive:
    for directory in directories:
        base = root / directory
        if not base.exists():
            continue
        for path in base.rglob("*"):
            relative = path.relative_to(root)
            if any(
                part in excluded or part.startswith("pytest-cache-")
                for part in relative.parts
            ):
                continue
            if path.is_file():
                archive.add(path, arcname=str(relative), recursive=False)
    for path in [
        *root.glob("*.yaml"),
        *root.glob("*.json"),
        root / "backend/pyproject.toml",
        root / "backend/uv.lock",
        root / "backend/Makefile",
        root / ".gitignore",
        root / "AGENTS.md",
        root / "README.md",
        root / "frontend/package.json",
        root / "frontend/next.config.js",
        root / "backend/Dockerfile",
        root / "backend/langgraph.json",
        root / "Makefile",
        root / "frontend/src/core/api/api-client.ts",
        root / "skills/custom/nir-coordinator/SKILL.md",
        root / "skills/custom/nir-io/SKILL.md",
    ]:
        if path.exists() and path.name not in {"config.yaml", "extensions_config.json"}:
            archive.add(path, arcname=str(path.relative_to(root)), recursive=False)
print(output)
