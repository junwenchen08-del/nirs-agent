from __future__ import annotations

import json
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_example_enables_local_chemotools_stdio_server():
    config_path = REPO_ROOT / "extensions_config.example.json"
    if not config_path.is_file():
        pytest.skip("repository-root extension template is not mounted in this container")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    server = config["mcpServers"]["chemotools"]

    assert server["enabled"] is True
    assert server["type"] == "stdio"
    assert server["command"] == "python"
    assert server["args"] == [
        "-m",
        "nir_core.chemotools_mcp.chemotools_server",
    ]


def test_backend_installs_nir_core_mcp_extra():
    backend_project = tomllib.loads((REPO_ROOT / "backend" / "pyproject.toml").read_text(encoding="utf-8"))

    assert "nir-core[deep,mat73,mcp]" in backend_project["project"]["dependencies"]
