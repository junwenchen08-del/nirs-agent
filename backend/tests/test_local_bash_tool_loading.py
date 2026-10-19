from pathlib import Path
from types import SimpleNamespace

import yaml

from deerflow.sandbox.security import is_host_bash_allowed
from deerflow.tools.tools import get_available_tools


def _make_config(*, allow_host_bash: bool, sandbox_use: str = "deerflow.sandbox.local:LocalSandboxProvider", extra_tools: list[SimpleNamespace] | None = None):
    return SimpleNamespace(
        tools=[
            SimpleNamespace(name="bash", group="bash", use="deerflow.sandbox.tools:bash_tool"),
            SimpleNamespace(name="ls", group="file:read", use="tests:ls_tool"),
            *(extra_tools or []),
        ],
        models=[],
        sandbox=SimpleNamespace(
            use=sandbox_use,
            allow_host_bash=allow_host_bash,
        ),
        tool_search=SimpleNamespace(enabled=False),
        get_model_config=lambda name: None,
    )


def test_get_available_tools_hides_bash_for_default_local_sandbox(monkeypatch):
    monkeypatch.setattr("deerflow.tools.tools.get_app_config", lambda: _make_config(allow_host_bash=False))
    monkeypatch.setattr(
        "deerflow.tools.tools.resolve_variable",
        lambda use, _: SimpleNamespace(name="bash" if "bash" in use else "ls"),
    )

    names = [tool.name for tool in get_available_tools(include_mcp=False, subagent_enabled=False)]

    assert "bash" not in names
    assert "ls" in names


def test_get_available_tools_keeps_bash_when_explicitly_enabled(monkeypatch):
    monkeypatch.setattr("deerflow.tools.tools.get_app_config", lambda: _make_config(allow_host_bash=True))
    monkeypatch.setattr(
        "deerflow.tools.tools.resolve_variable",
        lambda use, _: SimpleNamespace(name="bash" if "bash" in use else "ls"),
    )

    names = [tool.name for tool in get_available_tools(include_mcp=False, subagent_enabled=False)]

    assert "bash" in names
    assert "ls" in names


def test_get_available_tools_hides_renamed_host_bash_alias(monkeypatch):
    config = _make_config(
        allow_host_bash=False,
        extra_tools=[SimpleNamespace(name="shell", group="bash", use="deerflow.sandbox.tools:bash_tool")],
    )
    monkeypatch.setattr("deerflow.tools.tools.get_app_config", lambda: config)
    monkeypatch.setattr(
        "deerflow.tools.tools.resolve_variable",
        lambda use, _: SimpleNamespace(name="bash" if "bash_tool" in use else "ls"),
    )

    names = [tool.name for tool in get_available_tools(include_mcp=False, subagent_enabled=False)]

    assert "bash" not in names
    assert "shell" not in names
    assert "ls" in names


def test_get_available_tools_keeps_nir_tool_misgrouped_as_bash(monkeypatch):
    config = _make_config(
        allow_host_bash=False,
        extra_tools=[SimpleNamespace(name="nir_workflow", group="bash", use="deerflow.community.nir.workflow:nir_workflow_tool")],
    )
    monkeypatch.setattr("deerflow.tools.tools.get_app_config", lambda: config)
    monkeypatch.setattr(
        "deerflow.tools.tools.resolve_variable",
        lambda use, _: SimpleNamespace(name="nir_workflow" if "nir_workflow_tool" in use else "ls"),
    )

    names = [tool.name for tool in get_available_tools(include_mcp=False, subagent_enabled=False)]

    assert "nir_workflow" in names
    assert "bash" not in names


def test_example_config_separates_nir_tools_from_host_bash():
    example_path = Path(__file__).resolve().parents[2] / "config.example.yaml"
    config = yaml.safe_load(example_path.read_text(encoding="utf-8"))
    groups = {group["name"] for group in config["tool_groups"]}
    nir_tools = [tool for tool in config["tools"] if tool["name"].startswith("nir_")]

    assert "nir" in groups
    assert nir_tools
    assert all(tool["group"] == "nir" for tool in nir_tools)
    assert {tool["name"] for tool in config["tools"] if tool["group"] == "bash"} == {"bash"}


def test_get_available_tools_keeps_bash_for_aio_sandbox(monkeypatch):
    config = _make_config(
        allow_host_bash=False,
        sandbox_use="deerflow.community.aio_sandbox:AioSandboxProvider",
    )
    monkeypatch.setattr("deerflow.tools.tools.get_app_config", lambda: config)
    monkeypatch.setattr(
        "deerflow.tools.tools.resolve_variable",
        lambda use, _: SimpleNamespace(name="bash" if "bash_tool" in use else "ls"),
    )

    names = [tool.name for tool in get_available_tools(include_mcp=False, subagent_enabled=False)]

    assert "bash" in names
    assert "ls" in names


def test_is_host_bash_allowed_defaults_false_when_sandbox_missing():
    assert is_host_bash_allowed(SimpleNamespace()) is False
    assert is_host_bash_allowed(SimpleNamespace(sandbox=None)) is False
