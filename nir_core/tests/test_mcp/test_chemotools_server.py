from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from nir_core.chemotools_mcp.chemotools_server import (
    apply_estimator,
    describe_capability_tool,
    fit_estimator,
    health,
    list_capabilities,
    mcp,
    validate_operation,
)
from nir_core.chemotools_mcp.workspace import ArtifactStore, ChemotoolsWorkspaceError


def _write_spectra(path: Path) -> np.ndarray:
    X = np.array(
        [
            [1.0, 2.0, 4.0, 8.0, 16.0],
            [2.0, 3.0, 5.0, 9.0, 17.0],
            [4.0, 6.0, 9.0, 13.0, 18.0],
        ]
    )
    np.savez_compressed(path, X=X)
    return X


def test_mcp_exposes_stable_grouped_tools():
    tools = asyncio.run(mcp.list_tools())
    names = {tool.name for tool in tools}

    assert names == {
        "health",
        "list_capabilities",
        "describe_capability",
        "validate_operation",
        "fit_estimator",
        "apply_estimator",
        "call_function",
        "render_plot",
        "run_inspector",
        "get_artifact_metadata",
    }


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="uv Windows venv launchers leave a grandchild process; covered in Docker/Linux",
)
def test_stdio_mcp_protocol_round_trip(tmp_path):
    async def exercise_server() -> tuple[set[str], dict]:
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "nir_core.chemotools_mcp.chemotools_server"],
            cwd=tmp_path,
        )
        async with (
            stdio_client(parameters) as (read_stream, write_stream),
            ClientSession(read_stream, write_stream) as session,
        ):
            await session.initialize()
            tools = await session.list_tools()
            health_result = await session.call_tool("health", {})
            return (
                {tool.name for tool in tools.tools},
                health_result.structuredContent,
            )

    names, status = asyncio.run(exercise_server())

    assert "list_capabilities" in names
    assert "fit_estimator" in names
    assert status["status"] == "ok"
    assert status["provider_version"] == "0.4.4"


def test_catalog_tools_return_pinned_structured_metadata():
    status = health()
    listed = list_capabilities(category="scatter", executable_only=True)
    described = describe_capability_tool("chemotools.scatter.StandardNormalVariate")
    validated = validate_operation(
        "chemotools.scatter.StandardNormalVariate",
        "fit",
    )

    assert status["status"] == "ok"
    assert listed["total"] == 4
    assert described["parameters"]["properties"] == {}
    assert validated["status"] == "valid"


def test_fit_and_apply_estimator_round_trip(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "spectra.npz"
    _write_spectra(input_path)

    fitted = fit_estimator(
        capability_id="chemotools.scatter.StandardNormalVariate",
        input_path=str(input_path),
        output_path="outputs/snv-fit.npz",
    )
    applied = apply_estimator(
        artifact_id=fitted["artifact_id"],
        operation="transform",
        input_path=str(input_path),
        output_path="outputs/snv-apply.npz",
    )

    with np.load(fitted["result"]["output_path"], allow_pickle=False) as archive:
        fitted_result = archive["data"]
    with np.load(applied["result"]["output_path"], allow_pickle=False) as archive:
        applied_result = archive["data"]

    assert fitted["provider_version"] == "0.4.4"
    assert fitted["result_operation"] == "transform"
    np.testing.assert_allclose(fitted_result, applied_result, atol=1e-12)
    np.testing.assert_allclose(fitted_result.mean(axis=1), 0.0, atol=1e-12)


def test_artifact_hash_tampering_fails_closed(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    input_path = tmp_path / "spectra.npz"
    _write_spectra(input_path)
    fitted = fit_estimator(
        capability_id="chemotools.scatter.StandardNormalVariate",
        input_path=str(input_path),
        result_operation="none",
    )
    object_path = Path(fitted["artifact_path"]) / "object.joblib"
    with object_path.open("ab") as handle:
        handle.write(b"tampered")

    with pytest.raises(ChemotoolsWorkspaceError, match="hash verification failed"):
        ArtifactStore().load(fitted["artifact_id"])


def test_workspace_path_escape_is_rejected(monkeypatch, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.npz"
    _write_spectra(outside)
    monkeypatch.chdir(workspace)

    with pytest.raises(ChemotoolsWorkspaceError, match="outside"):
        fit_estimator(
            capability_id="chemotools.scatter.StandardNormalVariate",
            input_path=str(outside),
        )
