"""Trusted ingestion, tool isolation and safe single-target lifecycle."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from langgraph.types import Command
from test_nir_workflow_middleware import _request

from deerflow.agents.middlewares.nir_workflow_middleware import NIRWorkflowMiddleware
from deerflow.community.nir.workflow import start_workflow


def workflow():
    return start_workflow(task_type="calibration", data_path="/mnt/user-data/uploads/data.npz", domain="default", analyte="protein", unit="%", validation_goal="internal_holdout", search_protocol="sealed_holdout")


@pytest.mark.parametrize("tool", ["read_file", "bash", "task", "python", "nir_inspect", "nir_train_model", "nir_compare", "chemotools_run_inspector", "chemotools_fit_estimator", "unknown_plugin", "web_search"])
def test_sealed_protocol_denies_every_unlisted_tool(tool):
    middleware = NIRWorkflowMiddleware()
    called = []
    result = middleware.wrap_tool_call(_request(tool, {"nir_workflow": workflow()}), lambda request: called.append(request))
    assert not called
    message = result.update["messages"][-1] if isinstance(result, Command) else result
    assert json.loads(message.content)["code"] == "nir_sealed_search_tool_denied"


def test_sealed_protocol_cannot_be_selected_for_unadapted_task():
    with pytest.raises(ValueError, match="single-target"):
        start_workflow(task_type="classification", search_protocol="sealed_holdout")


@pytest.mark.parametrize("suffix", [".npz", ".csv"])
def test_safe_tools_to_final_report_and_prediction_preserve_history(tmp_path, monkeypatch, suffix):
    from deerflow.community.nir import search_tools
    from deerflow.community.nir.dataset_session import SearchSessionStore
    from deerflow.community.nir.process import ProcessStore

    rng = np.random.default_rng(606)
    X = rng.normal(size=(60, 8))
    y = X[:, 0] * 2 - X[:, 1] + rng.normal(scale=0.02, size=60)
    source = tmp_path / ("data" + suffix)
    if suffix == ".npz":
        np.savez(source, X=X, y=y, wv=np.arange(8, dtype=float))
    else:
        np.savetxt(source, np.column_stack([y, X]), delimiter=",")
    output = tmp_path / "outputs"
    output.mkdir()
    historical = {"model.joblib": b"historical model bytes", "metrics.json": b"historical metrics", "report.md": b"historical report"}
    for name, contents in historical.items():
        (output / name).write_bytes(contents)
    service = SearchSessionStore(ProcessStore(tmp_path / "private"))
    monkeypatch.setattr(search_tools, "_service", lambda runtime: service)
    monkeypatch.setattr(search_tools, "_resolve", lambda runtime, path, **kw: str(source) if kw.get("read_only", True) else str(tmp_path / "outputs"))
    runtime = SimpleNamespace(state={"nir_workflow": workflow()}, tool_call_id="prepare", context={}, config={})

    def invoke(tool, **kwargs):
        command = tool.func(runtime=runtime, **kwargs)
        assert isinstance(command, Command)
        payload = json.loads(command.update["messages"][-1].content)
        assert payload.get("status") != "error", payload
        runtime.state["nir_workflow"] = command.update.get("nir_workflow", runtime.state["nir_workflow"])
        return payload

    prepared = invoke(search_tools.nir_prepare_search_tool, cv_folds=3)
    runtime.tool_call_id = "train"
    fitted = invoke(search_tools.nir_search_model_tool, method="pls", max_components=3)
    assert "test" not in fitted["metrics"]
    runtime.tool_call_id = "freeze"
    invoke(search_tools.nir_freeze_search_tool, experiment_id=fitted["experiment_id"])
    runtime.tool_call_id = "final"
    final = invoke(search_tools.nir_finalize_search_tool)
    assert final["metrics"]["test"]["scope"] == "final_test"
    assert runtime.state["nir_workflow"]["stage"] == "review"
    assert (tmp_path / "outputs" / prepared["session_id"] / "report.md").is_file()
    assert (tmp_path / "outputs" / prepared["session_id"] / "model.joblib.manifest.json").is_file()
    from deerflow.community.nir._common import _load_trusted_model_artifact

    artifact = _load_trusted_model_artifact(str(tmp_path / "outputs" / prepared["session_id"] / "model.joblib"), f"/mnt/user-data/outputs/{prepared['session_id']}/model.joblib")
    assert artifact["format"] == "nir_model_artifact"
    prediction = artifact["model"].predict(artifact["preprocessing"]["pipeline"].transform(X[:2]))
    assert np.isfinite(prediction).all()
    import joblib

    selected = joblib.load(service._dir(prepared["session_id"]) / (fitted["experiment_id"] + ".joblib"))
    np.testing.assert_allclose(prediction.ravel(), selected.predict(X[:2]))
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("Completed export must be replayed without data IO"))
    assert invoke(search_tools.nir_finalize_search_tool) == final
    for name, contents in historical.items():
        assert (output / name).read_bytes() == contents
