"""Unrecoverable intake errors must not cause blind retries or invented progress."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from deerflow.community.nir import search_tools
from deerflow.community.nir.dataset_session import SearchSessionStore
from deerflow.community.nir.process import ProcessStore
from deerflow.community.nir.workflow import start_workflow


def runtime_for(tmp_path, monkeypatch, *, invalid=False):
    rng = np.random.default_rng(535)
    data = {}
    for name, count in (("cal", 20), ("val", 6), ("test", 6)):
        data["X_" + name] = rng.normal(size=(count, 8))
        data["y_" + name] = data["X_" + name][:, 0].copy()
    if invalid:
        data["y_test"][0] = np.nan
    source = tmp_path / "source.npz"
    np.savez(source, **data)
    service = SearchSessionStore(ProcessStore(tmp_path / "process"))
    monkeypatch.setattr(search_tools, "_service", lambda _: service)
    monkeypatch.setattr(search_tools, "_resolve", lambda *args, **kwargs: str(source))
    workflow = start_workflow(
        task_type="calibration",
        data_path="/mnt/user-data/uploads/source.npz",
        domain="default",
        analyte="protein",
        unit="%",
        validation_goal="internal_holdout",
        search_protocol="sealed_holdout",
        decision_strategy="llm_autonomous",
        autonomous_rounds=True,
        autonomous_delivery=True,
    )
    return SimpleNamespace(state={"nir_workflow": workflow}, context={}, config={}, tool_call_id="ingest"), service


def test_numeric_integrity_failure_closes_intake_without_test_statistics(tmp_path, monkeypatch):
    runtime, service = runtime_for(tmp_path, monkeypatch, invalid=True)
    result = search_tools.nir_prepare_search_tool.func(runtime=runtime, cv_folds=3)
    payload = json.loads(result.update["messages"][0].content)
    assert payload["code"] == "nir_ingestion_numeric_integrity"
    workflow = result.update["nir_workflow"]
    assert workflow["stage"] == "blocked" and workflow["audit_status"] == "failed"
    assert workflow["ingestion_evidence"]["source_tool"] == "nir_prepare_search"
    assert workflow["next_action"] == "resolve_ingestion"
    assert not workflow.get("search_session_id")
    for forbidden in ("y_test", "X_test", "NaN", "nan", "RMSE", str(tmp_path)):
        assert forbidden not in json.dumps([payload, workflow])
    with service.process._connection() as db:
        assert db.execute("SELECT count(*) FROM search_sessions").fetchone()[0] == 0


def test_cv_argument_error_allows_correction_before_ingestion(tmp_path, monkeypatch):
    runtime, _ = runtime_for(tmp_path, monkeypatch)
    with monkeypatch.context() as scoped:
        scoped.setattr(search_tools, "_load_npz_safely", lambda _: (_ for _ in ()).throw(AssertionError("invalid CV must not read data")))
        result = search_tools.nir_prepare_search_tool.func(runtime=runtime, cv_folds=1)
    payload = json.loads(result.update["messages"][0].content)
    assert payload["code"] == "nir_prepare_arguments_invalid"
    assert payload["field"] == "cv_folds" and payload["allowed_range"] == [2, 10]
    assert "nir_workflow" not in result.update
    corrected = search_tools.nir_prepare_search_tool.func(runtime=runtime, cv_folds=3)
    assert corrected.update["nir_workflow"]["search_session_id"]


@pytest.mark.parametrize("roles,message", [([1] * 20 + [2] * 6 + [3] * 5 + [0], "待分配"), ([1] * 32, "分别分配")])
def test_incomplete_saved_roles_stop_before_split_or_session(tmp_path, monkeypatch, roles, message):
    import nir_core.model.evaluation as evaluation

    from deerflow.agents.middlewares.nir_workflow_middleware import NIRWorkflowMiddleware
    from deerflow.community.nir.datasets.workspace import Workbook, validate_settings, write_workspace_csv
    from deerflow.community.nir.workflow import nir_workflow_tool

    runtime, service = runtime_for(tmp_path, monkeypatch)
    X = np.random.default_rng(937).normal(size=(32, 8))
    book = Workbook(X, np.arange(8) + 1000, [f"s{i}" for i in range(32)], ["含量"], [[v] for v in X[:, 0]], roles, {})
    source = tmp_path / "saved.csv"
    write_workspace_csv(book, validate_settings(book, roles, [1000, 1007]), source)
    original = source.read_bytes()
    monkeypatch.setattr(search_tools, "_resolve", lambda *a, **k: str(source))
    monkeypatch.setattr(evaluation, "split_dataset", lambda *a, **k: pytest.fail("Explicit roles must never be randomized"))
    monkeypatch.setattr(service, "prepare", lambda *a, **k: pytest.fail("Invalid roles must not start a search session"))
    command = search_tools.nir_prepare_search_tool.func(runtime=runtime, cv_folds=3)
    payload = json.loads(command.update["messages"][0].content)
    assert payload["code"] == "nir_ingestion_workspace_invalid"
    assert message in payload["error"]
    assert payload["retryable"] is False
    workflow = command.update["nir_workflow"]
    assert workflow["stage"] == "blocked" and not workflow.get("search_session_id")
    assert str(tmp_path) not in json.dumps(payload)
    with service.process._connection() as db:
        assert db.execute("SELECT count(*) FROM search_sessions").fetchone()[0] == 0
    assert source.read_bytes() == original

    from langchain.agents.middleware.types import ModelRequest
    from langchain_core.messages import SystemMessage

    request = ModelRequest(model=None, state={"nir_workflow": workflow}, tools=[nir_workflow_tool, search_tools.nir_prepare_search_tool], system_message=SystemMessage(content=""), messages=[])
    projected = NIRWorkflowMiddleware._with_delivery_contract(request)
    assert [item.name for item in projected.tools] == ["nir_workflow"]
