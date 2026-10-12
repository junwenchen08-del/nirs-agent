"""Unrecoverable intake errors must not cause blind retries or invented progress."""

import json
from types import SimpleNamespace

import numpy as np

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
