"""Docker regression: a completed MAT inspection can enter sealed modeling."""

import asyncio
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
from langchain_core.messages import HumanMessage
from scipy.io import savemat
from test_nir_delivery_agent import make_delivery_lead
from test_nir_unified_agent import UnifiedFakeLLM
from test_nir_workflow_middleware import _request

from deerflow.agents.middlewares.nir_workflow_middleware import _authorize
from deerflow.community.nir import search_tools
from deerflow.community.nir.dataset_session import SearchSessionStore
from deerflow.community.nir.process import ProcessStore, runtime_store
from deerflow.community.nir.web_entry import bind_web_request, stored_workflow
from deerflow.community.nir.workflow import start_workflow
from deerflow.config import paths
from deerflow.runtime.user_context import get_effective_user_id


def mat_file(path, *, transpose=False, nested=False):
    X = np.random.default_rng(910).normal(size=(60, 100))
    data = {"X": X.T if transpose else X, "y": (X[:, 0] * 3 - X[:, 1])[None, :], "wv": np.linspace(1100, 2500, 100)[None, :]}
    savemat(path, {"data": data} if nested else data)
    return X


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_inspection_and_loader_agree_on_normalized_mat_dimensions(tmp_path, transpose, nested):
    from nir_core.io.loaders import load_mat
    from nir_core.io.sniffers import inspect_file

    source = tmp_path / "source.mat"
    X = mat_file(source, transpose=transpose, nested=nested)
    data = load_mat(str(source))
    info = json.loads(inspect_file(str(source)))
    np.testing.assert_array_equal(data.X, X)
    assert info["estimated_samples"] == 60
    assert info["estimated_wavelengths"] == 100
    assert info["structure"] == "samples_in_rows"


def prepared_runtime(tmp_path, monkeypatch, source):
    store = SearchSessionStore(ProcessStore(tmp_path / "private"))
    monkeypatch.setattr(search_tools, "_service", lambda runtime: store)
    monkeypatch.setattr(search_tools, "_resolve", lambda *a, **k: str(source))
    workflow = start_workflow(task_type="calibration", data_path="/mnt/user-data/uploads/source.mat", domain="default", analyte="target", unit="source unit unspecified", validation_goal="internal_holdout", search_protocol="sealed_holdout")
    runtime = SimpleNamespace(state={"nir_workflow": workflow}, tool_call_id="prepare", context={}, config={})
    return runtime, store


@pytest.mark.parametrize("transpose", [False, True])
def test_mat_prepare_seals_holdout_once_and_keeps_axis(tmp_path, monkeypatch, transpose):
    source = tmp_path / "source.mat"
    mat_file(source, transpose=transpose)
    runtime, store = prepared_runtime(tmp_path, monkeypatch, source)
    command = search_tools.nir_prepare_search_tool.func(runtime=runtime, cv_folds=3)
    payload = json.loads(command.update["messages"][-1].content)
    assert payload.get("status") != "error", payload
    sid = payload["session_id"]
    runtime.state.update(command.update)
    monkeypatch.setattr(search_tools, "_resolve", lambda *a, **k: pytest.fail("Repeated preparation must not reread source or Test"))
    again = search_tools.nir_prepare_search_tool.func(runtime=runtime, cv_folds=3)
    assert json.loads(again.update["messages"][-1].content)["session_id"] == sid
    with np.load(store._dir(sid) / "search.npz", allow_pickle=False) as arrays:
        assert "X_test" not in arrays and arrays["X_cal"].shape == (36, 100)
        np.testing.assert_array_equal(arrays["wv"], np.linspace(1100, 2500, 100))


def test_ambiguous_mat_returns_mapping_error_and_can_be_corrected(tmp_path, monkeypatch):
    source = tmp_path / "source.mat"
    X = np.random.default_rng(912).normal(size=(60, 100))
    savemat(source, {"spectra_a": X, "spectra_b": X + 1, "y": X[:, 0], "wv": np.arange(100)})
    runtime, store = prepared_runtime(tmp_path, monkeypatch, source)
    failed = search_tools.nir_prepare_search_tool.func(runtime=runtime, cv_folds=3)
    payload = json.loads(failed.update["messages"][-1].content)
    assert payload["code"] == "nir_mat_mapping_required", payload
    assert "x_var" in payload["fields"] and "spectra_a" in payload["mapping"]["x_candidates"]
    assert "nir_workflow" not in failed.update, "Mapping clarification is not an irreversible task failure"
    with store.process._connection() as db:
        assert not db.execute("SELECT id FROM search_sessions").fetchall()
    corrected = search_tools.nir_prepare_search_tool.func(runtime=runtime, cv_folds=3, x_var="spectra_b", y_var="y", wv_var="wv")
    assert json.loads(corrected.update["messages"][-1].content).get("session_id")


def completed_inspection():
    old = start_workflow(task_type="inspection", data_path="/mnt/user-data/uploads/source.mat")
    old.update(stage="completed", audit_status="passed", next_action="workflow_completed")
    old["history"].append({"action": "complete", "notes": "Existing immutable inspection evidence"})
    return old


@pytest.mark.parametrize("legacy_tool", ["nir_train_model", "nir_analyze", "chemotools_fit_estimator", "nir_search_model"])
def test_inspection_handoff_cannot_bypass_unified_start(legacy_tool):
    request = _request(legacy_tool, {"nir_workflow": completed_inspection()}, context={"nir_unified_entry": {"schema_version": 1, "intent": "pending"}})
    denial = _authorize(request)
    assert json.loads(denial.content)["code"] == "nir_unified_start_required"


class MATFakeLLM(UnifiedFakeLLM):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        for call in result.generations[0].message.tool_calls:
            if call["name"] == "nir_workflow":
                call["args"]["data_path"] = "/mnt/user-data/uploads/source.mat"
                call["args"].pop("validation_goal", None)
        return result


@pytest.mark.parametrize("after_inspection", [False, True])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_real_lead_mat_two_experiments_stop_and_automatic_delivery(tmp_path, monkeypatch, after_inspection, asynchronous):
    monkeypatch.setattr(paths, "_paths", paths.Paths(tmp_path / "home"))
    monkeypatch.setenv("NIR_AUTONOMOUS_MAX_EXPERIMENTS", "3")
    monkeypatch.setenv("NIR_AUTONOMOUS_MAX_FITS", "12")
    thread = "mat-inspection-" + str(after_inspection)
    uid = get_effective_user_id()
    paths.get_paths().ensure_thread_dirs(thread, user_id=uid)
    source = paths.get_paths().sandbox_uploads_dir(thread, user_id=uid) / "source.mat"
    output = paths.get_paths().sandbox_outputs_dir(thread, user_id=uid)
    mat_file(source)
    monkeypatch.setattr(search_tools, "_resolve", lambda *a, **k: str(source if k.get("read_only") else output))
    task = HumanMessage(content="请使用已上传 MAT 光谱预测指定目标，依据验证反馈自主优化并交付模型。")
    config = {"configurable": {"thread_id": thread}, "recursion_limit": 90}
    body = SimpleNamespace(assistant_id="lead_agent", context={}, input={"messages": [task]})
    bind_web_request(config, body)
    old = completed_inspection() if after_inspection else None
    saved_old = copy.deepcopy(old)
    state = {**body.input, **({"nir_workflow": old} if old else {})}
    original_load = np.load
    tests = []

    def load(path, *a, **k):
        if "holdout.npz" in str(path):
            tests.append(str(path))
        return original_load(path, *a, **k)

    monkeypatch.setattr(np, "load", load)
    graph = make_delivery_lead(monkeypatch, MATFakeLLM())
    result = asyncio.run(graph.ainvoke(state, config, context=config["context"])) if asynchronous else graph.invoke(state, config, context=config["context"])
    workflow = result["nir_workflow"]
    assert workflow["stage"] == "review", result["messages"][-1]
    assert workflow["validation_goal"] == "internal_holdout"
    assert workflow["decision_strategy"] == "llm_autonomous"
    process = runtime_store(SimpleNamespace(config=config, context=config["context"]))
    with process._connection() as db:
        session = SearchSessionStore(process)._session(db, workflow["search_session_id"])
        decisions = [json.loads(row[0]) for row in db.execute("SELECT payload FROM initial_decisions ORDER BY rowid")]
        if old:
            archived = json.loads(db.execute("SELECT payload FROM model_issued_calls WHERE id=?", (workflow["previous_workflow_id"],)).fetchone()[0])
            assert archived["workflow"] == saved_old and old == saved_old
    assert [d["action"] for d in decisions] == ["EXPLORE", "TUNE", "STOP"]
    assert session["experiments"] == 2 and session["fits"] == 8
    assert len(tests) == 1 and session["stop_source"] == "llm"
    assert session["automatic_delivery"]["status"] == "delivered"
    assert session["freeze"]["experiment_id"] == decisions[0]["experiment_id"]
    assert stored_workflow(process)["search_session_id"] == session["id"]


@pytest.mark.parametrize("changes", [{"stage": "execution"}, {"task_type": "calibration"}, {"search_session_id": "existing"}, {"attempts": [{"method": "ridge"}]}, {"attempt_evidence": {"model_path": "existing.joblib"}}])
def test_only_completed_non_modeling_inspection_can_transition(changes):
    from deerflow.community.nir.web_entry import completed_intake_workflow

    assert completed_intake_workflow(completed_inspection())
    assert not completed_intake_workflow({**completed_inspection(), **changes})


@pytest.mark.parametrize("format", ["v5", "v73"])
def test_multiple_mat_targets_require_explicit_choice(tmp_path, monkeypatch, format):
    source = tmp_path / "source.mat"
    X = np.random.default_rng(913).normal(size=(60, 100))
    data = {"X": X, "protein": X[:, 0], "moisture": X[:, 1], "wv": np.arange(100)}
    if format == "v5":
        savemat(source, data)
    else:
        import h5py

        with h5py.File(source, "w") as f:
            for name, arr in data.items():
                f.create_dataset(name, data=arr)
    runtime, _ = prepared_runtime(tmp_path, monkeypatch, source)
    failed = search_tools.nir_prepare_search_tool.func(runtime=runtime, cv_folds=3)
    assert json.loads(failed.update["messages"][-1].content)["code"] == "nir_mat_mapping_required"
    corrected = search_tools.nir_prepare_search_tool.func(runtime=runtime, cv_folds=3, x_var="X", y_var="moisture", wv_var="wv")
    assert json.loads(corrected.update["messages"][-1].content).get("session_id")


def test_explicit_nested_mapping_and_bad_variable_feedback(tmp_path, monkeypatch):
    source = tmp_path / "source.mat"
    mat_file(source, nested=True)
    runtime, _ = prepared_runtime(tmp_path, monkeypatch, source)
    failed = search_tools.nir_prepare_search_tool.func(runtime=runtime, cv_folds=3, x_var="data.missing", y_var="data.y", wv_var="data.wv")
    payload = json.loads(failed.update["messages"][-1].content)
    assert payload["code"] == "nir_mat_mapping_required" and payload["field"] == "x_var"
    corrected = search_tools.nir_prepare_search_tool.func(runtime=runtime, cv_folds=3, x_var="data.X", y_var="data.y", wv_var="data.wv")
    assert json.loads(corrected.update["messages"][-1].content).get("session_id")


@pytest.mark.parametrize("current_kind", ["active_inspection", "closed_modeling", "bound_unified"])
def test_unified_start_does_not_reset_other_workflows(tmp_path, monkeypatch, current_kind):
    monkeypatch.setattr(paths, "_paths", paths.Paths(tmp_path / "home"))
    thread = current_kind
    uid = get_effective_user_id()
    paths.get_paths().ensure_thread_dirs(thread, user_id=uid)
    source = paths.get_paths().sandbox_uploads_dir(thread, user_id=uid) / "source.mat"
    mat_file(source)
    old = completed_inspection()
    if current_kind == "active_inspection":
        old["stage"] = "data_audit"
    elif current_kind == "closed_modeling":
        old["task_type"] = "calibration"
    else:
        old.update(search_protocol="sealed_holdout", search_session_id="immutable-session")
    config = {"configurable": {"thread_id": thread}}
    ctx = {"nir_unified_entry": {"schema_version": 1, "intent": "pending"}}
    from deerflow.community.nir.workflow import nir_workflow_tool

    runtime = SimpleNamespace(state={"nir_workflow": old}, config=config, context=ctx, tool_call_id="new")
    result = nir_workflow_tool.func(runtime=runtime, action="start", task_type="calibration", data_path="/mnt/user-data/uploads/source.mat")
    assert json.loads(result.update["messages"][-1].content)["status"] == "error"
    assert "nir_workflow" not in result.update and stored_workflow(runtime_store(runtime)) is None


def test_completed_intake_cannot_hide_existing_search_records(tmp_path, monkeypatch):
    from test_nir_test_sealing import partitions

    from deerflow.community.nir.web_entry import persist_unified_workflow, unified_policy

    monkeypatch.setattr(paths, "_paths", paths.Paths(tmp_path / "home"))
    runtime = SimpleNamespace(config={"configurable": {"thread_id": "old-search"}}, context={})
    process = runtime_store(runtime)
    SearchSessionStore(process).prepare("old-project", partitions())
    new = {"project_id": "new-project", "delivery_policy": unified_policy()}
    with pytest.raises(ValueError, match="cannot restart"):
        persist_unified_workflow(runtime, new, starting=True, previous=completed_inspection())
    assert stored_workflow(process) is None
