"""Publish phase-4 results through existing report/artifact/prediction gates."""

import json
from types import SimpleNamespace

import numpy as np
from test_nir_delivery_freeze import stopped_session

from deerflow.community.nir import search_tools
from deerflow.community.nir._common import _load_trusted_model_artifact


def final_runtime(tmp_path, monkeypatch):
    service, ctx, result, stop = stopped_session(tmp_path)
    service.search.freeze(ctx["session_id"], result["experiment_id"])
    output = tmp_path / "outputs"
    output.mkdir()
    (output / "report.md").write_text("historical report")
    monkeypatch.setattr(search_tools, "_service", lambda runtime: service.search)
    monkeypatch.setattr(search_tools, "_resolve", lambda *a, **k: str(output))
    from deerflow.community.nir.workflow import start_workflow

    wf = {
        **start_workflow(
            task_type="calibration",
            data_path="/mnt/user-data/uploads/a.npz",
            domain="default",
            analyte="protein",
            unit="%",
            validation_goal="internal_holdout",
            search_protocol="sealed_holdout",
            decision_strategy="llm_autonomous",
            autonomous_rounds=True,
            autonomous_delivery=True,
        ),
        "task_type": "calibration",
        "search_protocol": "sealed_holdout",
        "decision_strategy": "llm_autonomous",
        "autonomous_rounds": True,
        "autonomous_delivery": True,
        "project_id": "delivery",
        "search_session_id": ctx["session_id"],
        "stage": "frozen",
        "revision": 8,
        "analyte": "protein",
        "unit": "%",
        "validation_goal": "internal_holdout",
        "stop_source": "llm",
        "attempts": [],
    }
    runtime = SimpleNamespace(state={"nir_workflow": wf}, tool_call_id="final", context={}, config={})
    return service, result, stop, output, runtime


def invoke_final(runtime):
    command = search_tools.nir_finalize_search_tool.func(runtime=runtime)
    payload = json.loads(command.update["messages"][-1].content)
    runtime.state["nir_workflow"] = command.update.get("nir_workflow", runtime.state["nir_workflow"])
    return payload


def test_delivery_report_manifest_prediction_and_approval(tmp_path, monkeypatch):
    service, result, stop, output, runtime = final_runtime(tmp_path, monkeypatch)
    final = invoke_final(runtime)
    assert final.get("status") != "error", final
    assert final["lifecycle"]["test_completed"] is True and final["lifecycle"]["user_approved"] is False
    assert final["lifecycle"]["registered"] is False
    relative = final["model"].removeprefix("/mnt/user-data/outputs/")
    artifact = _load_trusted_model_artifact(str(output / relative), final["model"])
    assert artifact["autonomous_binding"]["parameters"] == result["actual_parameters"]
    import joblib

    selected = joblib.load(service.search._dir(final["session_id"]) / (result["experiment_id"] + ".joblib"))
    X = np.random.default_rng(808).normal(size=(3, 8))
    np.testing.assert_allclose(artifact["model"].predict(artifact["preprocessing"]["pipeline"].transform(X)).ravel(), selected.predict(X))
    report = (output / final["report"].removeprefix("/mnt/user-data/outputs/")).read_text(encoding="utf-8")
    for value in ("Cal CV", "Val", "Test", "LLM 主动停止", stop["decision_id"], "提出", "实际执行", "alpha"):
        assert value in report
    assert (output / final["bundle"].removeprefix("/mnt/user-data/outputs/")).is_file()
    assert (output / "report.md").read_text() == "historical report"
    from deerflow.community.nir.registration import _registration_gate
    from deerflow.community.nir.workflow import registration_is_approved

    assert not registration_is_approved(runtime.state["nir_workflow"])
    assert _registration_gate(final["metrics"])[0] == final["metrics"]["quality"]["passed"]
    monkeypatch.setattr(np, "load", lambda *a, **k: (_ for _ in ()).throw(AssertionError("No replay IO")))
    assert invoke_final(runtime) == final


def test_completed_test_export_failure_recovers_without_second_test(tmp_path, monkeypatch):
    service, result, stop, output, runtime = final_runtime(tmp_path, monkeypatch)
    from deerflow.community.nir import delivery

    original = delivery.build_delivery
    monkeypatch.setattr(delivery, "build_delivery", lambda *a, **k: (_ for _ in ()).throw(OSError("synthetic export interruption")))
    assert invoke_final(runtime)["status"] == "error"
    sid = runtime.state["nir_workflow"]["search_session_id"]
    with service.search.process._connection() as db:
        session = service.search._session(db, sid)
    assert session["stage"] == "finalized" and session["export_state"]["status"] == "failed"
    original_load = np.load

    def spy(path, *a, **k):
        assert "holdout" not in str(path)
        return original_load(path, *a, **k)

    monkeypatch.setattr(np, "load", spy)
    monkeypatch.setattr(delivery, "build_delivery", original)
    assert invoke_final(runtime).get("status") != "error"
    assert len(list((output / sid).glob("delivery_*"))) == 1
