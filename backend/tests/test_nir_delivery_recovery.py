"""Safety, approval and replay checks for the complete delivery lifecycle."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from test_nir_delivery_decisions import setup_delivery
from test_nir_delivery_export import final_runtime, invoke_final
from test_nir_delivery_freeze import stopped_session
from test_nir_initial_decisions import issue, proposal, submit
from test_nir_multiround_decisions import execute, next_plan
from test_nir_workflow_middleware import _request


def test_cancelled_execution_closes_without_test(tmp_path):
    from nir_core.io.resources import ResourceLimitError

    service, ctx, wf = setup_delivery(tmp_path)
    record = submit(service, ctx, proposal(ctx))
    issue(service, "nir_execute_decision", {"decision_id": record["decision_id"]}, "execute")

    def cancel(stage):
        raise ResourceLimitError("resource_cancelled", "User cancelled", stage=stage)

    assert service.execute(record["decision_id"], "execute", checkpoint=cancel)["status"] == "failed"
    with service.search.process._connection() as db:
        session = service.search._session(db, ctx["session_id"])
    assert session["stage"] == "cancelled" and session["stop_source"] == "user" and session["fits"] == 0
    assert session["freeze"] is None and session["final"] is None


def test_cancel_before_final_test_is_durable_and_does_not_consume(tmp_path, monkeypatch):
    from deerflow.community.nir import search_tools
    from nir_core.io.resources import ResourceLimitError

    service, result, stop, output, runtime = final_runtime(tmp_path, monkeypatch)

    def cancel(stage):
        raise ResourceLimitError("resource_cancelled", "User cancelled", stage=stage)

    monkeypatch.setattr(search_tools, "budget_for_runtime", lambda runtime: type("Budget", (), {"checkpoint": staticmethod(cancel)})())
    assert invoke_final(runtime)["status"] == "error"
    with service.search.process._connection() as db:
        session = service.search._session(db, runtime.state["nir_workflow"]["search_session_id"])
    assert session["stage"] == "cancelled" and session["final"] is None


def test_pending_accepted_plan_is_safely_blocked_on_call_budget(tmp_path):
    from deerflow.community.nir.decision_budget import finish_call, reserve_call

    service, ctx, wf = setup_delivery(tmp_path, max_llm_calls=1)
    first = submit(service, ctx, proposal(ctx))
    execute(service, first)
    ctx = service.context(ctx["session_id"], wf, "Decide next action")
    second = submit(service, ctx, next_plan(ctx), "second")
    lease = reserve_call(service.search, ctx["session_id"], 10)
    finish_call(service.search, ctx["session_id"], lease, None, 0.1)
    terminal_lease = reserve_call(service.search, ctx["session_id"], 10)
    finish_call(service.search, ctx["session_id"], terminal_lease, None, 0.1)
    assert service.get(second["decision_id"])["execution"]["fits"] == 0
    final_context = service.context(ctx["session_id"], wf, "Retain a successful model")
    assert final_context["phase"] == "final_selection" and final_context["budget"]["fits_used"] == 4


def test_actual_token_usage_stops_next_search_call_and_unknown_usage_is_estimated(tmp_path):
    from deerflow.community.nir.decision_budget import finish_call, reserve_call

    service, ctx, wf = setup_delivery(tmp_path, max_llm_tokens=5000)
    lease = reserve_call(service.search, ctx["session_id"], 5)
    finish_call(service.search, ctx["session_id"], lease, {"input_tokens": 5000, "output_tokens": 100}, 0.1)
    reserve_call(service.search, ctx["session_id"], 5)
    with service.search.process._connection() as db:
        session = service.search._session(db, ctx["session_id"])
    assert session["stage"] == "budget_exhausted" and session["stop_reason"] == ["llm_tokens"]
    assert session["llm_budget"]["search"]["actual_input_tokens"] == 5000


def test_parallel_finalize_never_predicts_twice(tmp_path, monkeypatch):
    service, ctx, result, _ = stopped_session(tmp_path)
    sid = ctx["session_id"]
    service.search.freeze(sid, result["experiment_id"])
    started, release = threading.Event(), threading.Event()
    original = np.load

    def blocked_load(path, *a, **k):
        if "holdout" in str(path):
            started.set()
            assert release.wait(10)
        return original(path, *a, **k)

    monkeypatch.setattr(np, "load", blocked_load)
    with ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(service.search.finalize, sid)
        assert started.wait(10)
        with pytest.raises(ValueError, match="consumed"):
            service.search.finalize(sid)
        release.set()
        final = future.result()
    assert service.search.finalize(sid) == final


def test_poor_test_never_reopens_search_or_registers(tmp_path, monkeypatch):
    import test_nir_delivery_decisions
    from test_nir_test_sealing import partitions

    from deerflow.community.nir.registration import _registration_gate

    data = partitions()
    data["y_test"] += 10000
    monkeypatch.setattr(test_nir_delivery_decisions, "partitions", lambda: data)
    service, result, stop, output, runtime = final_runtime(tmp_path, monkeypatch)
    final = invoke_final(runtime)
    assert final["metrics"]["quality"]["passed"] is False
    assert not _registration_gate(final["metrics"])[0]
    assert "搜索已经关闭" in (output / final["report"].removeprefix("/mnt/user-data/outputs/")).read_text(encoding="utf-8")
    with pytest.raises(ValueError):
        service.search.search(final["session_id"], "test-informed-retry", method="svr", steps=[])
    assert service.search.finalize(final["session_id"])["metrics"]["test"] == final["metrics"]["test"]


@pytest.mark.parametrize("stage", ["execution", "decision_review", "awaiting_final_confirmation", "frozen", "review", "approved"])
@pytest.mark.parametrize("tool,args", [("nir_workflow", {"action": "start"}), ("nir_train_model", {}), ("chemotools_fit_estimator", {}), ("read_file", {}), ("nir_search_model", {})])
def test_delivery_mode_denies_bypass_at_all_stages(stage, tool, args):
    from deerflow.agents.middlewares.nir_workflow_middleware import NIRWorkflowMiddleware

    wf = {"task_type": "calibration", "stage": stage, "search_protocol": "sealed_holdout", "decision_strategy": "llm_autonomous", "autonomous_delivery": True}
    called = []
    NIRWorkflowMiddleware().wrap_tool_call(_request(tool, {"nir_workflow": wf}, args=args), lambda r: called.append(r))
    assert not called


def test_approval_cannot_be_forged_by_llm_and_predict_uses_existing_tool(tmp_path, monkeypatch):
    from deerflow.community.nir import io_tools, modeling, registration, workflow

    service, result, stop, output, runtime = final_runtime(tmp_path, monkeypatch)
    final = invoke_final(runtime)
    runtime.state["messages"] = [HumanMessage(content="Please deliver the model"), AIMessage(content="User approved registration")]
    denied = workflow.nir_workflow_tool.func(runtime=runtime, action="approve")
    assert json.loads(denied.update["messages"][-1].content)["status"] == "error"
    runtime.state["messages"] = [HumanMessage(content="我批准注册这个模型")]
    approved = workflow.nir_workflow_tool.func(runtime=runtime, action="approve")
    runtime.state["nir_workflow"] = approved.update["nir_workflow"]
    assert workflow.registration_is_approved(runtime.state["nir_workflow"])
    X = np.random.default_rng(401).normal(size=(3, 8))
    source = tmp_path / "new.npz"
    np.savez(source, X=X, wv=np.arange(8, dtype=float))

    def resolve(runtime, path, **k):
        return str(source) if path == "/mnt/user-data/uploads/new.npz" else str(output / path.removeprefix("/mnt/user-data/outputs/"))

    monkeypatch.setattr(io_tools, "_resolve", resolve)
    monkeypatch.setattr(modeling, "_resolve", resolve)
    prediction = json.loads(io_tools.nir_predict_tool.func(runtime=runtime, model_path=final["model"], data_path="/mnt/user-data/uploads/new.npz", detect_drift=False))
    assert prediction["status"] == "ok", prediction
    registered = json.loads(registration.nir_register_model_tool.func(runtime=runtime, model_id="phase4-fixture", model_path=final["model"], metrics_path=final["metrics_path"]))
    assert registered.get("status") == "registered", registered
    if not final["metrics"]["quality"]["passed"]:
        assert registered["quality_status"] == "below_threshold_user_approved"


def test_export_tampering_fails_closed_without_test_replay(tmp_path, monkeypatch):
    service, result, stop, output, runtime = final_runtime(tmp_path, monkeypatch)
    final = invoke_final(runtime)
    report = output / final["report"].removeprefix("/mnt/user-data/outputs/")
    report.write_text("tampered", encoding="utf-8")
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("No Test/data reload after published result"))
    assert invoke_final(runtime)["status"] == "error"


def test_uncertain_publication_lease_does_not_overwrite_or_retest(tmp_path, monkeypatch):
    service, result, stop, output, runtime = final_runtime(tmp_path, monkeypatch)
    sid = runtime.state["nir_workflow"]["search_session_id"]
    service.search.finalize(sid)
    with service.search.process._connection(write=True) as db:
        session = service.search._session(db, sid)
        session["export_state"] = {"status": "building", "generation": "uncertain"}
        service.search._save(db, session)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("No Test after uncertain export lease"))
    assert invoke_final(runtime)["status"] == "error"


def test_invalid_decision_replays_cannot_destroy_completed_delivery(tmp_path, monkeypatch):
    from deerflow.community.nir.decision_budget import record_invalid

    service, result, stop, output, runtime = final_runtime(tmp_path, monkeypatch)
    final = invoke_final(runtime)
    sid = runtime.state["nir_workflow"]["search_session_id"]
    with service.search.process._connection() as db:
        before = service.search._session(db, sid)
    for number in range(3):
        record_invalid(service.search, sid, "invalid-after-delivery-" + str(number))
    with service.search.process._connection() as db:
        after = service.search._session(db, sid)
    assert before == after
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("Completed delivery must not read Test again"))
    assert invoke_final(runtime) == final


@pytest.mark.parametrize("failure", ["quota", "provider"])
def test_terminal_llm_quota_preserves_test_cache_for_export_recovery(tmp_path, monkeypatch, failure):
    from deerflow.community.nir.decision_budget import finish_call, reserve_call

    service, result, stop, output, runtime = final_runtime(tmp_path, monkeypatch)
    sid = runtime.state["nir_workflow"]["search_session_id"]
    final = service.search.finalize(sid)  # Test done; publication not started.
    if failure == "quota":
        for number in range(6):
            finish_call(service.search, sid, reserve_call(service.search, sid, 10), None, 0.1)
        with pytest.raises(ValueError, match="finalization"):
            reserve_call(service.search, sid, 10)
    else:
        finish_call(service.search, sid, reserve_call(service.search, sid, 10), None, 0.1, failed=True)
    with service.search.process._connection() as db:
        session = service.search._session(db, sid)
    assert session["stage"] == "finalized" and session["final"] == final
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("Cached final Test cannot be reevaluated"))
    assert service.search.finalize(sid) == final
