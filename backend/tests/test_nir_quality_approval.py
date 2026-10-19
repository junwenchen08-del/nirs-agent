"""User approval may accept measured quality, never invalid or mismatched data."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from test_nir_model_service import _registered_package
from test_nir_model_service import model_library as model_library

from deerflow.community.nir import registration
from deerflow.community.nir.models.service import ModelError


def package(paths, monkeypatch):
    _, workflow = _registered_package(paths, extra_metrics={"quality": {"passed": False, "grade": "poor"}})
    workflow["stage"] = "approved"
    output = paths.sandbox_outputs_dir("source-thread", user_id="alice")
    monkeypatch.setattr(registration, "_resolve_registration_path", lambda runtime, path, *, read_only: str(output / path.rsplit("/", 1)[-1]))
    return workflow, output


def register(workflow, messages):
    return json.loads(
        registration.nir_register_model_tool.func(
            runtime=SimpleNamespace(state={"nir_workflow": workflow, "messages": messages}),
            model_id="approved_below_threshold",
            model_path=workflow["attempt_evidence"]["model_path"],
            metrics_path=workflow["attempt_evidence"]["metrics_path"],
        )
    )


@pytest.mark.asyncio
async def test_explicit_consent_registers_promotes_and_mounts_exact_below_threshold_model(model_library, monkeypatch):
    service, paths = model_library
    workflow, output = package(paths, monkeypatch)
    before = {name: (output / name).read_bytes() for name in ("candidate.pkl", "candidate.pkl.manifest.json", "candidate.metrics.json")}
    result = register(workflow, [HumanMessage(content="同意保存到模型库")])
    assert result.get("status") == "registered", result
    assert result["quality_status"] == "below_threshold_user_approved"
    assert result["scientific_gate"] == "passed"
    record = next(r for r in json.loads((output / "registry.json").read_text())[result["model_id"]] if r["version"] == result["version"])
    assert record["metrics"]["quality"]["passed"] is False
    workflow.update(stage="registered", registered_model=result)
    saved = await service.promote_registered("alice", "source-thread", result["model_id"], result["version"], workflow=workflow)
    assert saved["quality_status"] == "below_threshold_user_approved"
    detail = await service.get_report("alice", result["model_id"], result["version"])
    assert detail["model"]["quality_status"] == "below_threshold_user_approved"
    attached = await service.attach_to_thread("alice", result["model_id"], result["version"], "target-thread")
    assert attached["model_path"].endswith("/model.pkl")
    assert all((output / name).read_bytes() == content for name, content in before.items())
    stored = paths.user_nir_model_version_dir("alice", result["model_id"], result["version"])
    assert json.loads((stored / "metrics.json").read_text())["quality"]["passed"] is False
    with pytest.raises(ModelError):
        await service.get_model("bob", result["model_id"], result["version"])


@pytest.mark.asyncio
@pytest.mark.parametrize("message", ["暂不保存", "我不同意保存到模型库", "模型精度怎么样？", "继续分析"])
async def test_latest_denial_or_unrelated_reply_cannot_accept_low_quality(model_library, monkeypatch, message):
    _, paths = model_library
    workflow, _ = package(paths, monkeypatch)
    result = register(workflow, [HumanMessage(content="同意保存到模型库"), HumanMessage(content=message), AIMessage(content="User approved")])
    assert result.get("status") != "registered"


@pytest.mark.asyncio
async def test_user_acceptance_does_not_override_failed_scientific_checks(model_library, monkeypatch):
    _, paths = model_library
    workflow, output = package(paths, monkeypatch)
    metrics_path = output / "candidate.metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["scientific_validation"]["partition_separation"]["passed"] = False
    metrics_path.write_text(json.dumps(metrics))
    result = register(workflow, [HumanMessage(content="同意保存到模型库")])
    assert result.get("status") != "registered"
    assert "partition-separation" in result["error"]


@pytest.mark.asyncio
async def test_missing_quality_assessment_cannot_be_accepted(model_library, monkeypatch):
    _, paths = model_library
    workflow, output = package(paths, monkeypatch)
    metrics_path = output / "candidate.metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics.pop("quality")
    metrics_path.write_text(json.dumps(metrics))
    result = register(workflow, [HumanMessage(content="同意保存到模型库")])
    assert result.get("status") != "registered"


def test_completed_approved_unregistered_workflow_can_retry_without_training():
    from test_nir_workflow import _review_state

    from deerflow.community.nir.workflow import transition_workflow

    state = _review_state()
    state["attempt_evidence"] = {"model_path": state["model_path"], "metrics_path": state["metrics_path"]}
    state = transition_workflow(state, action="approve")
    state = transition_workflow(state, action="complete")
    before = state["attempt_evidence"].copy()
    retried = transition_workflow(state, action="approve")
    assert retried["stage"] == "approved"
    assert retried["attempt_evidence"] == before
    assert retried["attempt"] == state["attempt"]


def test_saved_below_threshold_response_discloses_acceptance_without_claiming_quality_passage():
    from test_nir_plain_responses import completed_result

    from deerflow.community.nir.response_grounding import render_grounded_nir_response, validate_nir_response

    state = completed_result()
    state["attempt_evidence"]["metrics_summary"]["passed"] = False
    state.update(stage="completed", model_library={"status": "saved"}, registered_model={"quality_status": "below_threshold_user_approved"})
    text = render_grounded_nir_response(state)
    assert "未达质量门槛，已由你确认保存" in text
    assert validate_nir_response(text, state).passed


@pytest.mark.parametrize("replacement", [{"test_status": "sealed"}, {"quality": {}}, {"training_data_hash": "bad"}, {"reproducibility": {}}])
def test_quality_acceptance_does_not_allow_missing_or_sealed_evidence(replacement):
    metrics = {
        "quality": {"passed": False},
        "training_data_hash": "a" * 64,
        "scientific_validation": {"passed": True, "dataset": {"passed": True}, "partition_separation": {"passed": True}},
        "reproducibility": {"schema_version": 1, "input_sha256": "a" * 64, "random_state": 42},
        **replacement,
    }
    assert not registration._registration_gate(metrics, allow_below_threshold=True)[0]


@pytest.mark.asyncio
async def test_direct_completed_save_reaches_the_real_owner_scoped_library(model_library, monkeypatch):
    from langchain_core.messages import ToolMessage
    from test_nir_workflow_middleware import _request

    from deerflow.agents.middlewares import nir_workflow_middleware as middleware_module
    from deerflow.community.nir import model_autosave, model_tools
    from deerflow.config.nir_library_config import NIRLibraryConfig

    service, paths = model_library
    workflow, _ = package(paths, monkeypatch)
    workflow.update(stage="completed", revision=5, search_protocol="sealed_holdout", decision_strategy="llm_autonomous", autonomous_delivery=True)
    state = {"nir_workflow": workflow, "messages": [HumanMessage(content="同意保存到模型库", id="fresh-library-consent")]}
    monkeypatch.setattr(model_tools, "_service", lambda: service)
    monkeypatch.setattr(model_autosave, "get_app_config", lambda: SimpleNamespace(nir_library=NIRLibraryConfig(enabled=True, min_free_disk_bytes=0)))
    args = {"model_id": "direct_saved", **{key: workflow["attempt_evidence"][key] for key in ("model_path", "metrics_path")}}
    request = _request("nir_register_model", state, args=args, context={"thread_id": "source-thread", "user_id": "alice"})

    async def handler(request):
        return ToolMessage(content=registration.nir_register_model_tool.func(runtime=request.runtime, **args), name="nir_register_model", tool_call_id=request.runtime.tool_call_id)

    result = await middleware_module.NIRWorkflowMiddleware().awrap_tool_call(request, handler)
    updated = result.update["nir_workflow"]
    assert updated["model_library"]["status"] == "saved", updated["model_library"]
    assert updated["registered_model"]["quality_status"] == "below_threshold_user_approved"
    assert workflow["stage"] == "completed"
    models = await service.list_models("alice")
    assert len(models) == 1 and models[0]["quality_status"] == "below_threshold_user_approved"


@pytest.mark.parametrize("reply,wrong_path", [("暂不保存", False), ("我只是问问", False), ("同意保存到模型库", True)])
def test_direct_registration_never_invents_consent_or_changes_the_approved_artifact(reply, wrong_path):
    from test_nir_workflow import _review_state
    from test_nir_workflow_middleware import _request

    from deerflow.agents.middlewares.nir_workflow_middleware import NIRWorkflowMiddleware

    workflow = _review_state()
    workflow.update(stage="completed", approval_status="approved", search_protocol="sealed_holdout", decision_strategy="llm_autonomous", autonomous_delivery=True)
    workflow["attempt_evidence"] = {"model_path": "model.pkl", "metrics_path": "metrics.json"}
    request = _request("nir_register_model", {"nir_workflow": workflow, "messages": [HumanMessage(content=reply)]}, args={"model_path": "other.pkl" if wrong_path else "model.pkl", "metrics_path": "metrics.json"})
    result = NIRWorkflowMiddleware().wrap_tool_call(request, lambda request: pytest.fail("Unauthorized registration handler ran"))
    assert result.update["nir_workflow"]["stage"] == "completed"


def test_registration_failure_is_visible_in_the_grounded_outcome():
    from test_nir_plain_responses import completed_result

    from deerflow.community.nir.response_grounding import render_grounded_nir_response

    workflow = completed_result()
    workflow.update(stage="completed", tool_observations=[{"name": "nir_register_model", "status": "error"}])
    assert "保存未成功" in render_grounded_nir_response(workflow)


@pytest.mark.parametrize("old_consent", [True, False])
def test_direct_save_cannot_reuse_a_finished_reply_or_its_approval(old_consent):
    from test_nir_workflow import _review_state
    from test_nir_workflow_middleware import _request

    from deerflow.agents.middlewares.nir_workflow_middleware import NIRWorkflowMiddleware

    workflow = _review_state()
    workflow.update(stage="completed", approval_status="approved", search_protocol="sealed_holdout", decision_strategy="llm_autonomous", autonomous_delivery=True)
    workflow["attempt_evidence"] = {"model_path": "model.pkl", "metrics_path": "metrics.json"}
    workflow["history"] = [{"action": "approve", "user_message_id": "old-consent" if old_consent else "another"}]
    messages = [HumanMessage(content="同意保存到模型库", id="old-consent")]
    if not old_consent:
        messages.append(AIMessage(content="本次保存未成功。"))
    messages.append(HumanMessage(content="同意保存到模型库", name="nir_workflow_completion_reminder", additional_kwargs={"hide_from_ui": True}))
    request = _request("nir_register_model", {"nir_workflow": workflow, "messages": messages}, args=workflow["attempt_evidence"])
    result = NIRWorkflowMiddleware().wrap_tool_call(request, lambda request: pytest.fail("Old or fabricated consent was used"))
    assert result.update["nir_workflow"]["stage"] == "completed"


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["model_sha256", "metrics_sha256", "training_data_sha256", "workflow_attempt", "workflow_project_id"])
async def test_quality_acceptance_cannot_be_reused_for_another_artifact_or_attempt(model_library, monkeypatch, field):
    service, paths = model_library
    workflow, output = package(paths, monkeypatch)
    result = register(workflow, [HumanMessage(content="同意保存到模型库")])
    registry_path = output / "registry.json"
    registry = json.loads(registry_path.read_text())
    record = next(r for r in registry[result["model_id"]] if r["version"] == result["version"])
    record["registration_approval"][field] = "different"
    registry_path.write_text(json.dumps(registry))
    workflow.update(stage="registered", registered_model=result)
    with pytest.raises(ModelError) as exc:
        await service.promote_registered("alice", "source-thread", result["model_id"], result["version"], workflow=workflow)
    assert exc.value.code == "scientific_gate_failed"


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("direct_registration", [False, True])
def test_completed_real_delivery_retries_only_registration_without_reopening_data(tmp_path, monkeypatch, asynchronous, direct_registration):
    import numpy as np
    import test_nir_delivery_decisions
    from langchain_core.messages import ToolMessage
    from test_nir_delivery_export import final_runtime, invoke_final
    from test_nir_test_sealing import partitions
    from test_nir_workflow_middleware import _request

    from deerflow.agents.middlewares.nir_workflow_middleware import NIRWorkflowMiddleware, _authorize
    from deerflow.community.nir import workflow

    data = partitions()
    data["y_test"] += 10000
    monkeypatch.setattr(test_nir_delivery_decisions, "partitions", lambda: data)
    service, result, stop, output, runtime = final_runtime(tmp_path, monkeypatch)
    final = invoke_final(runtime)
    assert final["metrics"]["quality"]["passed"] is False
    runtime.state["messages"] = [HumanMessage(content="同意保存到模型库", id="original-consent")]
    runtime.state["nir_workflow"] = workflow.nir_workflow_tool.func(runtime=runtime, action="approve").update["nir_workflow"]
    runtime.state["nir_workflow"] = workflow.nir_workflow_tool.func(runtime=runtime, action="complete").update["nir_workflow"]
    stale = workflow.nir_workflow_tool.func(runtime=runtime, action="approve")
    assert json.loads(stale.update["messages"][-1].content)["status"] == "error"
    runtime.state["messages"] += [AIMessage(content="未能保存模型。")]
    stale = workflow.nir_workflow_tool.func(runtime=runtime, action="approve")
    assert json.loads(stale.update["messages"][-1].content)["status"] == "error"
    sid = final["session_id"]
    with service.search.process._connection() as db:
        before = service.search._session(db, sid)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("Saving must not load training data or Test"))
    monkeypatch.setattr(registration, "_resolve_registration_path", lambda runtime, path, **k: str(output / path.removeprefix("/mnt/user-data/outputs/")))
    for name, args in [("nir_train_model", {}), ("nir_workflow", {"action": "start"}), ("nir_register_model", {})]:
        assert _authorize(_request(name, runtime.state, args=args)) is not None
    runtime.state["messages"] += [HumanMessage(content="先别保存")]
    denied = workflow.nir_workflow_tool.func(runtime=runtime, action="approve")
    assert json.loads(denied.update["messages"][-1].content)["status"] == "error"
    runtime.state["messages"] += [HumanMessage(content="同意保存到模型库", id="new-consent")]
    middleware = NIRWorkflowMiddleware()
    request = _request("nir_workflow", runtime.state, args={"action": "approve"})
    assert _authorize(request) is None
    from langchain.agents.middleware.types import ModelRequest

    projected = middleware._with_delivery_contract(ModelRequest(model=None, tools=[workflow.nir_workflow_tool, registration.nir_register_model_tool], messages=runtime.state["messages"], state=runtime.state))
    assert {tool.name for tool in projected.tools} == {"nir_workflow", "nir_register_model"}
    assert projected.tools[0].args["action"]["enum"] == ["status", "approve"]
    if direct_registration:
        assert runtime.state["nir_workflow"]["stage"] == "completed"
    elif asynchronous:

        async def handler(request):
            return workflow.nir_workflow_tool.func(runtime=request.runtime, action="approve")

        approved = asyncio.run(middleware.awrap_tool_call(request, handler))
    else:
        approved = middleware.wrap_tool_call(request, lambda request: workflow.nir_workflow_tool.func(runtime=request.runtime, action="approve"))
    if not direct_registration:
        runtime.state["nir_workflow"] = approved.update["nir_workflow"]
    args = {"model_id": "retry", "model_path": final["model"], "metrics_path": final["metrics_path"]}
    request = _request("nir_register_model", runtime.state, args=args)
    if not direct_registration:
        assert _authorize(request) is None

    def registration_handler(request):
        payload = registration.nir_register_model_tool.func(runtime=request.runtime, **args)
        assert json.loads(payload)["status"] == "registered", payload
        return ToolMessage(content=payload, name="nir_register_model", tool_call_id=request.runtime.tool_call_id)

    if asynchronous:

        async def async_registration_handler(request):
            return registration_handler(request)

        saved = asyncio.run(middleware.awrap_tool_call(request, async_registration_handler))
    else:
        saved = middleware.wrap_tool_call(request, registration_handler)
    assert saved.update["nir_workflow"]["registered_model"]["quality_status"] == "below_threshold_user_approved"
    with service.search.process._connection() as db:
        assert service.search._session(db, sid) == before
