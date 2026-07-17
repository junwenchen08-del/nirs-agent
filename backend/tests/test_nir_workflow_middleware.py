"""Tests for runtime enforcement of the NIR agent workflow."""

from __future__ import annotations

import json

import pytest
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from deerflow.agents.middlewares.nir_workflow_middleware import NIRWorkflowMiddleware
from deerflow.community.nir.workflow import start_workflow, transition_workflow


def _request(tool_name: str, state: dict, *, call_id: str = "call-1", context: dict | None = None) -> ToolCallRequest:
    runtime = ToolRuntime(
        state=state,
        context=context or {},
        config={"configurable": {}},
        stream_writer=lambda _: None,
        tools=[],
        tool_call_id=call_id,
        store=None,
    )
    return ToolCallRequest(
        tool_call={"id": call_id, "name": tool_name, "args": {}},
        tool=None,
        state=state,
        runtime=runtime,
    )


def _execution_state(*, max_attempts: int = 3, task_type: str = "calibration") -> dict:
    state = start_workflow(
        task_type=task_type,
        data_path="/mnt/user-data/uploads/data.npz",
        analyte="protein",
        unit="%",
        domain="food_protein",
        max_attempts=max_attempts,
    )
    state = transition_workflow(state, action="record_audit", audit_passed=True)
    return transition_workflow(state, action="plan_ready")


def _result(tool_name: str, payload: dict, *, call_id: str = "call-1") -> ToolMessage:
    return ToolMessage(
        content=json.dumps(payload),
        tool_call_id=call_id,
        name=tool_name,
    )


def test_domain_tool_requires_started_workflow() -> None:
    middleware = NIRWorkflowMiddleware()
    called = False

    def handler(_: ToolCallRequest) -> ToolMessage:
        nonlocal called
        called = True
        return _result("nir_analyze", {"status": "ok"})

    result = middleware.wrap_tool_call(_request("nir_analyze", {}), handler)

    assert called is False
    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    payload = json.loads(result.content)
    assert payload["code"] == "nir_workflow_required"
    assert payload["next_action"] == "start_workflow"


def test_domain_tool_is_denied_outside_authorized_stage() -> None:
    middleware = NIRWorkflowMiddleware()
    workflow = start_workflow(
        task_type="calibration",
        data_path="data.npz",
        analyte="protein",
        unit="%",
        domain="food_protein",
    )

    result = middleware.wrap_tool_call(
        _request("nir_train_model", {"nir_workflow": workflow}),
        lambda _: _result("nir_train_model", {"status": "ok"}),
    )

    assert isinstance(result, Command)
    payload = json.loads(result.update["messages"][0].content)
    assert payload["code"] == "nir_workflow_stage_denied"
    assert payload["stage"] == "data_audit"
    assert payload["allowed_stages"] == ["execution"]
    assert payload["next_action"] == "inspect_data"


def test_tool_is_denied_for_incompatible_task_type() -> None:
    middleware = NIRWorkflowMiddleware()
    workflow = _execution_state(task_type="calibration")

    result = middleware.wrap_tool_call(
        _request("nir_predict", {"nir_workflow": workflow}),
        lambda _: _result("nir_predict", {"status": "ok"}),
    )

    assert isinstance(result, Command)
    assert json.loads(result.update["messages"][0].content)["code"] == "nir_workflow_task_denied"


def test_unrelated_tool_passes_through_without_workflow() -> None:
    middleware = NIRWorkflowMiddleware()
    original = ToolMessage(content="ok", tool_call_id="call-1", name="read_file")

    result = middleware.wrap_tool_call(_request("read_file", {}), lambda _: original)

    assert result is original


def test_successful_model_result_automatically_enters_review() -> None:
    middleware = NIRWorkflowMiddleware()
    workflow = _execution_state()
    message = _result(
        "nir_train_model",
        {
            "status": "ok",
            "passed": True,
            "grade": "A",
            "model_path": "/mnt/user-data/outputs/model.pkl",
            "metrics_path": "/mnt/user-data/outputs/metrics.json",
        },
    )

    result = middleware.wrap_tool_call(
        _request("nir_train_model", {"nir_workflow": workflow}),
        lambda _: message,
    )

    assert isinstance(result, Command)
    assert result.update["messages"] == [message]
    updated = result.update["nir_workflow"]
    assert updated["stage"] == "review"
    assert updated["attempt"] == 1
    assert updated["approval_status"] == "pending"
    assert updated["model_path"] == "/mnt/user-data/outputs/model.pkl"
    assert updated["metrics_path"] == "/mnt/user-data/outputs/metrics.json"


def test_successful_tool_observation_is_checkpointed_with_run_context() -> None:
    middleware = NIRWorkflowMiddleware()
    workflow = _execution_state()

    result = middleware.wrap_tool_call(
        _request(
            "nir_train_model",
            {"nir_workflow": workflow},
            context={"run_id": "run-001", "deerflow_trace_id": "trace-001"},
        ),
        lambda _: _result(
            "nir_train_model",
            {
                "status": "ok",
                "passed": True,
                "grade": "A",
                "model_path": "model.pkl",
                "metrics_path": "metrics.json",
            },
        ),
    )

    assert isinstance(result, Command)
    updated = result.update["nir_workflow"]
    assert updated["run_ids"] == ["run-001"]
    assert updated["trace_ids"] == ["trace-001"]
    assert updated["tool_observations"][-1] == {
        "name": "nir_train_model",
        "status": "success",
        "stage_before": "execution",
        "stage_after": "review",
        "code": None,
        "call_id": "call-1",
        "run_id": "run-001",
        "trace_id": "trace-001",
    }


def test_unrelated_tool_is_not_recorded_in_active_nir_workflow() -> None:
    middleware = NIRWorkflowMiddleware()
    workflow = _execution_state()
    message = _result("read_file", {"status": "ok"})

    result = middleware.wrap_tool_call(
        _request("read_file", {"nir_workflow": workflow}),
        lambda _: message,
    )

    assert result is message


def test_denied_tool_observation_is_checkpointed_for_agent_evaluation() -> None:
    middleware = NIRWorkflowMiddleware()
    workflow = start_workflow(
        task_type="calibration",
        data_path="data.npz",
        analyte="protein",
        unit="%",
        domain="food_protein",
    )

    result = middleware.wrap_tool_call(
        _request(
            "nir_train_model",
            {"nir_workflow": workflow},
            context={"run_id": "run-denied"},
        ),
        lambda _: _result("nir_train_model", {"status": "ok"}),
    )

    assert isinstance(result, Command)
    message = result.update["messages"][0]
    assert json.loads(message.content)["code"] == "nir_workflow_stage_denied"
    observation = result.update["nir_workflow"]["tool_observations"][-1]
    assert observation["status"] == "error"
    assert observation["code"] == "nir_workflow_stage_denied"
    assert observation["stage_before"] == "data_audit"
    assert observation["stage_after"] == "data_audit"


def test_failed_model_result_enters_knowledge_and_search_returns_to_planning() -> None:
    middleware = NIRWorkflowMiddleware()
    workflow = _execution_state()
    model_result = middleware.wrap_tool_call(
        _request("nir_analyze", {"nir_workflow": workflow}),
        lambda _: _result(
            "nir_analyze",
            {
                "status": "ok",
                "passed": False,
                "grade": "C",
                "model": "/mnt/user-data/outputs/model.pkl",
                "metrics": "/mnt/user-data/outputs/metrics.json",
            },
        ),
    )
    assert isinstance(model_result, Command)
    failed_state = model_result.update["nir_workflow"]
    assert failed_state["stage"] == "knowledge"
    assert failed_state["attempt"] == 1

    search_result = middleware.wrap_tool_call(
        _request("nir_search_knowledge", {"nir_workflow": failed_state}, call_id="call-2"),
        lambda _: _result("nir_search_knowledge", {"results": [], "count": 0}, call_id="call-2"),
    )

    assert isinstance(search_result, Command)
    retried_state = search_result.update["nir_workflow"]
    assert retried_state["stage"] == "planning"
    assert retried_state["next_action"] == "prepare_retry_plan"


def test_tool_error_does_not_advance_workflow() -> None:
    middleware = NIRWorkflowMiddleware()
    workflow = _execution_state()
    message = ToolMessage(
        content=json.dumps({"status": "error", "error": "training failed"}),
        tool_call_id="call-1",
        name="nir_train_model",
        status="error",
    )

    result = middleware.wrap_tool_call(
        _request("nir_train_model", {"nir_workflow": workflow}),
        lambda _: message,
    )

    assert isinstance(result, Command)
    assert result.update["messages"] == [message]
    updated = result.update["nir_workflow"]
    assert updated["stage"] == "execution"
    assert updated["attempt"] == 0
    assert updated["tool_observations"][-1]["status"] == "error"


def test_compare_uses_best_result_and_preserves_existing_command_fields() -> None:
    middleware = NIRWorkflowMiddleware()
    workflow = _execution_state(task_type="compare")
    message = _result(
        "nir_compare",
        {
            "status": "ok",
            "best": {"passed": True, "grade": "B"},
            "all_metrics": "/mnt/user-data/outputs/all_metrics.json",
        },
    )
    command = Command(update={"messages": [message], "artifact": "gallery"}, goto="next")

    result = middleware.wrap_tool_call(
        _request("nir_compare", {"nir_workflow": workflow}),
        lambda _: command,
    )

    assert isinstance(result, Command)
    assert result.goto == "next"
    assert result.update["artifact"] == "gallery"
    assert result.update["nir_workflow"]["stage"] == "review"
    assert result.update["nir_workflow"]["metrics_path"].endswith("all_metrics.json")


def test_knowledge_task_completes_after_successful_search() -> None:
    middleware = NIRWorkflowMiddleware()
    workflow = start_workflow(task_type="knowledge")

    result = middleware.wrap_tool_call(
        _request("nir_search_knowledge", {"nir_workflow": workflow}),
        lambda _: _result("nir_search_knowledge", {"results": [], "count": 0}),
    )

    assert isinstance(result, Command)
    assert result.update["nir_workflow"]["stage"] == "completed"
    assert result.update["nir_workflow"]["next_action"] == "none"


def test_successful_registration_automatically_enters_registered_stage() -> None:
    middleware = NIRWorkflowMiddleware()
    workflow = _execution_state()
    workflow = transition_workflow(workflow, action="record_attempt", attempt_passed=True)
    workflow = transition_workflow(workflow, action="approve")

    result = middleware.wrap_tool_call(
        _request("nir_register_model", {"nir_workflow": workflow}),
        lambda _: _result(
            "nir_register_model",
            {"status": "registered", "model_id": "corn-protein", "version": 1},
        ),
    )

    assert isinstance(result, Command)
    assert result.update["nir_workflow"]["stage"] == "registered"
    assert result.update["nir_workflow"]["next_action"] == "complete_workflow"


@pytest.mark.anyio
async def test_async_wrapper_enforces_the_same_policy() -> None:
    middleware = NIRWorkflowMiddleware()
    called = False

    async def handler(_: ToolCallRequest) -> ToolMessage:
        nonlocal called
        called = True
        return _result("nir_predict", {"status": "ok"})

    result = await middleware.awrap_tool_call(_request("nir_predict", {}), handler)

    assert called is False
    assert isinstance(result, ToolMessage)
    assert json.loads(result.content)["code"] == "nir_workflow_required"
