"""Tests for the durable NIR agent workflow control plane."""

import json
from types import SimpleNamespace
from typing import get_type_hints

import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from deerflow.agents.middlewares.durable_context_middleware import (
    _render_durable_context_data,
)
from deerflow.agents.thread_state import ThreadState, merge_nir_workflow
from deerflow.community.nir.modeling import nir_register_model_tool
from deerflow.community.nir.workflow import (
    NIRWorkflowError,
    nir_workflow_tool,
    start_workflow,
    transition_workflow,
)


def test_start_collects_missing_professional_requirements():
    state = start_workflow(task_type="calibration", data_path="/mnt/user-data/uploads/corn.npz")

    assert state["stage"] == "intake"
    assert state["missing_inputs"] == ["analyte", "unit", "domain"]
    assert state["next_action"] == "collect_requirements"


def test_start_multi_modeling_enters_data_audit_with_complete_requirements():
    state = start_workflow(
        task_type="multi_modeling",
        data_path="multi.npz",
        analyte="protein, moisture, oil",
        unit="percent",
        domain="feed",
    )

    assert state["stage"] == "data_audit"
    assert state["missing_inputs"] == []


def test_complete_calibration_requires_explicit_approval_before_registration():
    state = start_workflow(
        task_type="calibration",
        data_path="/mnt/user-data/uploads/corn.npz",
        analyte="protein",
        unit="%",
        domain="food_protein",
    )
    assert state["stage"] == "data_audit"

    state = transition_workflow(state, action="record_audit", audit_passed=True)
    state = transition_workflow(state, action="plan_ready")
    state = transition_workflow(
        state,
        action="record_attempt",
        attempt_passed=True,
        grade="A",
        model_path="/mnt/user-data/outputs/model.pkl",
        metrics_path="/mnt/user-data/outputs/metrics.json",
    )

    assert state["stage"] == "review"
    assert state["approval_status"] == "pending"
    with pytest.raises(NIRWorkflowError, match="approved workflow"):
        transition_workflow(state, action="registered")

    state = transition_workflow(state, action="approve", notes="User approved model")
    state = transition_workflow(state, action="registered")
    state = transition_workflow(state, action="complete")

    assert state["stage"] == "completed"
    assert state["approval_status"] == "approved"


def test_failed_attempt_uses_knowledge_stage_until_retry_budget_is_exhausted():
    state = start_workflow(
        task_type="analysis",
        data_path="data.npz",
        analyte="moisture",
        unit="%",
        domain="food_moisture",
        max_attempts=2,
    )
    state = transition_workflow(state, action="record_audit", audit_passed=True)
    state = transition_workflow(state, action="plan_ready")
    state = transition_workflow(state, action="record_attempt", attempt_passed=False, grade="C")

    assert state["stage"] == "knowledge"
    assert state["next_action"] == "retrieve_evidence_for_retry"

    state = transition_workflow(state, action="knowledge_retrieved")
    state = transition_workflow(state, action="plan_ready")
    state = transition_workflow(state, action="record_attempt", attempt_passed=False, grade="D")

    assert state["attempt"] == 2
    assert state["stage"] == "blocked"
    assert state["next_action"] == "report_best_effort"


def test_knowledge_evidence_accumulates_across_bounded_retries():
    state = start_workflow(
        task_type="calibration",
        data_path="data.npz",
        analyte="protein",
        unit="%",
        domain="food_protein",
        max_attempts=3,
    )
    state = transition_workflow(state, action="record_audit", audit_passed=True)
    state = transition_workflow(state, action="plan_ready")
    state = transition_workflow(state, action="record_attempt", attempt_passed=False)
    state = transition_workflow(state, action="knowledge_retrieved", evidence_ids=["paper-001", "paper-002"])
    state = transition_workflow(state, action="plan_ready")
    state = transition_workflow(state, action="record_attempt", attempt_passed=False)
    state = transition_workflow(state, action="knowledge_retrieved", evidence_ids=["paper-002", "paper-003"])

    assert state["knowledge_evidence"] == ["paper-001", "paper-002", "paper-003"]


def test_nir_workflow_reducer_merges_same_revision_sibling_tool_observations():
    existing = start_workflow(task_type="inspection", data_path="data.npz")
    first = {
        **existing,
        "tool_observations": [
            {
                "name": "nir_inspect",
                "status": "success",
                "stage_before": "data_audit",
                "stage_after": "data_audit",
                "code": None,
                "call_id": "call-1",
                "run_id": "run-1",
                "trace_id": "trace-1",
            }
        ],
        "run_ids": ["run-1"],
        "trace_ids": ["trace-1"],
        "history": [*existing["history"], {"action": "tool_call", "tool": "nir_inspect"}],
    }
    second = {
        **existing,
        "tool_observations": [
            {
                "name": "nir_load_data",
                "status": "success",
                "stage_before": "data_audit",
                "stage_after": "data_audit",
                "code": None,
                "call_id": "call-2",
                "run_id": "run-1",
                "trace_id": "trace-1",
            }
        ],
        "run_ids": ["run-1"],
        "trace_ids": ["trace-1"],
        "history": [*existing["history"], {"action": "tool_call", "tool": "nir_load_data"}],
    }

    merged = merge_nir_workflow(first, second)

    assert merged is not None
    assert [item["name"] for item in merged["tool_observations"]] == ["nir_inspect", "nir_load_data"]
    assert merged["run_ids"] == ["run-1"]
    assert merged["trace_ids"] == ["trace-1"]


def test_nir_workflow_reducer_prefers_progressed_same_revision_state():
    existing = start_workflow(
        task_type="calibration",
        data_path="data.npz",
        analyte="protein",
        unit="%",
        domain="food_protein",
    )
    denied_tool_update = {
        **existing,
        "tool_observations": [{"name": "nir_train_model", "status": "error"}],
        "history": [*existing["history"], {"action": "tool_call", "tool": "nir_train_model"}],
    }
    audit_update = {
        **transition_workflow(existing, action="record_audit", audit_passed=True),
        "revision": existing["revision"],
    }

    merged = merge_nir_workflow(denied_tool_update, audit_update)

    assert merged is not None
    assert merged["stage"] == "planning"
    assert merged["next_action"] == "prepare_analysis_plan"
    assert merged["tool_observations"] == [{"name": "nir_train_model", "status": "error"}]


def test_nir_workflow_reducer_collapses_same_revision_project_swap():
    existing = start_workflow(task_type="inspection", data_path="data.npz", project_id="project-a")
    conflicting = {
        **existing,
        "project_id": "project-b",
        "tool_observations": [{"name": "nir_inspect", "status": "success"}],
    }

    merged = merge_nir_workflow(existing, conflicting)

    assert merged is not None
    assert merged["project_id"] == "project-a"
    assert merged["tool_observations"] == [{"name": "nir_inspect", "status": "success"}]
    assert merged["history"][-1] == {
        "action": "project_conflict_merged",
        "kept_project_id": "project-a",
        "dropped_project_id": "project-b",
        "revision": existing["revision"],
    }


def test_thread_state_wires_nir_workflow_reducer():
    hints = get_type_hints(ThreadState, include_extras=True)
    assert merge_nir_workflow in hints["nir_workflow"].__metadata__


def test_tool_returns_checkpoint_update_and_structured_message():
    runtime = SimpleNamespace(state={}, tool_call_id="call-1")

    command = nir_workflow_tool.func(
        runtime=runtime,
        action="start",
        task_type="inspection",
        data_path="/mnt/user-data/uploads/sample.csv",
    )

    state = command.update["nir_workflow"]
    assert state["stage"] == "data_audit"
    message = command.update["messages"][0]
    assert isinstance(message, ToolMessage)
    payload = json.loads(message.content)
    assert payload["status"] == "ok"
    assert payload["next_action"] == "inspect_data"


def test_workflow_tool_associates_runtime_run_and_trace_ids():
    runtime = SimpleNamespace(
        state={},
        context={"run_id": "run-start", "deerflow_trace_id": "trace-start"},
        tool_call_id="call-context",
    )

    command = nir_workflow_tool.func(
        runtime=runtime,
        action="start",
        task_type="inspection",
        data_path="sample.csv",
    )

    state = command.update["nir_workflow"]
    assert state["run_ids"] == ["run-start"]
    assert state["trace_ids"] == ["trace-start"]


def test_tool_reports_invalid_transition_without_mutating_state():
    state = start_workflow(task_type="inspection", data_path="data.npz")
    runtime = SimpleNamespace(state={"nir_workflow": state}, tool_call_id="call-2")

    command = nir_workflow_tool.func(runtime=runtime, action="registered")

    assert "nir_workflow" not in command.update
    message = command.update["messages"][0]
    assert message.status == "error"
    assert json.loads(message.content)["status"] == "error"


def test_workflow_state_is_rendered_into_durable_context():
    state = start_workflow(task_type="inspection", data_path="data.npz")

    rendered = _render_durable_context_data(None, [], [], state)

    assert "Active NIR workflow state" in rendered
    assert '"stage": "data_audit"' in rendered
    assert '"next_action": "inspect_data"' in rendered


def _review_state():
    state = start_workflow(
        task_type="calibration",
        data_path="data.npz",
        analyte="protein",
        unit="%",
        domain="food_protein",
    )
    state = transition_workflow(state, action="record_audit", audit_passed=True)
    state = transition_workflow(state, action="plan_ready")
    return transition_workflow(
        state,
        action="record_attempt",
        attempt_passed=True,
        model_path="model.pkl",
        metrics_path="metrics.json",
    )


def test_approve_tool_rejects_missing_user_authorization():
    state = _review_state()
    runtime = SimpleNamespace(
        state={
            "nir_workflow": state,
            "messages": [HumanMessage(content="模型效果怎么样？")],
        },
        tool_call_id="call-approve-1",
    )

    command = nir_workflow_tool.func(runtime=runtime, action="approve")

    assert "nir_workflow" not in command.update
    assert json.loads(command.update["messages"][0].content)["status"] == "error"


@pytest.mark.parametrize(
    "message",
    [
        "我不批准这个模型",
        "我不同意使用这个模型",
        "This model is not approved",
    ],
)
def test_approve_tool_rejects_explicit_denial(message: str):
    state = _review_state()
    runtime = SimpleNamespace(
        state={"nir_workflow": state, "messages": [HumanMessage(content=message)]},
        tool_call_id="call-approve-denied",
    )

    command = nir_workflow_tool.func(runtime=runtime, action="approve")

    assert "nir_workflow" not in command.update
    assert json.loads(command.update["messages"][0].content)["status"] == "error"


def test_approve_tool_accepts_explicit_user_authorization():
    state = _review_state()
    runtime = SimpleNamespace(
        state={
            "nir_workflow": state,
            "messages": [HumanMessage(content="我确认采用并注册这个模型")],
        },
        tool_call_id="call-approve-2",
    )

    command = nir_workflow_tool.func(runtime=runtime, action="approve")

    assert command.update["nir_workflow"]["stage"] == "approved"
    assert command.update["nir_workflow"]["approval_status"] == "approved"


def test_nir_workflow_reducer_preserves_evidence_from_lower_revision():
    existing = start_workflow(
        task_type="calibration",
        data_path="data.npz",
        analyte="protein",
        unit="%",
        domain="food_protein",
    )
    observed = {
        **existing,
        "tool_observations": [{"name": "nir_inspect", "status": "success"}],
        "run_ids": ["run-lower"],
        "history": [*existing["history"], {"action": "tool_call", "tool": "nir_inspect"}],
    }
    progressed = transition_workflow(existing, action="record_audit", audit_passed=True)

    merged = merge_nir_workflow(observed, progressed)

    assert merged is not None
    assert merged["revision"] == progressed["revision"]
    assert merged["stage"] == "planning"
    assert merged["tool_observations"] == [{"name": "nir_inspect", "status": "success"}]
    assert merged["run_ids"] == ["run-lower"]


def test_model_registration_rejects_unapproved_workflow_before_file_access():
    state = _review_state()
    runtime = SimpleNamespace(state={"nir_workflow": state})

    result = nir_register_model_tool.func(
        runtime=runtime,
        model_id="corn-protein",
        model_path="missing-model.pkl",
        metrics_path="missing-metrics.json",
    )

    payload = json.loads(result)
    assert payload["status"] == "error"
    assert "approved NIR workflow" in payload["error"]


def test_starting_after_terminal_workflow_advances_revision():
    previous = start_workflow(task_type="inspection", data_path="old.npz")
    previous = transition_workflow(previous, action="record_audit", audit_passed=True)
    previous = transition_workflow(previous, action="plan_ready")
    previous = transition_workflow(previous, action="complete")
    runtime = SimpleNamespace(
        state={"nir_workflow": previous},
        tool_call_id="call-new-project",
    )

    command = nir_workflow_tool.func(
        runtime=runtime,
        action="start",
        task_type="inspection",
        data_path="new.npz",
    )

    new_state = command.update["nir_workflow"]
    assert new_state["project_id"] != previous["project_id"]
    assert new_state["revision"] == previous["revision"] + 1
    assert merge_nir_workflow(previous, new_state) == new_state
