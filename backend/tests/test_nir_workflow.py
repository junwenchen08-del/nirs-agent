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
    workflow_agent_view,
)


def test_start_audits_data_before_asking_inferable_professional_requirements():
    state = start_workflow(task_type="calibration", data_path="/mnt/user-data/uploads/corn.npz")

    assert state["stage"] == "data_audit"
    assert state["missing_inputs"] == []
    assert state["clarification_questions"] == []
    assert state["next_action"] == "inspect_data"


def test_passed_audit_requests_only_missing_decision_context():
    state = start_workflow(task_type="calibration", data_path="/mnt/user-data/uploads/corn.npz")

    state = transition_workflow(
        state,
        action="record_audit",
        audit_passed=True,
        domain="food_protein",
        analyte="protein",
        unit="%",
    )

    assert state["stage"] == "clarification"
    assert state["audit_status"] == "passed"
    assert state["missing_inputs"] == ["validation_goal"]
    assert state["next_action"] == "ask_targeted_clarification"
    assert state["clarification_questions"] == [
        {
            "fields": ["validation_goal"],
            "question": ("这次任务的验证目标是什么：探索分析、同一数据集独立留出、独立外部验证，还是生产部署？"),
            "reason": "验证目标决定数据划分、质量声明和是否需要额外域信息。",
        }
    ]


def test_external_validation_requests_high_impact_context_in_one_question():
    state = start_workflow(
        task_type="calibration",
        data_path="corn.npz",
        analyte="protein",
        unit="%",
        domain="food_protein",
        validation_goal="external_validation",
    )

    state = transition_workflow(state, action="record_audit", audit_passed=True)

    assert state["stage"] == "clarification"
    assert state["missing_inputs"] == [
        "instrument",
        "grouping_column",
        "reference_method",
    ]
    assert len(state["clarification_questions"]) == 1
    assert state["clarification_questions"][0]["fields"] == [
        "instrument",
        "grouping_column",
        "reference_method",
    ]


@pytest.mark.parametrize(
    ("provided", "expected"),
    [
        ("exploratory analysis", "exploratory"),
        ("探索分析", "exploratory"),
        ("same-dataset holdout", "internal_holdout"),
        ("外部验证", "external_validation"),
        ("生产部署", "production"),
    ],
)
def test_validation_goal_aliases_are_persisted_canonically(provided, expected):
    state = start_workflow(
        task_type="calibration",
        data_path="corn.npz",
        validation_goal=provided,
    )

    assert state["validation_goal"] == expected


def test_unsupported_validation_goal_is_rejected():
    with pytest.raises(NIRWorkflowError, match="Unsupported validation_goal"):
        start_workflow(
            task_type="calibration",
            data_path="corn.npz",
            validation_goal="make it production-ish",
        )


def test_exploratory_plan_routes_to_findings_without_modeling():
    state = start_workflow(
        task_type="calibration",
        data_path="corn.npz",
        analyte="protein",
        unit="%",
        domain="food_protein",
        validation_goal="exploratory",
    )
    state = transition_workflow(state, action="record_audit", audit_passed=True)

    state = transition_workflow(state, action="plan_ready")

    assert state["stage"] == "completed"
    assert state["next_action"] == "none"
    assert state["approval_status"] == "not_required"
    assert state["history"][-1]["outcome"] == "exploratory_findings_ready"


def test_requirements_after_passed_audit_advance_without_repeating_inspection():
    state = start_workflow(
        task_type="calibration",
        data_path="corn.npz",
        analyte="protein",
        unit="%",
        domain="food_protein",
        validation_goal="production",
    )
    state = transition_workflow(state, action="record_audit", audit_passed=True)

    state = transition_workflow(
        state,
        action="set_requirements",
        instrument="unknown",
        grouping_column="none",
        reference_method="unknown",
    )

    assert state["stage"] == "planning"
    assert state["audit_status"] == "passed"
    assert state["missing_inputs"] == []
    assert state["clarification_questions"] == []
    assert state["next_action"] == "prepare_analysis_plan"


def test_start_multi_modeling_enters_data_audit_with_complete_requirements():
    state = start_workflow(
        task_type="multi_modeling",
        data_path="multi.npz",
        analyte="protein, moisture, oil",
        unit="percent",
        domain="feed",
        validation_goal="internal_holdout",
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
        validation_goal="internal_holdout",
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
        validation_goal="internal_holdout",
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
        validation_goal="internal_holdout",
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
        validation_goal="internal_holdout",
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
    state = start_workflow(task_type="calibration", data_path="data.npz")
    state = transition_workflow(
        state,
        action="record_audit",
        audit_passed=True,
        domain="food_protein",
        analyte="protein",
        unit="%",
    )

    rendered = _render_durable_context_data(None, [], [], state)

    assert "Active NIR workflow state" in rendered
    assert '"stage": "clarification"' in rendered
    assert '"next_action": "ask_targeted_clarification"' in rendered
    assert '"fields": ["validation_goal"]' in rendered


def test_agent_view_omits_history_and_trace_identifiers():
    state = start_workflow(
        task_type="calibration",
        data_path="corn.npz",
        validation_goal="exploratory",
    )
    state["run_ids"] = ["run-secret"]
    state["trace_ids"] = ["trace-secret"]
    state["tool_observations"] = [
        {
            "name": "nir_inspect",
            "status": "success",
            "call_id": "call-secret",
            "run_id": "run-secret",
            "trace_id": "trace-secret",
        }
    ]

    projected = workflow_agent_view(state)

    assert "history" not in projected
    assert "run_ids" not in projected
    assert "trace_ids" not in projected
    assert "tool_observations" not in projected
    assert projected["tool_observation_count"] == 1
    assert projected["last_tool_observation"] == {
        "name": "nir_inspect",
        "status": "success",
    }


def test_agent_view_bounds_multi_target_attempt_evidence():
    state = start_workflow(
        task_type="multi_modeling",
        data_path="multi.npz",
        validation_goal="internal_holdout",
    )
    state["attempt_evidence"] = {
        "schema_version": 1,
        "metrics_summary": {
            "component_count": 12,
            "per_component": [{"name": f"target-{index}", "R2_val": 0.9} for index in range(12)],
        },
    }

    projected = workflow_agent_view(state)

    assert projected["attempt_evidence"]["metrics_summary"]["component_count"] == 12
    assert len(projected["attempt_evidence"]["metrics_summary"]["per_component"]) == 5


def test_workflow_tool_returns_compact_agent_view_but_persists_full_state():
    state = start_workflow(
        task_type="calibration",
        data_path="corn.npz",
        validation_goal="exploratory",
    )
    state["history"] = [{"action": "large-event", "notes": "x" * 10_000}]
    state["tool_observations"] = [
        {
            "name": "nir_inspect",
            "status": "success",
            "call_id": "call-secret",
            "run_id": "run-secret",
        }
    ]
    runtime = SimpleNamespace(state={"nir_workflow": state}, tool_call_id="call-status")

    command = nir_workflow_tool.func(runtime=runtime, action="status")

    payload = json.loads(command.update["messages"][0].content)
    assert len(command.update["messages"][0].content) < 2_000
    assert "history" not in payload["workflow"]
    assert "tool_observations" not in payload["workflow"]
    assert payload["workflow"]["tool_observation_count"] == 1
    assert "nir_workflow" not in command.update


def _review_state():
    state = start_workflow(
        task_type="calibration",
        data_path="data.npz",
        analyte="protein",
        unit="%",
        domain="food_protein",
        validation_goal="internal_holdout",
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
        validation_goal="internal_holdout",
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
