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
    retry_execution_signature,
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


def test_classification_workflow_requires_label_context_but_not_regression_unit():
    state = start_workflow(
        task_type="classification",
        data_path="/mnt/user-data/uploads/origin.csv",
    )

    assert state["stage"] == "data_audit"
    state = transition_workflow(
        state,
        action="record_audit",
        audit_passed=True,
        domain="food_authenticity",
        label_column="origin",
        validation_goal="internal_holdout",
    )

    assert state["stage"] == "planning"
    assert state["label_column"] == "origin"
    assert state["missing_inputs"] == []


def test_classification_workflow_asks_for_missing_label_column_after_audit():
    state = start_workflow(
        task_type="classification",
        data_path="/mnt/user-data/uploads/origin.csv",
        domain="food_authenticity",
        validation_goal="internal_holdout",
    )

    state = transition_workflow(state, action="record_audit", audit_passed=True)

    assert state["stage"] == "clarification"
    assert state["missing_inputs"] == ["label_column"]
    assert state["clarification_questions"][0]["fields"] == ["label_column"]


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


def test_high_assurance_placeholders_do_not_satisfy_required_context():
    state = start_workflow(
        task_type="calibration",
        data_path="mango.csv",
        analyte="DM",
        unit="%",
        domain="food_moisture",
        validation_goal="external_validation",
        instrument="unknown",
        grouping_column="none",
        reference_method="N/A",
    )

    state = transition_workflow(state, action="record_audit", audit_passed=True)

    assert state["stage"] == "clarification"
    assert state["missing_inputs"] == [
        "instrument",
        "grouping_column",
        "reference_method",
    ]
    assert state["instrument"] is None
    assert state["grouping_column"] is None
    assert state["reference_method"] is None


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


def test_unknown_high_assurance_requirements_do_not_advance_to_planning():
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

    assert state["stage"] == "clarification"
    assert state["audit_status"] == "passed"
    assert state["missing_inputs"] == [
        "instrument",
        "grouping_column",
        "reference_method",
    ]
    assert state["clarification_questions"][0]["fields"] == state["missing_inputs"]
    assert state["next_action"] == "ask_targeted_clarification"


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


def test_retry_execution_signature_ignores_transport_paths() -> None:
    steps, model_args, signature = retry_execution_signature(
        tool_name="nir_train_partitioned_model",
        method="auto",
        pipeline_steps=["snv"],
        model_args={
            "file_path": "/mnt/user-data/uploads/mango.csv",
            "model_output": "/mnt/user-data/outputs/retry.pkl",
            "metrics_output": "/mnt/user-data/outputs/retry.json",
            "x_cols": "309:1149",
            "y_col": "DM",
        },
    )
    _same_steps, same_model_args, same_signature = retry_execution_signature(
        tool_name="nir_train_partitioned_model",
        method="auto",
        pipeline_steps=["snv"],
        model_args={"x_cols": "309:1149", "y_col": "DM"},
    )

    assert steps == [{"method": "snv", "params": {}}]
    assert model_args == same_model_args == {"x_cols": "309:1149", "y_col": "DM"}
    assert signature == same_signature


def test_retry_execution_signature_normalizes_tool_defaults_and_aliases() -> None:
    _steps, _args, implicit_default = retry_execution_signature(
        tool_name="nir_train_model",
        method=None,
        pipeline_steps=["snv"],
    )
    _steps, _args, explicit_default = retry_execution_signature(
        tool_name="nir_train_model",
        method="pls",
        pipeline_steps=["snv"],
    )
    _steps, _args, cnn_alias = retry_execution_signature(
        tool_name="nir_train_model",
        method="1d-cnn",
        pipeline_steps=["snv"],
    )
    _steps, _args, canonical_cnn = retry_execution_signature(
        tool_name="nir_train_model",
        method="cnn",
        pipeline_steps=["snv"],
    )

    assert implicit_default == explicit_default
    assert cnn_alias == canonical_cnn


def test_failed_attempt_requires_reflection_and_bound_retry_plan_until_budget_is_exhausted():
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
    first_steps, _first_model_args, first_signature = retry_execution_signature(
        tool_name="nir_analyze",
        method="auto",
        pipeline_steps=["snv"],
    )
    state = transition_workflow(
        state,
        action="record_attempt",
        attempt_passed=False,
        grade="C",
        attempt_evidence={
            "tool_name": "nir_analyze",
            "method": "auto",
            "pipeline_steps": first_steps,
            "execution_signature": first_signature,
        },
    )

    assert state["stage"] == "evaluation"
    assert state["next_action"] == "reflect_on_attempt"
    with pytest.raises(NIRWorkflowError, match="planning"):
        transition_workflow(state, action="plan_ready")
    with pytest.raises(NIRWorkflowError, match="execution stage"):
        transition_workflow(state, action="record_attempt", attempt_passed=False)

    state = transition_workflow(
        state,
        action="record_reflection",
        reflection_result={
            "attempt": 1,
            "should_retry": True,
            "diagnostics": {"residual_trend": "curved"},
            "knowledge_hint": {"query": "curved NIR residual preprocessing"},
        },
    )
    assert state["stage"] == "knowledge"
    state = transition_workflow(state, action="knowledge_retrieved", evidence_ids=["paper-001"])
    with pytest.raises(NIRWorkflowError, match="materially change"):
        transition_workflow(
            state,
            action="record_retry_plan",
            retry_tool="nir_analyze",
            retry_method="auto",
            retry_pipeline_steps='["snv"]',
            retry_rationale="Repeat the prior pipeline.",
            expected_improvement="Improve R2.",
        )
    state = transition_workflow(
        state,
        action="record_retry_plan",
        retry_tool="nir_analyze",
        retry_method="auto",
        retry_pipeline_steps='["snv", {"method": "derivative1", "params": {"window": 11}}]',
        retry_rationale="Address curved residual structure with a first derivative.",
        expected_improvement="Reduce RMSEP and remove residual curvature.",
    )
    planned_signature = state["retry_plan"]["execution_signature"]
    assert state["retry_plan"]["evidence_ids"] == ["paper-001"]
    assert state["retry_plan"]["reflection_id"] == state["reflection"]["reflection_id"]
    state = transition_workflow(state, action="plan_ready")
    state = transition_workflow(
        state,
        action="record_attempt",
        attempt_passed=False,
        grade="D",
        attempt_evidence={
            "tool_name": "nir_analyze",
            "method": "auto",
            "pipeline_steps": state["retry_plan"]["pipeline_steps"],
            "execution_signature": planned_signature,
        },
    )
    assert state["stage"] == "evaluation"
    state = transition_workflow(
        state,
        action="record_reflection",
        reflection_result={
            "attempt": 2,
            "should_retry": True,
            "diagnostics": {"residual_trend": "curved"},
        },
    )

    assert state["attempt"] == 2
    assert state["stage"] == "blocked"
    assert state["next_action"] == "report_best_effort"
    assert state["reflection"]["stop_reason"] == "retry_budget_exhausted"


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
    state = transition_workflow(
        state,
        action="record_reflection",
        reflection_result={
            "attempt": 1,
            "should_retry": True,
            "knowledge_hint": {"query": "first retry"},
        },
    )
    state = transition_workflow(state, action="knowledge_retrieved", evidence_ids=["paper-001", "paper-002"])
    state = transition_workflow(
        state,
        action="record_retry_plan",
        retry_tool="nir_train_model",
        retry_method="pls",
        retry_pipeline_steps='["snv"]',
        retry_rationale="First evidence-backed retry.",
        expected_improvement="Improve RPD.",
    )
    signature = state["retry_plan"]["execution_signature"]
    state = transition_workflow(state, action="plan_ready")
    state = transition_workflow(
        state,
        action="record_attempt",
        attempt_passed=False,
        attempt_evidence={
            "tool_name": "nir_train_model",
            "method": "pls",
            "pipeline_steps": state["retry_plan"]["pipeline_steps"],
            "execution_signature": signature,
        },
    )
    state = transition_workflow(
        state,
        action="record_reflection",
        reflection_result={
            "attempt": 2,
            "should_retry": True,
            "knowledge_hint": {"query": "second retry"},
        },
    )
    state = transition_workflow(state, action="knowledge_retrieved", evidence_ids=["paper-002", "paper-003"])

    assert state["knowledge_evidence"] == ["paper-001", "paper-002", "paper-003"]


def test_required_retry_knowledge_with_no_evidence_stops_instead_of_guessing() -> None:
    state = start_workflow(
        task_type="analysis",
        data_path="data.npz",
        analyte="moisture",
        unit="%",
        domain="food_moisture",
        validation_goal="internal_holdout",
    )
    state = transition_workflow(state, action="record_audit", audit_passed=True)
    state = transition_workflow(state, action="plan_ready")
    state = transition_workflow(state, action="record_attempt", attempt_passed=False)
    state = transition_workflow(
        state,
        action="record_reflection",
        reflection_result={
            "attempt": 1,
            "should_retry": True,
            "knowledge_hint": {"query": "NIR moisture retry evidence"},
        },
    )

    state = transition_workflow(state, action="knowledge_retrieved", evidence_ids=[])

    assert state["stage"] == "blocked"
    assert state["next_action"] == "report_best_effort"
    assert state["history"][-1]["outcome"] == "required_retry_evidence_unavailable"


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


def test_agent_view_exposes_bounded_attempt_evidence_for_grounded_summary():
    state = start_workflow(
        task_type="calibration",
        data_path="mango.csv",
        validation_goal="external_validation",
    )
    state["attempts"] = [
        {
            "attempt": index,
            "passed": False,
            "protocol": "named_partition_external_validation",
            "validation_scope": "independent_external_validation",
            "metrics_summary": {"external": {"R2": index / 10}},
            "model_path": f"/mnt/user-data/outputs/model-v{index}.pkl",
            "metrics_path": f"/mnt/user-data/outputs/metrics-v{index}.json",
            "execution_signature": f"secret-{index}",
        }
        for index in range(1, 13)
    ]

    projected = workflow_agent_view(state)

    assert len(projected["attempts"]) == 10
    assert projected["attempts"][0]["attempt"] == 3
    assert projected["attempts"][-1]["metrics_summary"]["external"]["R2"] == 1.2
    assert "execution_signature" not in projected["attempts"][-1]


def test_agent_view_preserves_durable_data_audit_evidence() -> None:
    state = start_workflow(
        task_type="calibration",
        data_path="mango.csv",
        validation_goal="external_validation",
    )
    state = transition_workflow(
        state,
        action="record_audit_evidence",
        audit_evidence={
            "raw_wavelength_range": [285.0, 1200.0],
            "usable_wavelength_range": [309.0, 1149.0],
            "constant_wavelength_count": 25,
            "usable_wavelength_count": 281,
        },
    )

    projected = workflow_agent_view(state)

    assert projected["audit_evidence"]["raw_wavelength_range"] == [285.0, 1200.0]
    assert projected["audit_evidence"]["usable_wavelength_range"] == [309.0, 1149.0]


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
        "response_guard_events": [
            {
                "stage": "data_audit",
                "message_id": "answer-unsafe",
                "violations": ["metric_not_in_current_evidence:r2"],
            }
        ],
        "run_ids": ["run-lower"],
        "history": [*existing["history"], {"action": "tool_call", "tool": "nir_inspect"}],
    }
    progressed = transition_workflow(existing, action="record_audit", audit_passed=True)

    merged = merge_nir_workflow(observed, progressed)

    assert merged is not None
    assert merged["revision"] == progressed["revision"]
    assert merged["stage"] == "planning"
    assert merged["tool_observations"] == [{"name": "nir_inspect", "status": "success"}]
    assert merged["response_guard_events"][0]["message_id"] == "answer-unsafe"
    assert merged["run_ids"] == ["run-lower"]


def test_nir_workflow_reducer_prefers_richer_data_audit_evidence() -> None:
    state = start_workflow(task_type="calibration", data_path="mango.csv")
    inspected = {
        **state,
        "revision": 2,
        "audit_evidence": {
            "source_tool": "nir_inspect",
            "raw_wavelength_range": [285.0, 1200.0],
        },
    }
    loaded = {
        **state,
        "revision": 2,
        "audit_evidence": {
            "source_tool": "nir_load_data",
            "raw_wavelength_range": [285.0, 1200.0],
            "usable_wavelength_range": [309.0, 1149.0],
            "constant_wavelength_count": 25,
            "usable_wavelength_count": 281,
        },
    }

    merged = merge_nir_workflow(inspected, loaded)

    assert merged is not None
    assert merged["audit_evidence"] == loaded["audit_evidence"]


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
