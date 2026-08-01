"""Grounding checks for NIR final responses."""

from __future__ import annotations

import json
from types import SimpleNamespace

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from deerflow.agents.middlewares.nir_workflow_middleware import NIRWorkflowMiddleware
from deerflow.agents.thread_state import ThreadState
from deerflow.community.nir.response_grounding import (
    NIRStreamMessageGate,
    render_grounded_nir_response,
    validate_nir_response,
)
from deerflow.community.nir.workflow import start_workflow, transition_workflow


class _FakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):  # noqa: ARG002
        return self


@tool("nir_reflect")
def _fake_nir_reflect(
    metrics_path: str,
    history: str,
    domain: str,
    attempt: int,
    max_retries: int,
) -> str:
    """Return a deterministic stop reflection for graph integration tests."""

    assert metrics_path.endswith("metrics.json")
    assert history == "[]"
    assert domain == "food_protein"
    assert max_retries == 3
    return json.dumps(
        {
            "attempt": attempt,
            "should_retry": False,
            "reason": "No evidence-backed retry is available.",
            "diagnostics": {"residual_trend": "none"},
        }
    )


def _review_workflow(
    *,
    validation_goal: str = "internal_holdout",
    passed: bool = True,
) -> dict:
    state = start_workflow(
        task_type="calibration",
        data_path="/mnt/user-data/uploads/calibration.csv",
        analyte="protein",
        unit="%",
        domain="food_protein",
        validation_goal=validation_goal,
        instrument="instrument-1" if validation_goal in {"external_validation", "production"} else None,
        grouping_column="batch" if validation_goal in {"external_validation", "production"} else None,
        reference_method="laboratory reference" if validation_goal in {"external_validation", "production"} else None,
    )
    state = transition_workflow(state, action="record_audit", audit_passed=True)
    state = transition_workflow(state, action="plan_ready")
    scope = "independent_external_validation" if validation_goal in {"external_validation", "production"} else "independent_holdout_not_external"
    protocol = "named_partition_external_validation" if validation_goal in {"external_validation", "production"} else "deterministic_auto_split_holdout"
    return transition_workflow(
        state,
        action="record_attempt",
        attempt_passed=passed,
        grade="A" if passed else "D",
        model_path="/mnt/user-data/outputs/model.pkl",
        metrics_path="/mnt/user-data/outputs/metrics.json",
        attempt_evidence={
            "schema_version": 1,
            "tool_name": "nir_train_partitioned_model" if validation_goal in {"external_validation", "production"} else "nir_train_auto_split_model",
            "protocol": protocol,
            "validation_scope": scope,
            "model_path": "/mnt/user-data/outputs/model.pkl",
            "metrics_path": "/mnt/user-data/outputs/metrics.json",
            "model_sha256": "a" * 64,
            "metrics_sha256": "b" * 64,
            "training_data_sha256": "c" * 64,
            "metrics_summary": {
                "R2_val": 0.91234,
                "RPD": 3.4567,
                "RMSEP": 0.12345,
                "grade": "A" if passed else "D",
                "passed": passed,
            },
        },
    )


def test_internal_holdout_response_accepts_rounded_current_metrics_and_negated_external_claim() -> None:
    verdict = validate_nir_response(
        "本次独立留出测试 R²=0.91，RPD=3.46，RMSEP=0.123。这不是外部验证。",
        _review_workflow(),
    )

    assert verdict.passed is True
    assert verdict.violations == ()


def test_external_validation_response_accepts_current_metrics_and_scope() -> None:
    verdict = validate_nir_response(
        "本次独立外部验证 R²=0.912，RPD=3.457，RMSEP=0.1235。",
        _review_workflow(validation_goal="external_validation"),
    )

    assert verdict.passed is True


def test_response_without_attempt_evidence_cannot_claim_metrics() -> None:
    workflow = start_workflow(
        task_type="inspection",
        data_path="/mnt/user-data/uploads/data.csv",
    )

    verdict = validate_nir_response("此前模型 R²=0.88。", workflow)

    assert verdict.passed is False
    assert verdict.violations == (
        "workflow_incomplete:inspect_data",
        "metric_not_in_current_evidence:r2",
    )


def test_response_rejects_metric_value_not_present_in_current_attempt() -> None:
    verdict = validate_nir_response(
        "模型表现很好，R²=0.99，RPD=5.20。",
        _review_workflow(),
    )

    assert verdict.passed is False
    assert "metric_value_mismatch:r2" in verdict.violations
    assert "metric_value_mismatch:rpd" in verdict.violations


def test_response_accepts_bounded_metrics_and_artifacts_from_prior_attempts() -> None:
    workflow = _review_workflow(validation_goal="external_validation", passed=False)
    workflow["stage"] = "blocked"
    workflow["next_action"] = "report_best_effort"
    workflow["attempts"] = [
        {
            "attempt": 1,
            "passed": False,
            "grade": "D",
            "protocol": "named_partition_external_validation",
            "validation_scope": "independent_external_validation",
            "metrics_summary": {
                "external": {"R2": 0.8326, "RMSE": 1.092, "RPD": 2.44, "bias": -0.021},
                "passed": False,
            },
            "model_path": "/mnt/user-data/outputs/model-v1.pkl",
            "metrics_path": "/mnt/user-data/outputs/metrics-v1.json",
        },
        {
            "attempt": 2,
            "passed": False,
            "grade": "D",
            "protocol": "named_partition_external_validation",
            "validation_scope": "independent_external_validation",
            "metrics_summary": {
                "external": {"R2": 0.7083, "RMSE": 1.441, "RPD": 1.85, "bias": -0.216},
                "passed": False,
            },
            "model_path": "/mnt/user-data/outputs/model-v2.pkl",
            "metrics_path": "/mnt/user-data/outputs/metrics-v2.json",
        },
    ]

    verdict = validate_nir_response(
        "独立外部验证共两次：尝试1 R²=0.8326、RMSE=1.092、RPD=2.44、偏差=-0.021，"
        "产物 `/mnt/user-data/outputs/model-v1.pkl` 和 `/mnt/user-data/outputs/metrics-v1.json`；"
        "尝试2 R²=0.7083、RMSE=1.441、RPD=1.85、偏差=-0.216。两次质量门禁均未通过。",
        workflow,
    )

    assert verdict.passed is True
    assert verdict.violations == ()


def test_response_still_rejects_metric_absent_from_all_attempt_evidence() -> None:
    workflow = _review_workflow(validation_goal="external_validation", passed=False)
    workflow["stage"] = "blocked"
    workflow["next_action"] = "report_best_effort"
    workflow["attempts"] = [
        {
            "attempt": 1,
            "metrics_summary": {"external": {"R2": 0.8326}},
            "model_path": "/mnt/user-data/outputs/model-v1.pkl",
            "metrics_path": "/mnt/user-data/outputs/metrics-v1.json",
        }
    ]

    verdict = validate_nir_response(
        "独立外部验证尝试1 R²=0.9999。",
        workflow,
    )

    assert verdict.passed is False
    assert "metric_value_mismatch:r2" in verdict.violations


def test_response_accepts_quality_thresholds_recorded_with_attempt() -> None:
    workflow = _review_workflow(validation_goal="external_validation", passed=False)
    workflow["stage"] = "blocked"
    workflow["next_action"] = "report_best_effort"
    workflow["attempt_evidence"]["metrics_summary"]["thresholds_used"] = {
        "min_r2": 0.90,
        "min_rpd": 4.0,
    }
    workflow["attempts"][0]["metrics_summary"]["thresholds_used"] = {
        "min_r2": 0.90,
        "min_rpd": 4.0,
    }

    verdict = validate_nir_response(
        "本次独立外部验证未通过；质量门槛为 R²=0.90、RPD=4.0。",
        workflow,
    )

    assert verdict.passed is True


def test_response_validates_raw_and_usable_wavelength_audit_evidence() -> None:
    workflow = _review_workflow()
    workflow["audit_evidence"] = {
        "raw_wavelength_range": [285.0, 1200.0],
        "usable_wavelength_range": [309.0, 1149.0],
        "constant_wavelength_count": 25,
        "usable_wavelength_count": 281,
    }

    supported = validate_nir_response(
        "原始采集光谱范围为 285–1200 nm，实际可用光谱范围为 309–1149 nm；共有 25 个恒定波长列和 281 个可用波长。",
        workflow,
    )
    unsupported = validate_nir_response(
        "原始采集光谱范围为 285–1200 nm，实际可用光谱范围为 285–1200 nm；constant_wavelength_count=0，共有 306 个可用波长。",
        workflow,
    )

    assert supported.passed is True
    assert "wavelength_range_mismatch:usable" in unsupported.violations
    assert "wavelength_count_mismatch:constant" in unsupported.violations
    assert "wavelength_count_mismatch:usable" in unsupported.violations


def test_internal_holdout_response_rejects_external_validation_overclaim() -> None:
    verdict = validate_nir_response(
        "该模型已经完成独立外部验证，可以用于外部样品。",
        _review_workflow(),
    )

    assert verdict.passed is False
    assert "validation_scope_overclaim:external" in verdict.violations


def test_response_rejects_unapproved_model_artifact_path() -> None:
    verdict = validate_nir_response(
        "已生成模型 /mnt/user-data/outputs/other.pkl。",
        _review_workflow(),
    )

    assert verdict.passed is False
    assert "artifact_path_mismatch:model" in verdict.violations


def test_failed_attempt_rejects_positive_quality_claim() -> None:
    verdict = validate_nir_response(
        "当前模型已经通过质量门禁并达到应用要求。",
        _review_workflow(passed=False),
    )

    assert verdict.passed is False
    assert "quality_overclaim:passed" in verdict.violations


def test_failed_attempt_cannot_publish_before_required_reflection() -> None:
    workflow = _review_workflow(passed=False)

    verdict = validate_nir_response(
        "本次内部留出测试 R²=0.91，RPD=3.46；这不是外部验证。",
        workflow,
    )

    assert verdict.passed is False
    assert "workflow_incomplete:reflect_on_attempt" in verdict.violations


def test_middleware_reengages_model_when_failed_attempt_skips_reflection() -> None:
    workflow = _review_workflow(passed=False)
    message = AIMessage(
        content="本次内部留出测试 R²=0.91，RPD=3.46；这不是外部验证。",
        id="premature-answer",
    )

    update = NIRWorkflowMiddleware().after_model(
        {"messages": [message], "nir_workflow": workflow},
        SimpleNamespace(context={"run_id": "run-mango"}),
    )

    assert update is not None
    assert update["jump_to"] == "model"
    reminder = update["messages"][0]
    assert isinstance(reminder, HumanMessage)
    assert reminder.name == "nir_workflow_completion_reminder"
    assert reminder.additional_kwargs["hide_from_ui"] is True
    assert "nir_reflect" in reminder.content
    assert update["nir_workflow"]["response_guard_events"][-1]["violations"] == ["workflow_incomplete:reflect_on_attempt"]


def test_graph_forces_reflection_before_releasing_failed_attempt_summary() -> None:
    workflow = _review_workflow(passed=False)
    agent = create_agent(
        model=_FakeModel(
            disable_streaming=True,
            responses=[
                AIMessage(
                    content="本次内部留出测试 R²=0.91，RPD=3.46；这不是外部验证。",
                    id="premature-graph-answer",
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "id": "reflect-call",
                            "name": "nir_reflect",
                            "args": {},
                        }
                    ],
                ),
                AIMessage(
                    content=("本次内部留出测试 R²=0.91，RPD=3.46，质量门禁未通过；这不是外部验证。"),
                    id="grounded-graph-answer",
                ),
            ],
        ),
        tools=[_fake_nir_reflect],
        middleware=[NIRWorkflowMiddleware()],
        state_schema=ThreadState,
    )

    result = agent.invoke(
        {
            "messages": [{"role": "user", "content": "请总结失败结果"}],
            "nir_workflow": workflow,
        }
    )

    assert result["nir_workflow"]["stage"] == "blocked"
    assert result["nir_workflow"]["reflection"]["should_retry"] is False
    assert any(event["action"] == "record_reflection" for event in result["nir_workflow"]["history"])
    assert result["messages"][-1].id == "grounded-graph-answer"


def test_response_rejects_unsupported_constant_column_removal_claim() -> None:
    workflow = _review_workflow()
    workflow["attempt_evidence"]["result_facts"] = {
        "wavelength_selection": {
            "n_original": 306,
            "n_selected": 306,
        }
    }

    verdict = validate_nir_response(
        "25 个恒定波长列已经自动排除。本次内部留出测试 R²=0.91；这不是外部验证。",
        workflow,
    )

    assert "unsupported_preprocessing_claim:constant_columns_removed" in verdict.violations


def test_response_rejects_literature_comparison_without_retrieved_evidence() -> None:
    workflow = _review_workflow()

    verdict = validate_nir_response(
        "本次内部留出测试 R²=0.91；这不是外部验证，结果与文献预期一致。",
        workflow,
    )

    assert "knowledge_claim_without_evidence" in verdict.violations


def test_grounded_replacement_contains_only_current_evidence() -> None:
    workflow = _review_workflow()

    rendered = render_grounded_nir_response(workflow, violations=("metric_value_mismatch:r2",))

    assert "0.91234" in rendered
    assert "3.4567" in rendered
    assert "0.99" not in rendered
    assert "不是外部验证" in rendered
    assert "/mnt/user-data/outputs/model.pkl" in rendered
    assert validate_nir_response(rendered, workflow).passed is True


def test_grounded_replacement_summarizes_all_bounded_attempts() -> None:
    workflow = _review_workflow(validation_goal="external_validation", passed=False)
    workflow["stage"] = "blocked"
    workflow["next_action"] = "report_best_effort"
    workflow["attempts"] = [
        {
            "attempt": 1,
            "passed": False,
            "grade": "D",
            "protocol": "named_partition_external_validation",
            "validation_scope": "independent_external_validation",
            "metrics_summary": {"external": {"R2": 0.8326, "RMSE": 1.092, "RPD": 2.44}},
            "model_path": "/mnt/user-data/outputs/model-v1.pkl",
            "metrics_path": "/mnt/user-data/outputs/metrics-v1.json",
        },
        {
            "attempt": 2,
            "passed": False,
            "grade": "D",
            "protocol": "named_partition_external_validation",
            "validation_scope": "independent_external_validation",
            "metrics_summary": {"external": {"R2": 0.7682, "RMSE": 1.285, "RPD": 2.08}},
            "model_path": "/mnt/user-data/outputs/model-v2.pkl",
            "metrics_path": "/mnt/user-data/outputs/metrics-v2.json",
        },
    ]

    rendered = render_grounded_nir_response(workflow, violations=("metric_value_mismatch:r2",))

    assert "尝试 1" in rendered
    assert "0.8326" in rendered
    assert "尝试 2" in rendered
    assert "0.7682" in rendered
    assert validate_nir_response(rendered, workflow).passed is True


def test_grounded_replacement_includes_durable_wavelength_ranges() -> None:
    workflow = _review_workflow()
    workflow["audit_evidence"] = {
        "raw_wavelength_range": [285.0, 1200.0],
        "usable_wavelength_range": [309.0, 1149.0],
        "constant_wavelength_count": 25,
        "usable_wavelength_count": 281,
    }

    rendered = render_grounded_nir_response(
        workflow,
        violations=("wavelength_count_mismatch:constant",),
    )

    assert "原始采集光谱范围：285.0–1200.0 nm" in rendered
    assert "实际可用光谱范围：309.0–1149.0 nm" in rendered
    assert "恒定波长列 25 个；可用波长 281 个" in rendered
    assert validate_nir_response(rendered, workflow).passed is True


def test_middleware_replaces_unsupported_final_answer_and_records_guard_event() -> None:
    workflow = _review_workflow()
    original = AIMessage(content="外部验证 R²=0.99，模型可以生产部署。", id="answer-1")
    middleware = NIRWorkflowMiddleware()

    update = middleware.after_model(
        {"messages": [original], "nir_workflow": workflow},
        SimpleNamespace(context={}),
    )

    assert update is not None
    replacement = update["messages"][0]
    assert replacement.id == "answer-1"
    assert "0.91234" in replacement.content
    assert "0.99" not in replacement.content
    assert replacement.additional_kwargs["nir_response_grounding"]["passed"] is False
    assert update["nir_workflow"]["response_guard_events"][-1]["violations"] == [
        "metric_value_mismatch:r2",
        "validation_scope_overclaim:external",
        "deployment_readiness_overclaim",
        "validation_scope_disclosure_missing",
    ]


def test_middleware_leaves_supported_final_answer_unchanged() -> None:
    workflow = _review_workflow()
    message = AIMessage(content="独立留出测试 R²=0.91；这不是外部验证。", id="answer-2")

    update = NIRWorkflowMiddleware().after_model(
        {"messages": [message], "nir_workflow": workflow},
        SimpleNamespace(context={}),
    )

    assert update is None


def test_middleware_does_not_validate_tool_call_intent_as_final_answer() -> None:
    workflow = _review_workflow()
    message = AIMessage(
        content="准备读取结果。",
        id="tool-intent",
        tool_calls=[{"id": "call-1", "name": "nir_reflect", "args": {}}],
    )

    update = NIRWorkflowMiddleware().after_model(
        {"messages": [message], "nir_workflow": workflow},
        SimpleNamespace(context={}),
    )

    assert update is None


def test_graph_persists_only_the_grounded_final_answer() -> None:
    workflow = _review_workflow()
    agent = create_agent(
        model=_FakeModel(
            disable_streaming=True,
            responses=[
                AIMessage(
                    content="外部验证 R²=0.99，可以生产部署。",
                    id="graph-answer",
                )
            ],
        ),
        tools=[],
        middleware=[NIRWorkflowMiddleware()],
        state_schema=ThreadState,
    )

    result = agent.invoke(
        {
            "messages": [{"role": "user", "content": "请总结结果"}],
            "nir_workflow": workflow,
        }
    )

    final_answer = result["messages"][-1]
    assert "0.99" not in final_answer.content
    assert "0.91234" in final_answer.content
    assert result["nir_workflow"]["response_guard_events"]


def test_stream_gate_releases_only_the_corrected_values_message() -> None:
    workflow = _review_workflow()
    raw = AIMessage(
        content="外部验证 R²=0.99，可以生产部署。",
        id="stream-answer",
    )
    corrected = AIMessage(
        content=render_grounded_nir_response(
            workflow,
            violations=("metric_value_mismatch:r2",),
        ),
        id="stream-answer",
    )
    gate = NIRStreamMessageGate()

    assert gate.observe_values({"messages": [{"role": "user"}], "nir_workflow": workflow}) is None
    assert gate.suppresses(raw) is True
    assert gate.observe_values({"messages": [raw], "nir_workflow": workflow}) is None
    assert gate.allows_values({"messages": [raw], "nir_workflow": workflow}) is False

    released = gate.observe_values({"messages": [corrected], "nir_workflow": workflow})
    assert released is corrected
    assert gate.allows_values({"messages": [corrected], "nir_workflow": workflow}) is True
    assert "0.99" not in released.content
    assert "0.91234" in released.content
