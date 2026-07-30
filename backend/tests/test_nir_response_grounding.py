"""Grounding checks for NIR final responses."""

from __future__ import annotations

from types import SimpleNamespace

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

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
    assert verdict.violations == ("metric_not_in_current_evidence:r2",)


def test_response_rejects_metric_value_not_present_in_current_attempt() -> None:
    verdict = validate_nir_response(
        "模型表现很好，R²=0.99，RPD=5.20。",
        _review_workflow(),
    )

    assert verdict.passed is False
    assert "metric_value_mismatch:r2" in verdict.violations
    assert "metric_value_mismatch:rpd" in verdict.violations


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


def test_grounded_replacement_contains_only_current_evidence() -> None:
    workflow = _review_workflow()

    rendered = render_grounded_nir_response(workflow, violations=("metric_value_mismatch:r2",))

    assert "0.91234" in rendered
    assert "3.4567" in rendered
    assert "0.99" not in rendered
    assert "不是外部验证" in rendered
    assert "/mnt/user-data/outputs/model.pkl" in rendered
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
