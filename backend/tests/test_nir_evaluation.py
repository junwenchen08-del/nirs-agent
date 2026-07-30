"""End-to-end trajectory evaluation for the NIR agent workflow."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from deerflow.agents.middlewares.nir_workflow_middleware import NIRWorkflowMiddleware
from deerflow.community.nir.evaluation import (
    NIREvalTrace,
    NIRToolObservation,
    evaluate_suite,
    evaluate_trace,
    load_scenarios,
    run_cli,
    write_evaluation_report,
)
from deerflow.community.nir.workflow import start_workflow, transition_workflow

SCENARIOS_PATH = Path(__file__).parents[1] / "evals" / "nir" / "scenarios.json"


def _request(tool_name: str, workflow: dict, call_id: str, args: dict | None = None) -> ToolCallRequest:
    state = {"nir_workflow": workflow}
    runtime = ToolRuntime(
        state=state,
        context={},
        config={"configurable": {}},
        stream_writer=lambda _: None,
        tools=[],
        tool_call_id=call_id,
        store=None,
    )
    return ToolCallRequest(
        tool_call={"id": call_id, "name": tool_name, "args": args or {}},
        tool=None,
        state=state,
        runtime=runtime,
    )


def _run_tool(
    middleware: NIRWorkflowMiddleware,
    workflow: dict,
    observations: list[NIRToolObservation],
    *,
    tool_name: str,
    payload: dict,
    args: dict | None = None,
) -> dict:
    call_id = f"call-{len(observations) + 1}"
    stage_before = workflow["stage"]
    message = ToolMessage(
        content=json.dumps(payload),
        tool_call_id=call_id,
        name=tool_name,
        status="error" if payload.get("status") == "error" else "success",
    )
    result = middleware.wrap_tool_call(
        _request(tool_name, workflow, call_id, args),
        lambda _: message,
    )
    updated = result.update.get("nir_workflow", workflow) if isinstance(result, Command) else workflow
    output_message = result.update["messages"][-1] if isinstance(result, Command) else result
    output_payload = json.loads(output_message.content)
    observations.append(
        NIRToolObservation(
            name=tool_name,
            status="error" if output_message.status == "error" else "success",
            stage_before=stage_before,
            stage_after=updated["stage"],
            code=output_payload.get("code"),
        )
    )
    return updated


def _scenario(scenario_id: str):
    return load_scenarios(SCENARIOS_PATH)[scenario_id]


def test_default_scenario_catalog_covers_core_nir_agent_paths() -> None:
    scenarios = load_scenarios(SCENARIOS_PATH)

    assert len(scenarios) == 23
    assert {
        "intake-missing-requirements",
        "inspection-complete",
        "calibration-register",
        "calibration-retry-recover",
        "retry-budget-exhausted",
        "prediction-complete",
        "knowledge-complete",
        "compare-review",
        "data-audit-blocked",
        "calibration-rejected",
        "compare-retry-recover",
        "prediction-missing-model",
        "knowledge-no-hit-complete",
        "soil-moisture-calibration-register",
        "pharmaceutical-api-calibration-review",
        "multi-component-basic",
        "multi-component-shared-preprocess",
        "multi-component-predict",
    }.issubset(scenarios)
    clarification = scenarios["intake-missing-requirements"]
    assert clarification.expected_final_stages == ("clarification",)
    assert clarification.expected_next_actions == ("ask_targeted_clarification",)
    assert clarification.expected_missing_inputs == ("validation_goal",)


def test_complete_calibration_trace_passes_every_evaluation_check() -> None:
    middleware = NIRWorkflowMiddleware()
    observations: list[NIRToolObservation] = []
    workflow = start_workflow(
        task_type="calibration",
        data_path="corn.npz",
        analyte="protein",
        unit="%",
        domain="food_protein",
        validation_goal="internal_holdout",
    )
    workflow = _run_tool(
        middleware,
        workflow,
        observations,
        tool_name="nir_inspect",
        payload={"status": "ok", "samples": 80},
    )
    workflow = transition_workflow(workflow, action="record_audit", audit_passed=True)
    workflow = transition_workflow(workflow, action="plan_ready")
    workflow = _run_tool(
        middleware,
        workflow,
        observations,
        tool_name="nir_train_model",
        payload={
            "status": "ok",
            "passed": True,
            "grade": "A",
            "model_path": "model.pkl",
            "metrics_path": "metrics.json",
        },
    )
    workflow = transition_workflow(workflow, action="approve", notes="User approved")
    workflow = _run_tool(
        middleware,
        workflow,
        observations,
        tool_name="nir_register_model",
        payload={"status": "registered", "model_id": "corn-protein", "version": 1},
        args={"model_path": "model.pkl", "metrics_path": "metrics.json"},
    )
    workflow = transition_workflow(workflow, action="complete")
    trace = NIREvalTrace(
        scenario_id="calibration-register",
        routed_skill="nir-coordinator",
        workflow=workflow,
        tool_calls=tuple(observations),
        duration_ms=1250,
        input_tokens=800,
        output_tokens=350,
    )

    result = evaluate_trace(_scenario(trace.scenario_id), trace)

    assert result.passed is True
    assert result.score == 100.0
    assert result.policy_violation_count == 0
    assert all(check.passed for check in result.checks)

    unsafe_result = evaluate_trace(
        _scenario(trace.scenario_id),
        replace(
            trace,
            response_text="模型已完成外部验证，R²=0.99，可以生产部署。",
        ),
    )
    grounding = next(check for check in unsafe_result.checks if check.name == "response_grounding")
    assert grounding.passed is False
    assert "unsupported response claims" in grounding.details


def test_failed_attempt_with_evidence_then_recovery_passes_retry_scenario() -> None:
    middleware = NIRWorkflowMiddleware()
    observations: list[NIRToolObservation] = []
    workflow = start_workflow(
        task_type="calibration",
        data_path="corn.npz",
        analyte="protein",
        unit="%",
        domain="food_protein",
        validation_goal="internal_holdout",
    )
    workflow = _run_tool(
        middleware,
        workflow,
        observations,
        tool_name="nir_inspect",
        payload={"status": "ok"},
    )
    workflow = transition_workflow(workflow, action="record_audit", audit_passed=True)
    workflow = transition_workflow(workflow, action="plan_ready")
    workflow = _run_tool(
        middleware,
        workflow,
        observations,
        tool_name="nir_train_model",
        payload={"status": "ok", "passed": False, "grade": "C"},
        args={"method": "pls", "pipeline_steps": '["snv"]'},
    )
    workflow = _run_tool(
        middleware,
        workflow,
        observations,
        tool_name="nir_reflect",
        payload={
            "status": "ok",
            "attempt": 1,
            "should_retry": True,
            "reason": "Residual curvature remains.",
            "diagnostics": {"residual_trend": "curved"},
            "knowledge_hint": {"query": "NIR curved residual derivative"},
        },
    )
    workflow = _run_tool(
        middleware,
        workflow,
        observations,
        tool_name="nir_search_knowledge",
        payload={"status": "ok", "results": [{"id": "paper-001"}], "count": 1},
    )
    workflow = transition_workflow(
        workflow,
        action="record_retry_plan",
        retry_tool="nir_train_model",
        retry_method="pls",
        retry_pipeline_steps='["snv", {"method": "derivative1", "params": {"window": 11}}]',
        retry_rationale="Use derivative preprocessing to address curved residuals.",
        expected_improvement="Reduce RMSEP and improve RPD.",
    )
    workflow = transition_workflow(workflow, action="plan_ready", notes="Retry with evidence")
    workflow = _run_tool(
        middleware,
        workflow,
        observations,
        tool_name="nir_train_model",
        payload={
            "status": "ok",
            "passed": True,
            "grade": "B",
            "model_path": "retry-model.pkl",
            "metrics_path": "retry-metrics.json",
        },
        args={
            "method": "pls",
            "pipeline_steps": '["snv", {"method": "derivative1", "params": {"window": 11}}]',
        },
    )
    trace = NIREvalTrace(
        scenario_id="calibration-retry-recover",
        routed_skill="nir-coordinator",
        workflow=workflow,
        tool_calls=tuple(observations),
    )

    result = evaluate_trace(_scenario(trace.scenario_id), trace)

    assert result.passed is True
    assert workflow["knowledge_evidence"] == ["paper-001"]
    assert any(event["action"] == "knowledge_retrieved" and event["evidence_ids"] == ["paper-001"] for event in workflow["history"])


def test_retry_budget_exhaustion_requires_final_stop_reflection() -> None:
    middleware = NIRWorkflowMiddleware()
    observations: list[NIRToolObservation] = []
    workflow = start_workflow(
        task_type="analysis",
        data_path="corn.npz",
        analyte="protein",
        unit="%",
        domain="food_protein",
        validation_goal="internal_holdout",
        max_attempts=2,
    )
    workflow = _run_tool(
        middleware,
        workflow,
        observations,
        tool_name="nir_inspect",
        payload={"status": "ok"},
    )
    workflow = transition_workflow(workflow, action="record_audit", audit_passed=True)
    workflow = transition_workflow(workflow, action="plan_ready")
    workflow = _run_tool(
        middleware,
        workflow,
        observations,
        tool_name="nir_train_model",
        payload={"status": "ok", "passed": False, "grade": "D"},
        args={"method": "pls", "pipeline_steps": '["snv"]'},
    )
    workflow = _run_tool(
        middleware,
        workflow,
        observations,
        tool_name="nir_reflect",
        payload={
            "status": "ok",
            "attempt": 1,
            "should_retry": True,
            "diagnostics": {"residual_variance": "high"},
            "knowledge_hint": {"query": "NIR residual variance smoothing"},
        },
    )
    workflow = _run_tool(
        middleware,
        workflow,
        observations,
        tool_name="nir_search_knowledge",
        payload={"status": "ok", "results": [{"id": "paper-002"}], "count": 1},
    )
    workflow = transition_workflow(
        workflow,
        action="record_retry_plan",
        retry_tool="nir_train_model",
        retry_method="pls",
        retry_pipeline_steps='["snv", {"method": "sg_smooth", "params": {"window": 11}}]',
        retry_rationale="Smooth high-variance residual noise.",
        expected_improvement="Reduce RMSEP variance.",
    )
    workflow = transition_workflow(workflow, action="plan_ready")
    workflow = _run_tool(
        middleware,
        workflow,
        observations,
        tool_name="nir_train_model",
        payload={"status": "ok", "passed": False, "grade": "D"},
        args={
            "method": "pls",
            "pipeline_steps": '["snv", {"method": "sg_smooth", "params": {"window": 11}}]',
        },
    )
    workflow = _run_tool(
        middleware,
        workflow,
        observations,
        tool_name="nir_reflect",
        payload={
            "status": "ok",
            "attempt": 2,
            "should_retry": False,
            "reason": "Retry budget exhausted without sufficient improvement.",
            "diagnostics": {"residual_variance": "high"},
        },
    )
    trace = NIREvalTrace(
        scenario_id="retry-budget-exhausted",
        routed_skill="nir-coordinator",
        workflow=workflow,
        tool_calls=tuple(observations),
    )

    result = evaluate_trace(_scenario(trace.scenario_id), trace)

    assert result.passed is True
    assert workflow["stage"] == "blocked"
    assert workflow["reflection"]["stop_reason"] == "retry_budget_exhausted"
    assert next(check for check in result.checks if check.name == "retry_evidence").passed is True


def test_unapproved_registration_attempt_is_reported_as_agent_policy_violation() -> None:
    workflow = start_workflow(
        task_type="calibration",
        data_path="corn.npz",
        analyte="protein",
        unit="%",
        domain="food_protein",
        validation_goal="internal_holdout",
    )
    workflow = transition_workflow(workflow, action="record_audit", audit_passed=True)
    workflow = transition_workflow(workflow, action="plan_ready")
    workflow = transition_workflow(
        workflow,
        action="record_attempt",
        attempt_passed=True,
        grade="A",
        model_path="model.pkl",
        metrics_path="metrics.json",
    )
    trace = NIREvalTrace(
        scenario_id="calibration-register",
        routed_skill="nir-coordinator",
        workflow=workflow,
        tool_calls=(
            NIRToolObservation(
                name="nir_register_model",
                status="error",
                stage_before="review",
                stage_after="review",
                code="nir_workflow_stage_denied",
            ),
        ),
    )

    result = evaluate_trace(_scenario(trace.scenario_id), trace)

    assert result.passed is False
    assert result.policy_violation_count == 1
    assert next(check for check in result.checks if check.name == "approval_safety").passed is False
    assert next(check for check in result.checks if check.name == "tool_policy").passed is False


def test_suite_report_is_machine_readable_and_human_readable(tmp_path: Path) -> None:
    workflow = start_workflow(task_type="inspection", data_path="sample.csv")
    trace = NIREvalTrace(
        scenario_id="inspection-complete",
        routed_skill="nir-coordinator",
        workflow=workflow,
        tool_calls=(),
    )
    summary = evaluate_suite(load_scenarios(SCENARIOS_PATH), [trace])

    json_path, markdown_path = write_evaluation_report(summary, tmp_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert payload["total"] == 1
    assert payload["results"][0]["scenario_id"] == "inspection-complete"
    assert "# NIR Agent Evaluation Report" in markdown
    assert "inspection-complete" in markdown


def test_cli_evaluates_trace_file_and_enforces_score_threshold(tmp_path: Path) -> None:
    workflow = start_workflow(task_type="calibration", data_path="corn.npz")
    workflow = transition_workflow(
        workflow,
        action="record_audit",
        audit_passed=True,
        domain="food_protein",
        analyte="protein",
        unit="%",
    )
    trace = NIREvalTrace(
        scenario_id="intake-missing-requirements",
        routed_skill="nir-coordinator",
        workflow=workflow,
        tool_calls=(
            NIRToolObservation(
                name="nir_inspect",
                status="success",
                stage_before="data_audit",
                stage_after="data_audit",
            ),
        ),
    )
    traces_path = tmp_path / "traces.json"
    traces_path.write_text(json.dumps({"version": 1, "traces": [trace.to_dict()]}), encoding="utf-8")
    report_dir = tmp_path / "report"

    exit_code = run_cli(
        [
            "--scenarios",
            str(SCENARIOS_PATH),
            "--traces",
            str(traces_path),
            "--output-dir",
            str(report_dir),
            "--fail-under",
            "100",
        ]
    )

    assert exit_code == 0
    assert (report_dir / "nir-eval-report.json").exists()
    assert (report_dir / "nir-eval-report.md").exists()
