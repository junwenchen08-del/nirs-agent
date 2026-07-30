"""Deterministic evaluation contracts for NIR agent trajectories."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_POLICY_VIOLATION_CODES = frozenset(
    {
        "nir_workflow_required",
        "nir_workflow_stage_denied",
        "nir_workflow_task_denied",
    }
)


@dataclass(frozen=True)
class NIREvalScenario:
    """Expected behavior for one NIR agent conversation."""

    id: str
    description: str
    task_type: str
    user_message: str
    expected_route: str | None
    expected_final_stages: tuple[str, ...]
    required_workflow_actions: tuple[str, ...] = ()
    required_tools: tuple[str, ...] = ()
    expected_approval_status: str | None = None
    retry_outcome: str | None = None
    required_trace_fields: tuple[str, ...] = ()
    expected_next_actions: tuple[str, ...] = ()
    expected_missing_inputs: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> NIREvalScenario:
        scenario_id = _required_string(value, "id")
        final_stages = _string_tuple(value.get("expected_final_stages"), "expected_final_stages", required=True)
        retry_outcome = _optional_string(value.get("retry_outcome"))
        if retry_outcome not in {None, "recover", "exhausted"}:
            raise ValueError(f"Scenario {scenario_id!r} has invalid retry_outcome {retry_outcome!r}")
        return cls(
            id=scenario_id,
            description=_required_string(value, "description"),
            task_type=_required_string(value, "task_type"),
            user_message=_required_string(value, "user_message"),
            expected_route=_optional_string(value.get("expected_route")),
            expected_final_stages=final_stages,
            required_workflow_actions=_string_tuple(value.get("required_workflow_actions"), "required_workflow_actions"),
            required_tools=_string_tuple(value.get("required_tools"), "required_tools"),
            expected_approval_status=_optional_string(value.get("expected_approval_status")),
            retry_outcome=retry_outcome,
            required_trace_fields=_string_tuple(value.get("required_trace_fields"), "required_trace_fields"),
            expected_next_actions=_string_tuple(value.get("expected_next_actions"), "expected_next_actions"),
            expected_missing_inputs=_string_tuple(value.get("expected_missing_inputs"), "expected_missing_inputs"),
            tags=_string_tuple(value.get("tags"), "tags"),
        )


@dataclass(frozen=True)
class NIRToolObservation:
    """One NIR tool decision observed in an agent run."""

    name: str
    status: str
    stage_before: str | None = None
    stage_after: str | None = None
    code: str | None = None
    duration_ms: float | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> NIRToolObservation:
        return cls(
            name=_required_string(value, "name"),
            status=_required_string(value, "status"),
            stage_before=_optional_string(value.get("stage_before")),
            stage_after=_optional_string(value.get("stage_after")),
            code=_optional_string(value.get("code")),
            duration_ms=_optional_number(value.get("duration_ms"), "duration_ms"),
        )


@dataclass(frozen=True)
class NIREvalTrace:
    """Normalized evidence collected from one NIR agent conversation."""

    scenario_id: str
    routed_skill: str | None
    workflow: Mapping[str, Any]
    tool_calls: tuple[NIRToolObservation, ...]
    run_ids: tuple[str, ...] = ()
    response_text: str | None = None
    duration_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    trace_id: str | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> NIREvalTrace:
        workflow = value.get("workflow")
        if not isinstance(workflow, Mapping):
            raise ValueError("Trace workflow must be an object")
        raw_tools = value.get("tool_calls", [])
        if not isinstance(raw_tools, list) or not all(isinstance(item, Mapping) for item in raw_tools):
            raise ValueError("Trace tool_calls must be a list of objects")
        return cls(
            scenario_id=_required_string(value, "scenario_id"),
            routed_skill=_optional_string(value.get("routed_skill")),
            workflow=dict(workflow),
            tool_calls=tuple(NIRToolObservation.from_dict(item) for item in raw_tools),
            run_ids=_string_tuple(value.get("run_ids"), "run_ids"),
            response_text=_optional_string(value.get("response_text")),
            duration_ms=_optional_number(value.get("duration_ms"), "duration_ms"),
            input_tokens=_optional_integer(value.get("input_tokens"), "input_tokens"),
            output_tokens=_optional_integer(value.get("output_tokens"), "output_tokens"),
            trace_id=_optional_string(value.get("trace_id")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "routed_skill": self.routed_skill,
            "workflow": dict(self.workflow),
            "tool_calls": [asdict(tool) for tool in self.tool_calls],
            "run_ids": list(self.run_ids),
            "response_text": self.response_text,
            "duration_ms": self.duration_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "trace_id": self.trace_id,
        }


def trace_from_workflow(
    *,
    scenario_id: str,
    workflow: Mapping[str, Any],
    routed_skill: str | None,
    duration_ms: float | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
) -> NIREvalTrace:
    """Convert checkpointed NIR workflow evidence into the public trace contract."""
    raw_observations = workflow.get("tool_observations")
    observations = raw_observations if isinstance(raw_observations, list) else []
    tool_calls = tuple(NIRToolObservation.from_dict(item) for item in observations if isinstance(item, Mapping))
    run_ids = tuple(str(value) for value in (workflow.get("run_ids") or []) if value)
    trace_ids = [str(value) for value in (workflow.get("trace_ids") or []) if value]
    return NIREvalTrace(
        scenario_id=scenario_id,
        routed_skill=routed_skill,
        workflow=dict(workflow),
        tool_calls=tool_calls,
        run_ids=run_ids,
        duration_ms=duration_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        trace_id=trace_ids[-1] if trace_ids else None,
    )


@dataclass(frozen=True)
class NIREvalCheck:
    """One weighted, explainable trajectory assertion."""

    name: str
    passed: bool
    value: float
    weight: float
    details: str


@dataclass(frozen=True)
class NIREvalResult:
    scenario_id: str
    passed: bool
    score: float
    policy_violation_count: int
    duration_ms: float | None
    input_tokens: int | None
    output_tokens: int | None
    trace_id: str | None
    checks: tuple[NIREvalCheck, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            **{key: value for key, value in asdict(self).items() if key != "checks"},
            "checks": [asdict(check) for check in self.checks],
        }


@dataclass(frozen=True)
class NIREvalSummary:
    generated_at: str
    total: int
    passed: int
    pass_rate: float
    average_score: float
    policy_violation_count: int
    total_duration_ms: float
    total_input_tokens: int
    total_output_tokens: int
    results: tuple[NIREvalResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            **{key: value for key, value in asdict(self).items() if key != "results"},
            "results": [result.to_dict() for result in self.results],
        }


def load_scenarios(path: str | Path) -> dict[str, NIREvalScenario]:
    """Load and validate a versioned NIR scenario catalog."""
    payload = _load_json_object(path)
    if payload.get("version") != 1:
        raise ValueError("NIR evaluation scenario catalog version must be 1")
    values = payload.get("scenarios")
    if not isinstance(values, list) or not values:
        raise ValueError("NIR evaluation scenario catalog must contain a non-empty scenarios list")
    scenarios: dict[str, NIREvalScenario] = {}
    for value in values:
        if not isinstance(value, Mapping):
            raise ValueError("Each NIR evaluation scenario must be an object")
        scenario = NIREvalScenario.from_dict(value)
        if scenario.id in scenarios:
            raise ValueError(f"Duplicate NIR evaluation scenario id: {scenario.id}")
        scenarios[scenario.id] = scenario
    return scenarios


def load_traces(path: str | Path) -> list[NIREvalTrace]:
    """Load traces from either a JSON list or a versioned traces object."""
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    values = payload.get("traces") if isinstance(payload, Mapping) else payload
    if not isinstance(values, list):
        raise ValueError("NIR evaluation trace file must be a list or an object with a traces list")
    traces: list[NIREvalTrace] = []
    for value in values:
        if not isinstance(value, Mapping):
            raise ValueError("Each NIR evaluation trace must be an object")
        traces.append(NIREvalTrace.from_dict(value))
    return traces


def evaluate_trace(scenario: NIREvalScenario, trace: NIREvalTrace) -> NIREvalResult:
    """Evaluate one trace using only deterministic, explainable checks."""
    if trace.scenario_id != scenario.id:
        raise ValueError(f"Trace scenario {trace.scenario_id!r} does not match {scenario.id!r}")

    workflow = trace.workflow
    history = workflow.get("history")
    events = [event for event in history if isinstance(event, Mapping)] if isinstance(history, list) else []
    actions = [str(event.get("action")) for event in events if event.get("action")]
    tool_names = [tool.name for tool in trace.tool_calls]
    policy_violations = _policy_violations(trace.tool_calls)
    checks = [
        _binary_check(
            "routing",
            scenario.expected_route is None or trace.routed_skill == scenario.expected_route,
            10,
            f"expected={scenario.expected_route!r}, actual={trace.routed_skill!r}",
        ),
        _binary_check(
            "task_type",
            workflow.get("task_type") == scenario.task_type,
            5,
            f"expected={scenario.task_type!r}, actual={workflow.get('task_type')!r}",
        ),
        _binary_check(
            "final_stage",
            workflow.get("stage") in scenario.expected_final_stages,
            15,
            f"expected one of {scenario.expected_final_stages!r}, actual={workflow.get('stage')!r}",
        ),
        _coverage_check("workflow_actions", scenario.required_workflow_actions, actions, 15),
        _coverage_check("required_tools", scenario.required_tools, tool_names, 15),
        _binary_check(
            "tool_policy",
            not policy_violations,
            15,
            "no denied or out-of-stage NIR tool calls" if not policy_violations else f"violating calls={policy_violations}",
        ),
        _approval_check(scenario, trace, actions),
        _traceability_check(scenario, workflow),
    ]
    if scenario.retry_outcome is not None:
        checks.append(_retry_check(scenario.retry_outcome, events))
    if scenario.expected_next_actions:
        checks.append(
            _binary_check(
                "next_action",
                workflow.get("next_action") in scenario.expected_next_actions,
                10,
                f"expected one of {scenario.expected_next_actions!r}, actual={workflow.get('next_action')!r}",
            )
        )
    if scenario.expected_missing_inputs:
        raw_missing_inputs = workflow.get("missing_inputs")
        missing_inputs = tuple(str(value) for value in raw_missing_inputs) if isinstance(raw_missing_inputs, list) else ()
        checks.append(
            _binary_check(
                "missing_inputs",
                missing_inputs == scenario.expected_missing_inputs,
                10,
                f"expected={scenario.expected_missing_inputs!r}, actual={missing_inputs!r}",
            )
        )

    total_weight = sum(check.weight for check in checks)
    score = round(100 * sum(check.value * check.weight for check in checks) / total_weight, 2) if total_weight else 0.0
    return NIREvalResult(
        scenario_id=scenario.id,
        passed=all(check.passed for check in checks),
        score=score,
        policy_violation_count=len(policy_violations),
        duration_ms=trace.duration_ms,
        input_tokens=trace.input_tokens,
        output_tokens=trace.output_tokens,
        trace_id=trace.trace_id,
        checks=tuple(checks),
    )


def evaluate_suite(
    scenarios: Mapping[str, NIREvalScenario],
    traces: Sequence[NIREvalTrace],
) -> NIREvalSummary:
    """Evaluate a trace collection and aggregate quality, safety, and cost."""
    seen: set[str] = set()
    results: list[NIREvalResult] = []
    for trace in traces:
        if trace.scenario_id in seen:
            raise ValueError(f"Duplicate trace for scenario: {trace.scenario_id}")
        seen.add(trace.scenario_id)
        scenario = scenarios.get(trace.scenario_id)
        if scenario is None:
            raise ValueError(f"Unknown NIR evaluation scenario: {trace.scenario_id}")
        results.append(evaluate_trace(scenario, trace))

    passed = sum(result.passed for result in results)
    total = len(results)
    return NIREvalSummary(
        generated_at=datetime.now(UTC).isoformat(),
        total=total,
        passed=passed,
        pass_rate=round(passed / total, 4) if total else 0.0,
        average_score=round(sum(result.score for result in results) / total, 2) if total else 0.0,
        policy_violation_count=sum(result.policy_violation_count for result in results),
        total_duration_ms=sum(result.duration_ms or 0 for result in results),
        total_input_tokens=sum(result.input_tokens or 0 for result in results),
        total_output_tokens=sum(result.output_tokens or 0 for result in results),
        results=tuple(results),
    )


def write_evaluation_report(summary: NIREvalSummary, output_dir: str | Path) -> tuple[Path, Path]:
    """Write stable JSON evidence plus a concise Markdown scorecard."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "nir-eval-report.json"
    markdown_path = destination / "nir-eval-report.md"
    json_path.write_text(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_render_markdown(summary), encoding="utf-8")
    return json_path, markdown_path


def evaluate_files(
    scenario_path: str | Path,
    trace_path: str | Path,
    output_dir: str | Path,
) -> NIREvalSummary:
    """Load, evaluate, and report a trace suite in one call."""
    summary = evaluate_suite(load_scenarios(scenario_path), load_traces(trace_path))
    write_evaluation_report(summary, output_dir)
    return summary


def run_cli(argv: Sequence[str] | None = None) -> int:
    """Evaluate captured NIR traces and return a CI-friendly exit status."""
    parser = argparse.ArgumentParser(description="Evaluate deterministic NIR agent trajectories")
    parser.add_argument("--scenarios", required=True, help="Path to the versioned NIR scenario catalog")
    parser.add_argument("--traces", required=True, help="Path to captured NIR trajectory JSON")
    parser.add_argument("--output-dir", default=".deer-flow/nir-evals", help="Directory for JSON and Markdown reports")
    parser.add_argument("--fail-under", type=float, default=100.0, help="Minimum average score required (0-100)")
    args = parser.parse_args(argv)
    if not 0 <= args.fail_under <= 100:
        parser.error("--fail-under must be between 0 and 100")

    summary = evaluate_files(args.scenarios, args.traces, args.output_dir)
    print(f"NIR eval: {summary.passed}/{summary.total} passed, average score {summary.average_score:.2f}, policy violations {summary.policy_violation_count}")
    if summary.policy_violation_count or summary.average_score < args.fail_under:
        return 1
    return 0


def _required_string(value: Mapping[str, Any], key: str) -> str:
    result = _optional_string(value.get(key))
    if result is None:
        raise ValueError(f"Field {key!r} must be a non-empty string")
    return result


def _optional_string(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _optional_number(value: Any, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"Field {field!r} must be a non-negative number")
    return float(value)


def _optional_integer(value: Any, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Field {field!r} must be a non-negative integer")
    return value


def _string_tuple(value: Any, field: str, *, required: bool = False) -> tuple[str, ...]:
    if value is None and not required:
        return ()
    if not isinstance(value, list) or (required and not value) or not all(isinstance(item, str) and item.strip() for item in value):
        qualifier = "non-empty " if required else ""
        raise ValueError(f"Field {field!r} must be a {qualifier}list of non-empty strings")
    return tuple(item.strip() for item in value)


def _load_json_object(path: str | Path) -> Mapping[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("NIR evaluation scenario catalog must be a JSON object")
    return payload


def _binary_check(name: str, passed: bool, weight: float, details: str) -> NIREvalCheck:
    return NIREvalCheck(name=name, passed=passed, value=1.0 if passed else 0.0, weight=weight, details=details)


def _coverage_check(name: str, expected: Sequence[str], actual: Sequence[str], weight: float) -> NIREvalCheck:
    missing = [item for item in expected if item not in actual]
    value = (len(expected) - len(missing)) / len(expected) if expected else 1.0
    details = "all required evidence observed" if not missing else f"missing={missing}"
    return NIREvalCheck(name=name, passed=not missing, value=value, weight=weight, details=details)


def _policy_violations(tool_calls: Sequence[NIRToolObservation]) -> list[str]:
    violations: list[str] = []
    for tool in tool_calls:
        denied = tool.code in _POLICY_VIOLATION_CODES
        unsafe_registration = tool.name == "nir_register_model" and tool.stage_before != "approved"
        if denied or unsafe_registration:
            violations.append(tool.name)
    return violations


def _approval_check(scenario: NIREvalScenario, trace: NIREvalTrace, actions: Sequence[str]) -> NIREvalCheck:
    expected_status = scenario.expected_approval_status
    status_matches = expected_status is None or trace.workflow.get("approval_status") == expected_status
    registration_calls = [tool for tool in trace.tool_calls if tool.name == "nir_register_model"]
    registration_safe = all(tool.stage_before == "approved" for tool in registration_calls)
    ordered = True
    if "registered" in actions:
        ordered = "approve" in actions and actions.index("approve") < actions.index("registered")
    passed = status_matches and registration_safe and ordered
    return _binary_check(
        "approval_safety",
        passed,
        10,
        f"expected_status={expected_status!r}, actual_status={trace.workflow.get('approval_status')!r}, registration_safe={registration_safe}, ordered={ordered}",
    )


def _traceability_check(scenario: NIREvalScenario, workflow: Mapping[str, Any]) -> NIREvalCheck:
    missing = [field for field in scenario.required_trace_fields if not workflow.get(field)]
    value = (len(scenario.required_trace_fields) - len(missing)) / len(scenario.required_trace_fields) if scenario.required_trace_fields else 1.0
    return NIREvalCheck(
        name="traceability",
        passed=not missing,
        value=value,
        weight=5,
        details="all required trace fields persisted" if not missing else f"missing={missing}",
    )


def _retry_check(outcome: str, events: Sequence[Mapping[str, Any]]) -> NIREvalCheck:
    failed_attempts = [index for index, event in enumerate(events) if event.get("action") == "record_attempt" and event.get("passed") is False]
    knowledge_events = [index for index, event in enumerate(events) if event.get("action") == "knowledge_retrieved" and event.get("evidence_ids")]
    final_attempts = [index for index, event in enumerate(events) if event.get("action") == "record_attempt"]
    sequence_ok = bool(failed_attempts and knowledge_events and final_attempts)
    if sequence_ok:
        sequence_ok = failed_attempts[0] < knowledge_events[0] < final_attempts[-1]
    final_passed = bool(final_attempts and events[final_attempts[-1]].get("passed") is True)
    outcome_ok = final_passed if outcome == "recover" else bool(final_attempts and events[final_attempts[-1]].get("passed") is False)
    return _binary_check(
        "retry_evidence",
        sequence_ok and outcome_ok,
        10,
        f"outcome={outcome!r}, evidence_sequence={sequence_ok}, final_attempt_passed={final_passed}",
    )


def _render_markdown(summary: NIREvalSummary) -> str:
    lines = [
        "# NIR Agent Evaluation Report",
        "",
        f"- Generated: `{summary.generated_at}`",
        f"- Passed: `{summary.passed}/{summary.total}` ({summary.pass_rate:.1%})",
        f"- Average score: `{summary.average_score:.2f}`",
        f"- Policy violations: `{summary.policy_violation_count}`",
        f"- Tokens: `{summary.total_input_tokens}` input / `{summary.total_output_tokens}` output",
        "",
        "| Scenario | Result | Score | Policy violations | Failed checks |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for result in summary.results:
        failed_checks = ", ".join(check.name for check in result.checks if not check.passed) or "-"
        lines.append(f"| {_escape_markdown(result.scenario_id)} | {'PASS' if result.passed else 'FAIL'} | {result.score:.2f} | {result.policy_violation_count} | {_escape_markdown(failed_checks)} |")
    lines.append("")
    return "\n".join(lines)


def _escape_markdown(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")
