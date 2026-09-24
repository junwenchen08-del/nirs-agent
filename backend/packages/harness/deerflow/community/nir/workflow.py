"""Deterministic control plane for NIR agent workflows.

The language model can choose domain actions, but it cannot skip required
inputs, retry budgets, or the approval gate encoded here. State is persisted in
``ThreadState.nir_workflow`` so a workflow survives multi-turn conversations.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command

from deerflow.agents.thread_state import NIRClarificationQuestion, NIRWorkflowState
from deerflow.tools.types import Runtime
from deerflow.utils.messages import get_original_user_content_text

_SUPPORTED_TASK_TYPES = frozenset({"analysis", "calibration", "classification", "multi_modeling", "compare", "prediction", "inspection", "knowledge"})
_MODELING_TASK_TYPES = frozenset({"analysis", "calibration", "classification", "multi_modeling", "compare"})
_PREREQUISITE_INPUTS = {
    "analysis": ("data_path",),
    "calibration": ("data_path",),
    "classification": ("data_path",),
    "multi_modeling": ("data_path",),
    "compare": ("data_path",),
    "prediction": ("data_path", "model_path"),
    "inspection": ("data_path",),
    "knowledge": (),
}
_DECISION_INPUTS = ("domain", "analyte", "unit", "validation_goal")
_CLASSIFICATION_DECISION_INPUTS = ("domain", "label_column", "validation_goal")
_HIGH_ASSURANCE_INPUTS = ("instrument", "grouping_column", "reference_method")
_REQUIREMENT_PLACEHOLDERS = frozenset(
    {
        "unknown",
        "none",
        "n/a",
        "na",
        "null",
        "unspecified",
        "not specified",
        "未知",
        "不清楚",
        "不知道",
        "无",
        "没有",
    }
)
_VALIDATION_GOAL_ALIASES = {
    "exploratory": "exploratory",
    "exploration": "exploratory",
    "exploratory_analysis": "exploratory",
    "探索": "exploratory",
    "探索分析": "exploratory",
    "internal_holdout": "internal_holdout",
    "internal": "internal_holdout",
    "holdout": "internal_holdout",
    "same_dataset_holdout": "internal_holdout",
    "同一数据集独立留出": "internal_holdout",
    "独立留出": "internal_holdout",
    "external_validation": "external_validation",
    "external": "external_validation",
    "独立外部验证": "external_validation",
    "外部验证": "external_validation",
    "production": "production",
    "deployment": "production",
    "production_deployment": "production",
    "生产部署": "production",
    "部署": "production",
}
_AGENT_VIEW_FIELDS = (
    "project_id",
    "task_type",
    "stage",
    "revision",
    "domain",
    "analyte",
    "unit",
    "label_column",
    "validation_goal",
    "instrument",
    "grouping_column",
    "reference_method",
    "data_path",
    "model_path",
    "metrics_path",
    "audit_evidence",
    "attempt_evidence",
    "attempts",
    "reflection",
    "retry_plan",
    "knowledge_evidence",
    "attempt",
    "max_attempts",
    "audit_status",
    "missing_inputs",
    "clarification_questions",
    "approval_status",
    "next_action",
)
_AGENT_OBSERVATION_FIELDS = ("name", "status", "code", "requested_method", "stage_after")
_HIGH_ASSURANCE_GOAL_MARKERS = (
    "external",
    "production",
    "deployment",
    "transfer",
    "外部",
    "生产",
    "部署",
    "迁移",
)
_HISTORY_LIMIT = 50
_TOOL_OBSERVATION_LIMIT = 100
_RESPONSE_GUARD_LIMIT = 20
_RETRY_RECORD_LIMIT = 10
_RETRY_MODELING_TOOLS = frozenset(
    {
        "nir_train_auto_split_model",
        "nir_train_model",
        "nir_train_partitioned_model",
        "nir_train_multi_model",
        "nir_analyze",
        "nir_analyze_collection",
        "nir_compare",
        "nir_train_classifier",
    }
)
_RETRY_MODEL_DEFAULTS = {
    "nir_train_auto_split_model": "auto",
    "nir_train_model": "pls",
    "nir_train_partitioned_model": "auto",
    "nir_train_multi_model": "pls",
    "nir_analyze": "auto",
    "nir_analyze_collection": "auto",
    "nir_compare": "pls",
    "nir_train_classifier": "auto",
}
_RETRY_SIGNATURE_IGNORED_MODEL_ARGS = frozenset(
    {
        "method",
        "pipeline_steps",
        "input_path",
        "file_path",
        "data_path",
        "output_path",
        "output_dir",
        "model_path",
        "metrics_path",
        "model_output",
        "metrics_output",
        "attempt",
        "tool_call_id",
    }
)
_RETRY_TOOLS_BY_TASK = {
    "analysis": frozenset(
        {
            "nir_train_auto_split_model",
            "nir_train_model",
            "nir_train_partitioned_model",
            "nir_analyze",
            "nir_analyze_collection",
            "nir_compare",
        }
    ),
    "calibration": frozenset(
        {
            "nir_train_auto_split_model",
            "nir_train_model",
            "nir_train_partitioned_model",
            "nir_analyze",
            "nir_analyze_collection",
            "nir_compare",
        }
    ),
    "multi_modeling": frozenset({"nir_train_multi_model", "nir_compare"}),
    "compare": frozenset({"nir_compare"}),
    "classification": frozenset({"nir_train_classifier"}),
}
_APPROVAL_PATTERNS = (
    r"批准",
    r"同意(?:采用|使用|注册)",
    r"确认(?:采用|使用|注册)",
    r"注册(?:这个|该)?模型",
    r"\bapprove(?:d)?\b",
    r"\bproceed\s+with\s+(?:the\s+)?registration\b",
)
_APPROVAL_DENIAL_PATTERNS = (
    r"(?:不|未|尚未|暂不|不要|拒绝|取消)\s*(?:批准|同意|确认|采用|使用|注册)",
    r"(?:不同意|不确认|不采用|不使用|不注册)",
    r"\b(?:do\s+not|don't|not|never|decline|reject|cancel)\b.{0,40}\b(?:approve(?:d)?|agree|proceed|register|registration)\b",
    r"\bwithout\s+(?:my\s+)?approval\b",
)


class NIRWorkflowError(ValueError):
    """Raised when an invalid workflow transition is requested."""


def normalize_retry_pipeline_steps(value: Any) -> list[dict[str, Any]]:
    """Canonicalize a retry pipeline for durable comparison and hashing."""

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, ValueError) as exc:
            raise NIRWorkflowError("retry_pipeline_steps must be valid JSON") from exc
    if not isinstance(value, list) or not value or len(value) > 20:
        raise NIRWorkflowError("retry_pipeline_steps must be a non-empty list with at most 20 steps")

    normalized: list[dict[str, Any]] = []
    for raw_step in value:
        if isinstance(raw_step, str):
            method = raw_step.strip().lower()
            params: Mapping[str, Any] = {}
        elif isinstance(raw_step, Mapping):
            method = str(raw_step.get("method") or "").strip().lower()
            raw_params = raw_step.get("params") or {}
            if not isinstance(raw_params, Mapping):
                raise NIRWorkflowError("retry pipeline step params must be an object")
            params = raw_params
        else:
            raise NIRWorkflowError("retry pipeline steps must be method strings or objects")
        if not method:
            raise NIRWorkflowError("retry pipeline step method cannot be empty")
        try:
            canonical_params = json.loads(json.dumps(dict(params), sort_keys=True, ensure_ascii=False))
        except (TypeError, ValueError) as exc:
            raise NIRWorkflowError("retry pipeline step params must be JSON serializable") from exc
        normalized.append({"method": method, "params": canonical_params})
    return normalized


def normalize_retry_model_method(tool_name: str, method: str | None) -> str:
    """Canonicalize tool-specific model defaults and accepted aliases."""

    normalized_tool = str(tool_name or "").strip()
    if normalized_tool not in _RETRY_MODELING_TOOLS:
        raise NIRWorkflowError(f"Unsupported retry modeling tool {normalized_tool!r}")
    default = _RETRY_MODEL_DEFAULTS[normalized_tool]
    normalized_method = str(method or default).strip().lower() or default
    return "cnn" if normalized_method in {"1d-cnn", "1d_cnn"} else normalized_method


def retry_execution_signature(
    *,
    tool_name: str,
    method: str | None,
    pipeline_steps: Any,
    model_args: Any = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    """Return the canonical pipeline and stable execution signature."""

    normalized_tool = str(tool_name or "").strip()
    if normalized_tool not in _RETRY_MODELING_TOOLS:
        raise NIRWorkflowError(f"Unsupported retry modeling tool {normalized_tool!r}")
    normalized_steps = normalize_retry_pipeline_steps(pipeline_steps)
    if isinstance(model_args, str):
        try:
            model_args = json.loads(model_args)
        except (TypeError, ValueError) as exc:
            raise NIRWorkflowError("retry_model_args must be valid JSON") from exc
    if model_args is None:
        model_args = {}
    if not isinstance(model_args, Mapping):
        raise NIRWorkflowError("retry_model_args must be a JSON object")
    try:
        semantic_model_args = {str(key): value for key, value in model_args.items() if str(key) not in _RETRY_SIGNATURE_IGNORED_MODEL_ARGS}
        normalized_model_args = json.loads(json.dumps(semantic_model_args, sort_keys=True, ensure_ascii=False))
    except (TypeError, ValueError) as exc:
        raise NIRWorkflowError("retry_model_args must be JSON serializable") from exc
    payload = {
        "tool_name": normalized_tool,
        "method": normalize_retry_model_method(normalized_tool, method),
        "pipeline_steps": normalized_steps,
        "model_args": normalized_model_args,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return normalized_steps, normalized_model_args, hashlib.sha256(encoded).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _is_meaningful_requirement(value: Any) -> bool:
    if not isinstance(value, str):
        return bool(value)
    normalized = re.sub(r"\s+", " ", value.strip().lower())
    return bool(normalized) and normalized not in _REQUIREMENT_PLACEHOLDERS


def _missing_inputs(state: NIRWorkflowState, required: tuple[str, ...]) -> list[str]:
    return [field for field in required if not _is_meaningful_requirement(state.get(field))]


def _prerequisite_missing(state: NIRWorkflowState) -> list[str]:
    return _missing_inputs(state, _PREREQUISITE_INPUTS.get(state["task_type"], ()))


def _requires_high_assurance_context(validation_goal: str | None) -> bool:
    normalized = str(validation_goal or "").strip().lower()
    return any(marker in normalized for marker in _HIGH_ASSURANCE_GOAL_MARKERS)


def _normalize_validation_goal(value: str) -> str:
    normalized = re.sub(r"[\s-]+", "_", value.strip().lower())
    canonical = _VALIDATION_GOAL_ALIASES.get(normalized)
    if canonical is None:
        raise NIRWorkflowError(f"Unsupported validation_goal {value!r}; expected exploratory, internal_holdout, external_validation, or production")
    return canonical


def _decision_missing(state: NIRWorkflowState) -> list[str]:
    if state["task_type"] not in _MODELING_TASK_TYPES:
        return []
    required = list(_CLASSIFICATION_DECISION_INPUTS if state["task_type"] == "classification" else _DECISION_INPUTS)
    if _requires_high_assurance_context(state.get("validation_goal")):
        required.extend(_HIGH_ASSURANCE_INPUTS)
    return _missing_inputs(state, tuple(required))


def _clarification_questions(missing: list[str]) -> list[NIRClarificationQuestion]:
    questions: list[NIRClarificationQuestion] = []
    target_fields = [field for field in ("domain", "analyte", "unit") if field in missing]
    if target_fields:
        questions.append(
            {
                "fields": target_fields,
                "question": "请确认样品/应用领域、目标成分，以及参考值或预测值的单位。",
                "reason": "这些信息决定质量阈值、目标列解释和最终报告口径。",
            }
        )
    if "validation_goal" in missing:
        questions.append(
            {
                "fields": ["validation_goal"],
                "question": "这次任务的验证目标是什么：探索分析、同一数据集独立留出、独立外部验证，还是生产部署？",
                "reason": "验证目标决定数据划分、质量声明和是否需要额外域信息。",
            }
        )
    if "label_column" in missing:
        questions.append(
            {
                "fields": ["label_column"],
                "question": "请确认用于定性判别的类别标签列名；该列应表示品种、产地、真伪或等级，而不是连续参考值。",
                "reason": "分类标签决定类别词表、分层划分、混淆矩阵和最终判别含义。",
            }
        )
    assurance_fields = [field for field in _HIGH_ASSURANCE_INPUTS if field in missing]
    if assurance_fields:
        questions.append(
            {
                "fields": assurance_fields,
                "question": "请补充仪器型号、可用于隔离批次/季节/产地的分组列，以及参考实验方法；没有或未知时请明确说明。",
                "reason": "外部验证或生产部署必须控制仪器、批次和参考方法带来的域偏移。",
            }
        )
    return questions


def _requirement_updates(
    *,
    data_path: str | None = None,
    model_path: str | None = None,
    domain: str | None = None,
    analyte: str | None = None,
    unit: str | None = None,
    label_column: str | None = None,
    validation_goal: str | None = None,
    instrument: str | None = None,
    grouping_column: str | None = None,
    reference_method: str | None = None,
) -> dict[str, str]:
    values = {
        "data_path": data_path,
        "model_path": model_path,
        "domain": domain,
        "analyte": analyte,
        "unit": unit,
        "label_column": label_column,
        "instrument": instrument,
        "grouping_column": grouping_column,
        "reference_method": reference_method,
    }
    updates = {key: value.strip() for key, value in values.items() if _is_meaningful_requirement(value)}
    if isinstance(validation_goal, str) and validation_goal.strip():
        updates["validation_goal"] = _normalize_validation_goal(validation_goal)
    return updates


def workflow_agent_view(state: NIRWorkflowState | dict) -> dict[str, Any]:
    """Return the compact decision state needed by the model, without audit traces."""
    projected = {field: state.get(field) for field in _AGENT_VIEW_FIELDS if field in state}
    evidence = projected.get("attempt_evidence")
    if isinstance(evidence, dict):
        compact_evidence = dict(evidence)
        metrics_summary = compact_evidence.get("metrics_summary")
        if isinstance(metrics_summary, dict) and isinstance(metrics_summary.get("per_component"), list):
            compact_evidence["metrics_summary"] = {
                **metrics_summary,
                "per_component": metrics_summary["per_component"][:5],
            }
        projected["attempt_evidence"] = compact_evidence
    attempts = projected.get("attempts")
    if isinstance(attempts, list):
        projected["attempts"] = [
            {
                field: attempt.get(field)
                for field in (
                    "attempt",
                    "passed",
                    "grade",
                    "tool_name",
                    "method",
                    "pipeline_steps",
                    "protocol",
                    "validation_scope",
                    "metrics_summary",
                    "model_path",
                    "metrics_path",
                )
                if attempt.get(field) is not None
            }
            for attempt in attempts[-_RETRY_RECORD_LIMIT:]
            if isinstance(attempt, Mapping)
        ]
    observations = state.get("tool_observations")
    projected["tool_observation_count"] = len(observations) if isinstance(observations, list) else 0
    if isinstance(observations, list) and observations and isinstance(observations[-1], dict):
        projected["last_tool_observation"] = {field: observations[-1].get(field) for field in _AGENT_OBSERVATION_FIELDS if observations[-1].get(field) is not None}
    guard_events = state.get("response_guard_events")
    projected["response_guard_count"] = len(guard_events) if isinstance(guard_events, list) else 0
    if isinstance(guard_events, list) and guard_events and isinstance(guard_events[-1], dict):
        projected["last_response_guard"] = {
            "violations": list(guard_events[-1].get("violations") or []),
            "stage": guard_events[-1].get("stage"),
        }
    return projected


def _merge_evidence(state: NIRWorkflowState, evidence_ids: list[str] | None) -> list[str]:
    return list(dict.fromkeys([*(state.get("knowledge_evidence") or []), *(evidence_ids or [])]))


def _append_identifier(existing: list[str] | None, value: str | None) -> list[str]:
    normalized = value.strip() if isinstance(value, str) else ""
    return list(dict.fromkeys([*(existing or []), *([normalized] if normalized else [])]))


def attach_workflow_runtime(
    state: NIRWorkflowState,
    *,
    run_id: str | None = None,
    trace_id: str | None = None,
) -> NIRWorkflowState:
    """Associate runtime identifiers with an already-revisioned workflow update."""
    updated: dict[str, Any] = dict(state)
    updated["run_ids"] = _append_identifier(state.get("run_ids"), run_id)
    updated["trace_ids"] = _append_identifier(state.get("trace_ids"), trace_id)
    return updated  # type: ignore[return-value]


def record_tool_observation(
    state: NIRWorkflowState,
    *,
    name: str,
    status: str,
    stage_before: str | None,
    stage_after: str | None,
    code: str | None = None,
    requested_method: str | None = None,
    call_id: str | None = None,
    run_id: str | None = None,
    trace_id: str | None = None,
) -> NIRWorkflowState:
    """Persist one bounded NIR tool decision for post-run evaluation."""
    observation = {
        "name": name,
        "status": status,
        "stage_before": stage_before,
        "stage_after": stage_after,
        "code": code,
        "requested_method": requested_method,
        "call_id": call_id,
        "run_id": run_id,
        "trace_id": trace_id,
    }
    observations = [*(state.get("tool_observations") or []), observation][-_TOOL_OBSERVATION_LIMIT:]
    return _with_update(
        state,
        action="tool_call",
        tool_observations=observations,
        run_ids=_append_identifier(state.get("run_ids"), run_id),
        trace_ids=_append_identifier(state.get("trace_ids"), trace_id),
        event_details={
            "tool": name,
            "status": status,
            "stage_before": stage_before,
            "stage_after": stage_after,
            "code": code,
            "requested_method": requested_method,
            "call_id": call_id,
            "run_id": run_id,
            "trace_id": trace_id,
        },
    )


def record_response_guard(
    state: NIRWorkflowState,
    *,
    violations: list[str],
    message_id: str | None,
) -> NIRWorkflowState:
    """Persist a bounded audit event when an unsupported response is replaced."""

    guard_event = {
        "at": _now(),
        "stage": state.get("stage"),
        "message_id": message_id,
        "violations": list(dict.fromkeys(violations)),
    }
    events = [*(state.get("response_guard_events") or []), guard_event][-_RESPONSE_GUARD_LIMIT:]
    return _with_update(
        state,
        action="response_guard",
        response_guard_events=events,
        event_details={
            "stage": state.get("stage"),
            "message_id": message_id,
            "violations": guard_event["violations"],
        },
    )


def block_workflow_after_continuation_failure(
    state: NIRWorkflowState,
    *,
    required_action: str,
) -> NIRWorkflowState:
    """Fail closed after repeated attempts to skip a mandatory workflow action."""

    return _with_update(
        state,
        action="continuation_guard_exhausted",
        stage="blocked",
        next_action="report_best_effort",
        approval_status="not_required",
        event_details={
            "required_action": required_action,
            "outcome": "agent_failed_to_complete_required_workflow_action",
        },
    )


def block_workflow_after_retry_plan_mismatch(
    state: NIRWorkflowState,
    *,
    plan_id: str | None,
) -> NIRWorkflowState:
    """Fail closed after the agent repeats a retry-plan signature mismatch."""

    active_plan = state.get("retry_plan")
    invalid_plan: dict[str, Any] | None = None
    if isinstance(active_plan, Mapping):
        invalid_plan = {
            **active_plan,
            "status": "invalid",
            "invalid_reason": "repeated_execution_signature_mismatch",
        }
    retry_plans = [invalid_plan if invalid_plan is not None and isinstance(plan, Mapping) and plan.get("plan_id") == invalid_plan.get("plan_id") else plan for plan in (state.get("retry_plans") or [])]
    return _with_update(
        state,
        action="retry_plan_mismatch_exhausted",
        stage="blocked",
        next_action="report_best_effort",
        approval_status="not_required",
        retry_plan=invalid_plan,
        retry_plans=retry_plans,
        event_details={
            "plan_id": plan_id,
            "outcome": "repeated_execution_signature_mismatch",
        },
    )


def _event(action: str, **details: Any) -> dict[str, Any]:
    return {
        "action": action,
        "at": _now(),
        **{key: value for key, value in details.items() if value is not None},
    }


def _with_update(
    state: NIRWorkflowState,
    *,
    action: str,
    stage: str | None = None,
    next_action: str | None = None,
    event_details: dict[str, Any] | None = None,
    **updates: Any,
) -> NIRWorkflowState:
    updated: dict[str, Any] = dict(state)
    updated.update({key: value for key, value in updates.items() if value is not None})
    if stage is not None:
        updated["stage"] = stage
    if next_action is not None:
        updated["next_action"] = next_action
    updated["revision"] = int(state.get("revision", 0)) + 1
    updated["updated_at"] = _now()
    history = [*(state.get("history") or []), _event(action, **(event_details or {}))]
    updated["history"] = history[-_HISTORY_LIMIT:]
    return updated  # type: ignore[return-value]


def start_workflow(
    *,
    task_type: str,
    project_id: str | None = None,
    data_path: str | None = None,
    model_path: str | None = None,
    domain: str | None = None,
    analyte: str | None = None,
    unit: str | None = None,
    label_column: str | None = None,
    validation_goal: str | None = None,
    instrument: str | None = None,
    grouping_column: str | None = None,
    reference_method: str | None = None,
    max_attempts: int = 3,
) -> NIRWorkflowState:
    """Create a new NIR workflow and determine its first executable stage."""
    normalized_task = task_type.strip().lower()
    if normalized_task not in _SUPPORTED_TASK_TYPES:
        raise NIRWorkflowError(f"Unsupported task_type {task_type!r}; expected one of {sorted(_SUPPORTED_TASK_TYPES)}")
    if not 1 <= max_attempts <= 10:
        raise NIRWorkflowError("max_attempts must be between 1 and 10")

    normalized_requirements = _requirement_updates(
        data_path=data_path,
        model_path=model_path,
        domain=domain,
        analyte=analyte,
        unit=unit,
        label_column=label_column,
        validation_goal=validation_goal,
        instrument=instrument,
        grouping_column=grouping_column,
        reference_method=reference_method,
    )
    state: NIRWorkflowState = {
        "project_id": project_id or f"nir-{uuid.uuid4().hex[:12]}",
        "task_type": normalized_task,
        "stage": "intake",
        "revision": 1,
        "domain": normalized_requirements.get("domain"),
        "analyte": normalized_requirements.get("analyte"),
        "unit": normalized_requirements.get("unit"),
        "label_column": normalized_requirements.get("label_column"),
        "validation_goal": normalized_requirements.get("validation_goal"),
        "instrument": normalized_requirements.get("instrument"),
        "grouping_column": normalized_requirements.get("grouping_column"),
        "reference_method": normalized_requirements.get("reference_method"),
        "data_path": normalized_requirements.get("data_path"),
        "model_path": normalized_requirements.get("model_path"),
        "metrics_path": None,
        "audit_evidence": None,
        "attempt_evidence": None,
        "attempts": [],
        "reflection": None,
        "reflections": [],
        "retry_plan": None,
        "retry_plans": [],
        "knowledge_evidence": [],
        "run_ids": [],
        "trace_ids": [],
        "tool_observations": [],
        "response_guard_events": [],
        "attempt": 0,
        "max_attempts": max_attempts,
        "audit_status": "pending",
        "missing_inputs": [],
        "clarification_questions": [],
        "approval_status": "not_required",
        "next_action": "collect_prerequisites",
        "history": [],
        "updated_at": _now(),
    }
    missing = _prerequisite_missing(state)
    state["missing_inputs"] = missing
    if missing:
        state["next_action"] = "collect_prerequisites"
    elif normalized_task == "knowledge":
        state["stage"] = "execution"
        state["next_action"] = "search_knowledge"
    else:
        state["stage"] = "data_audit"
        state["next_action"] = "inspect_data"
    state["history"] = [_event("start", task_type=normalized_task)]
    return state


def transition_workflow(
    state: NIRWorkflowState,
    *,
    action: str,
    data_path: str | None = None,
    model_path: str | None = None,
    metrics_path: str | None = None,
    domain: str | None = None,
    analyte: str | None = None,
    unit: str | None = None,
    label_column: str | None = None,
    validation_goal: str | None = None,
    instrument: str | None = None,
    grouping_column: str | None = None,
    reference_method: str | None = None,
    audit_passed: bool | None = None,
    audit_evidence: dict[str, Any] | None = None,
    attempt_passed: bool | None = None,
    grade: str | None = None,
    missing_inputs: list[str] | None = None,
    evidence_ids: list[str] | None = None,
    attempt_evidence: dict[str, Any] | None = None,
    reflection_result: dict[str, Any] | None = None,
    retry_tool: str | None = None,
    retry_method: str | None = None,
    retry_pipeline_steps: Any = None,
    retry_model_args: Any = None,
    retry_rationale: str | None = None,
    expected_improvement: str | None = None,
    notes: str | None = None,
) -> NIRWorkflowState:
    """Apply one validated state transition to an existing workflow."""
    action = action.strip().lower()

    if action == "record_audit_evidence":
        if state["stage"] != "data_audit":
            raise NIRWorkflowError("record_audit_evidence is only allowed during data_audit")
        if not isinstance(audit_evidence, Mapping) or not audit_evidence:
            raise NIRWorkflowError("audit_evidence is required for record_audit_evidence")
        bounded_evidence = dict(audit_evidence)
        return _with_update(
            state,
            action=action,
            audit_evidence=bounded_evidence,
            event_details={
                "source_tool": bounded_evidence.get("source_tool"),
                "n_samples": bounded_evidence.get("n_samples"),
                "n_wavelengths": bounded_evidence.get("n_wavelengths"),
            },
        )

    if action == "set_requirements":
        requirement_updates = _requirement_updates(
            data_path=data_path,
            model_path=model_path,
            domain=domain,
            analyte=analyte,
            unit=unit,
            label_column=label_column,
            validation_goal=validation_goal,
            instrument=instrument,
            grouping_column=grouping_column,
            reference_method=reference_method,
        )
        updated = _with_update(
            state,
            action=action,
            **requirement_updates,
        )
        prerequisites = _prerequisite_missing(updated)
        if prerequisites:
            updated["missing_inputs"] = prerequisites
            updated["clarification_questions"] = []
            updated["stage"] = "intake"
            updated["next_action"] = "collect_prerequisites"
            return updated
        if updated.get("audit_status") != "passed":
            updated["missing_inputs"] = []
            updated["clarification_questions"] = []
            updated["stage"] = "data_audit"
            updated["next_action"] = "inspect_data"
            return updated
        decision_missing = _decision_missing(updated)
        updated["missing_inputs"] = decision_missing
        updated["clarification_questions"] = _clarification_questions(decision_missing)
        updated["stage"] = "clarification" if decision_missing else "planning"
        updated["next_action"] = "ask_targeted_clarification" if decision_missing else "prepare_analysis_plan"
        return updated

    if action == "record_audit":
        if state["stage"] != "data_audit":
            raise NIRWorkflowError("record_audit is only allowed during data_audit")
        if audit_passed is None:
            raise NIRWorkflowError("audit_passed is required for record_audit")
        requirement_updates = _requirement_updates(
            data_path=data_path,
            model_path=model_path,
            domain=domain,
            analyte=analyte,
            unit=unit,
            label_column=label_column,
            validation_goal=validation_goal,
            instrument=instrument,
            grouping_column=grouping_column,
            reference_method=reference_method,
        )
        if not audit_passed:
            reasons = missing_inputs or ["data_quality_issue"]
            return _with_update(
                state,
                action=action,
                stage="blocked",
                next_action="resolve_data_issues",
                audit_status="failed",
                missing_inputs=reasons,
                clarification_questions=[],
                event_details={"passed": False, "notes": notes},
                **requirement_updates,
            )
        preview: NIRWorkflowState = {
            **state,
            **requirement_updates,
            "audit_status": "passed",
        }
        decision_missing = _decision_missing(preview)
        if decision_missing:
            return _with_update(
                state,
                action=action,
                stage="clarification",
                next_action="ask_targeted_clarification",
                audit_status="passed",
                missing_inputs=decision_missing,
                clarification_questions=_clarification_questions(decision_missing),
                event_details={"passed": True, "notes": notes},
                **requirement_updates,
            )
        return _with_update(
            state,
            action=action,
            stage="planning",
            next_action="prepare_analysis_plan",
            audit_status="passed",
            missing_inputs=[],
            clarification_questions=[],
            event_details={"passed": True, "notes": notes},
            **requirement_updates,
        )

    if action == "plan_ready":
        if state["stage"] != "planning":
            raise NIRWorkflowError("plan_ready is only allowed during planning")
        if state.get("attempt", 0) > 0:
            plan = state.get("retry_plan")
            if not isinstance(plan, Mapping) or plan.get("source_attempt") != state["attempt"]:
                raise NIRWorkflowError("A retry plan bound to the latest failed attempt is required before plan_ready")
            if plan.get("status") != "ready":
                raise NIRWorkflowError("The current retry plan is not ready for execution")
        if state.get("validation_goal") == "exploratory":
            return _with_update(
                state,
                action=action,
                stage="completed",
                next_action="none",
                approval_status="not_required",
                event_details={"notes": notes, "outcome": "exploratory_findings_ready"},
            )
        return _with_update(
            state,
            action=action,
            stage="execution",
            next_action="execute_nir_tools",
            event_details={"notes": notes},
        )

    if action == "record_attempt":
        if state["stage"] != "execution":
            raise NIRWorkflowError("record_attempt requires execution stage")
        if attempt_passed is None:
            raise NIRWorkflowError("attempt_passed is required for record_attempt")
        attempt = state["attempt"] + 1
        evidence = attempt_evidence if isinstance(attempt_evidence, Mapping) else {}
        active_retry_plan = state.get("retry_plan") if state.get("attempt", 0) > 0 else None
        if isinstance(active_retry_plan, Mapping):
            canonical_steps, canonical_model_args, canonical_signature = retry_execution_signature(
                tool_name=str(active_retry_plan.get("tool_name") or ""),
                method=str(active_retry_plan.get("method") or "auto"),
                pipeline_steps=active_retry_plan.get("pipeline_steps"),
                model_args=active_retry_plan.get("model_args"),
            )
            accepted_signatures = {
                active_retry_plan.get("execution_signature"),
                canonical_signature,
            }
            if evidence.get("execution_signature") not in accepted_signatures:
                raise NIRWorkflowError("The modeling attempt does not match the active retry plan")
            if evidence.get("execution_signature") == canonical_signature:
                active_retry_plan = {
                    **active_retry_plan,
                    "pipeline_steps": canonical_steps,
                    "model_args": canonical_model_args,
                    "execution_signature": canonical_signature,
                }
        attempt_record = {
            "attempt": attempt,
            "passed": attempt_passed,
            "grade": grade,
            "tool_name": evidence.get("tool_name"),
            "method": evidence.get("method"),
            "pipeline_steps": evidence.get("pipeline_steps") or [],
            "model_args": evidence.get("model_args") or {},
            "execution_signature": evidence.get("execution_signature"),
            "protocol": evidence.get("protocol"),
            "validation_scope": evidence.get("validation_scope"),
            "metrics_summary": evidence.get("metrics_summary") or {},
            "model_path": model_path,
            "metrics_path": metrics_path,
            "retry_plan_id": active_retry_plan.get("plan_id") if isinstance(active_retry_plan, Mapping) else None,
        }
        attempts = [*(state.get("attempts") or []), attempt_record][-_RETRY_RECORD_LIMIT:]
        executed_plan = None
        retry_plans = list(state.get("retry_plans") or [])
        if isinstance(active_retry_plan, Mapping):
            executed_plan = {**active_retry_plan, "status": "executed", "executed_attempt": attempt}
            retry_plans = [
                *[executed_plan if plan.get("plan_id") == executed_plan.get("plan_id") else plan for plan in retry_plans if isinstance(plan, Mapping)],
            ][-_RETRY_RECORD_LIMIT:]
        common = {
            "attempt": attempt,
            "model_path": model_path,
            "metrics_path": metrics_path,
            "attempt_evidence": attempt_evidence,
            "attempts": attempts,
            "retry_plan": executed_plan,
            "retry_plans": retry_plans,
            "event_details": {
                "attempt": attempt,
                "passed": attempt_passed,
                "grade": grade,
                "model_path": model_path,
                "metrics_path": metrics_path,
                "attempt_evidence": attempt_evidence,
                "execution_signature": evidence.get("execution_signature"),
                "retry_plan_id": active_retry_plan.get("plan_id") if isinstance(active_retry_plan, Mapping) else None,
                "notes": notes,
            },
        }
        if attempt_passed:
            return _with_update(
                state,
                action=action,
                stage="review",
                next_action="request_user_approval",
                approval_status="pending",
                **common,
            )
        return _with_update(
            state,
            action=action,
            stage="evaluation",
            next_action="reflect_on_attempt",
            approval_status="not_required",
            **common,
        )

    if action == "record_reflection":
        if state["stage"] != "evaluation":
            raise NIRWorkflowError("record_reflection requires evaluation stage")
        if not isinstance(reflection_result, Mapping):
            raise NIRWorkflowError("reflection_result is required for record_reflection")
        if reflection_result.get("attempt") != state["attempt"]:
            raise NIRWorkflowError("Reflection attempt must match the latest modeling attempt")
        requested_retry = reflection_result.get("should_retry")
        if not isinstance(requested_retry, bool):
            raise NIRWorkflowError("Reflection should_retry must be boolean")
        budget_available = state["attempt"] < state["max_attempts"]
        should_retry = requested_retry and budget_available
        knowledge_hint = reflection_result.get("knowledge_hint")
        knowledge_required = isinstance(knowledge_hint, Mapping)
        reflection = {
            "reflection_id": f"reflection-{state['attempt']}-{int(state.get('revision', 0)) + 1}",
            "source_attempt": state["attempt"],
            "metrics_path": state.get("metrics_path"),
            "should_retry": should_retry,
            "stop_reason": None if should_retry else ("retry_budget_exhausted" if not budget_available else "reflection_stop"),
            "reason": str(reflection_result.get("reason") or ""),
            "diagnostics": dict(reflection_result.get("diagnostics") or {}) if isinstance(reflection_result.get("diagnostics"), Mapping) else {},
            "fallback_suggestion_steps": list(reflection_result.get("fallback_suggestion_steps") or []),
            "lv_adjustment": dict(reflection_result.get("lv_adjustment") or {}) if isinstance(reflection_result.get("lv_adjustment"), Mapping) else {},
            "current_quality": dict(reflection_result.get("current_quality") or {}) if isinstance(reflection_result.get("current_quality"), Mapping) else {},
            "best_so_far": dict(reflection_result.get("best_so_far") or {}) if isinstance(reflection_result.get("best_so_far"), Mapping) else {},
            "knowledge_required": knowledge_required,
            "knowledge_query": knowledge_hint.get("query") if isinstance(knowledge_hint, Mapping) else None,
            "knowledge_evidence_count": len(state.get("knowledge_evidence") or []),
        }
        reflections = [*(state.get("reflections") or []), reflection][-_RETRY_RECORD_LIMIT:]
        if not should_retry:
            return _with_update(
                state,
                action=action,
                stage="blocked",
                next_action="report_best_effort",
                reflection=reflection,
                reflections=reflections,
                retry_plan={},
                event_details={
                    "attempt": state["attempt"],
                    "reflection_id": reflection["reflection_id"],
                    "should_retry": False,
                    "stop_reason": reflection["stop_reason"],
                    "metrics_path": reflection["metrics_path"],
                },
            )
        return _with_update(
            state,
            action=action,
            stage="knowledge" if knowledge_required else "planning",
            next_action="retrieve_evidence_for_retry" if knowledge_required else "prepare_retry_plan",
            reflection=reflection,
            reflections=reflections,
            retry_plan={},
            event_details={
                "attempt": state["attempt"],
                "reflection_id": reflection["reflection_id"],
                "should_retry": True,
                "knowledge_required": knowledge_required,
                "diagnostics": reflection["diagnostics"],
                "metrics_path": reflection["metrics_path"],
            },
        )

    if action == "knowledge_retrieved":
        if state["task_type"] == "knowledge" and state["stage"] == "execution":
            return _with_update(
                state,
                action=action,
                stage="completed",
                next_action="none",
                knowledge_evidence=_merge_evidence(state, evidence_ids),
                event_details={"notes": notes, "evidence_ids": evidence_ids},
            )
        if state["stage"] != "knowledge":
            raise NIRWorkflowError("knowledge_retrieved requires a knowledge task or the knowledge retry stage")
        reflection = state.get("reflection")
        if not isinstance(reflection, Mapping) or reflection.get("should_retry") is not True:
            raise NIRWorkflowError("knowledge_retrieved requires a retryable reflection")
        merged_evidence = _merge_evidence(state, evidence_ids)
        if reflection.get("knowledge_required") and len(merged_evidence) <= int(reflection.get("knowledge_evidence_count") or 0):
            return _with_update(
                state,
                action=action,
                stage="blocked",
                next_action="report_best_effort",
                knowledge_evidence=merged_evidence,
                event_details={
                    "notes": notes,
                    "evidence_ids": evidence_ids,
                    "outcome": "required_retry_evidence_unavailable",
                },
            )
        return _with_update(
            state,
            action=action,
            stage="planning",
            next_action="prepare_retry_plan",
            knowledge_evidence=merged_evidence,
            event_details={"notes": notes, "evidence_ids": evidence_ids},
        )

    if action == "record_retry_plan":
        if state["stage"] != "planning" or state.get("attempt", 0) < 1:
            raise NIRWorkflowError("record_retry_plan requires retry planning after a failed attempt")
        reflection = state.get("reflection")
        if not isinstance(reflection, Mapping) or reflection.get("source_attempt") != state["attempt"]:
            raise NIRWorkflowError("A reflection bound to the latest failed attempt is required")
        if reflection.get("should_retry") is not True:
            raise NIRWorkflowError("The latest reflection does not allow another retry")
        if reflection.get("knowledge_required"):
            prior_count = int(reflection.get("knowledge_evidence_count") or 0)
            if len(state.get("knowledge_evidence") or []) <= prior_count:
                raise NIRWorkflowError("The reflection requires new knowledge evidence before retry planning")
        rationale = str(retry_rationale or "").strip()
        improvement = str(expected_improvement or "").strip()
        if not rationale or not improvement:
            raise NIRWorkflowError("retry_rationale and expected_improvement are required")
        normalized_retry_tool = str(retry_tool or "").strip()
        if normalized_retry_tool not in _RETRY_TOOLS_BY_TASK.get(state["task_type"], frozenset()):
            raise NIRWorkflowError(f"Retry tool {normalized_retry_tool!r} is not compatible with task type {state['task_type']!r}")
        normalized_steps, normalized_model_args, signature = retry_execution_signature(
            tool_name=normalized_retry_tool,
            method=retry_method,
            pipeline_steps=retry_pipeline_steps,
            model_args=retry_model_args,
        )
        previous_signature = None
        if isinstance(state.get("attempt_evidence"), Mapping):
            previous_signature = state["attempt_evidence"].get("execution_signature")
        if previous_signature and signature == previous_signature:
            raise NIRWorkflowError("Retry plan must materially change the previous modeling execution")
        evidence_start = int(reflection.get("knowledge_evidence_count") or 0)
        plan = {
            "plan_id": f"retry-{state['attempt'] + 1}-{signature[:12]}",
            "source_attempt": state["attempt"],
            "target_attempt": state["attempt"] + 1,
            "reflection_id": reflection.get("reflection_id"),
            "tool_name": normalized_retry_tool,
            "method": normalize_retry_model_method(normalized_retry_tool, retry_method),
            "pipeline_steps": normalized_steps,
            "model_args": normalized_model_args,
            "execution_signature": signature,
            "previous_execution_signature": previous_signature,
            "diagnostics": dict(reflection.get("diagnostics") or {}),
            "evidence_ids": list(state.get("knowledge_evidence") or [])[evidence_start:],
            "rationale": rationale,
            "expected_improvement": improvement,
            "status": "ready",
        }
        prior_plans = [
            {
                **prior_plan,
                "status": "superseded",
                "superseded_by": plan["plan_id"],
            }
            if isinstance(prior_plan, Mapping) and prior_plan.get("source_attempt") == state["attempt"] and prior_plan.get("status") == "ready"
            else prior_plan
            for prior_plan in (state.get("retry_plans") or [])
        ]
        retry_plans = [*prior_plans, plan][-_RETRY_RECORD_LIMIT:]
        return _with_update(
            state,
            action=action,
            retry_plan=plan,
            retry_plans=retry_plans,
            next_action="activate_retry_plan",
            event_details={
                "plan_id": plan["plan_id"],
                "source_attempt": plan["source_attempt"],
                "target_attempt": plan["target_attempt"],
                "reflection_id": plan["reflection_id"],
                "tool_name": plan["tool_name"],
                "method": plan["method"],
                "pipeline_steps": plan["pipeline_steps"],
                "model_args": plan["model_args"],
                "execution_signature": signature,
                "previous_execution_signature": previous_signature,
                "evidence_ids": plan["evidence_ids"],
                "rationale": rationale,
                "expected_improvement": improvement,
            },
        )

    if action == "approve":
        if state["stage"] != "review" or state["approval_status"] != "pending":
            raise NIRWorkflowError("approve requires a pending review")
        return _with_update(
            state,
            action=action,
            stage="approved",
            next_action="register_or_deliver_model",
            approval_status="approved",
            event_details={"notes": notes},
        )

    if action == "reject":
        if state["stage"] != "review" or state["approval_status"] != "pending":
            raise NIRWorkflowError("reject requires a pending review")
        return _with_update(
            state,
            action=action,
            stage="blocked",
            next_action="await_user_direction",
            approval_status="rejected",
            event_details={"notes": notes},
        )

    if action == "registered":
        if state["stage"] != "approved":
            raise NIRWorkflowError("registered requires an approved workflow")
        return _with_update(
            state,
            action=action,
            stage="registered",
            next_action="complete_workflow",
            event_details={"notes": notes},
        )

    if action == "complete":
        inspection_ready = state["task_type"] == "inspection" and state["stage"] == "data_audit" and isinstance(state.get("audit_evidence"), Mapping) and state["audit_evidence"].get("source_tool") == "nir_inspect"
        if state["stage"] not in {"approved", "registered", "execution"} and not inspection_ready:
            raise NIRWorkflowError("complete requires approved, registered, or execution stage, or a successful inspection audit")
        return _with_update(
            state,
            action=action,
            stage="completed",
            next_action="none",
            event_details={"notes": notes},
        )

    raise NIRWorkflowError(f"Unsupported workflow action: {action!r}")


def _tool_payload(state: NIRWorkflowState) -> str:
    return json.dumps(
        {
            "status": "ok",
            "workflow": workflow_agent_view(state),
            "stage": state["stage"],
            "next_action": state["next_action"],
            "missing_inputs": state["missing_inputs"],
            "clarification_questions": state.get("clarification_questions", []),
        },
        ensure_ascii=False,
    )


def registration_is_approved(state: dict | None) -> bool:
    """Return whether workflow state permits model registration."""
    return bool(state and state.get("approval_status") == "approved" and state.get("stage") in {"approved", "registered"} and isinstance(state.get("attempt_evidence"), dict))


def _latest_user_explicitly_approved(runtime: Runtime) -> bool:
    messages = (runtime.state or {}).get("messages") or []
    for message in reversed(messages):
        if not isinstance(message, HumanMessage):
            continue
        if message.additional_kwargs.get("hide_from_ui"):
            continue
        text = get_original_user_content_text(message.content, message.additional_kwargs)
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in _APPROVAL_DENIAL_PATTERNS):
            return False
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in _APPROVAL_PATTERNS):
            return True
    return False


def _runtime_identifier(runtime: Runtime, key: str) -> str | None:
    context = getattr(runtime, "context", None)
    if not isinstance(context, dict):
        return None
    value = context.get(key)
    return str(value) if value else None


@tool("nir_workflow", parse_docstring=True)
def nir_workflow_tool(
    runtime: Runtime,
    action: str,
    task_type: str | None = None,
    project_id: str | None = None,
    data_path: str | None = None,
    model_path: str | None = None,
    metrics_path: str | None = None,
    domain: str | None = None,
    analyte: str | None = None,
    unit: str | None = None,
    label_column: str | None = None,
    validation_goal: str | None = None,
    instrument: str | None = None,
    grouping_column: str | None = None,
    reference_method: str | None = None,
    max_attempts: int = 3,
    audit_passed: bool | None = None,
    attempt_passed: bool | None = None,
    grade: str | None = None,
    missing_inputs: list[str] | None = None,
    evidence_ids: list[str] | None = None,
    retry_tool: str | None = None,
    retry_method: str | None = None,
    retry_pipeline_steps: str | None = None,
    retry_model_args: str | None = None,
    retry_rationale: str | None = None,
    expected_improvement: str | None = None,
    notes: str | None = None,
) -> Command:
    """Create or advance the durable state of a NIR agent workflow.

    Call ``start`` before using domain tools, then explicitly record requirements,
    data audit, planning, approval, and final completion decisions. Runtime
    middleware automatically records successful modeling attempts, knowledge
    retrieval, and model registration. A passed model enters ``review`` and
    cannot be registered until the user is recorded with ``approve``.

    Args:
        action: Workflow action: start, status, set_requirements, record_audit,
            plan_ready, record_attempt, record_retry_plan, knowledge_retrieved,
            approve, reject, registered, or complete.
        task_type: For start: analysis, calibration, multi_modeling, compare,
            classification, prediction, inspection, or knowledge.
        project_id: Optional stable identifier for a new workflow.
        data_path: Input spectral data path.
        model_path: Model artifact path.
        metrics_path: Metrics artifact path for a completed attempt.
        domain: NIR application domain.
        analyte: Property being predicted or analysed.
        unit: Reference/prediction unit.
        label_column: Category-label column for classification tasks.
        validation_goal: Exploratory, internal_holdout, external_validation, or
            production. This controls the validation claim and extra context.
        instrument: Instrument model or an explicit ``unknown``.
        grouping_column: Batch/season/origin grouping column, or ``none``.
        reference_method: Laboratory reference method or an explicit ``unknown``.
        max_attempts: Retry budget for a new workflow (1-10).
        audit_passed: Required by record_audit.
        attempt_passed: Required by record_attempt when used outside middleware.
        grade: Optional quality grade for a manually recorded attempt.
        missing_inputs: Missing or invalid fields found by data audit.
        evidence_ids: Stable document or source identifiers used for a knowledge retry.
        retry_tool: Modeling tool that the next retry will execute.
        retry_method: Model family bound to the next retry.
        retry_pipeline_steps: JSON preprocessing pipeline bound to the next retry.
        retry_model_args: JSON object containing every additional decision parameter.
        retry_rationale: Diagnostic/evidence-based reason for changing the execution.
        expected_improvement: Metric or failure mode the retry is expected to improve.
        notes: Short evidence or decision note stored in workflow history.
    """
    current = (runtime.state or {}).get("nir_workflow")
    try:
        normalized_action = action.strip().lower()
        if normalized_action == "status":
            if current is None:
                content = json.dumps(
                    {"status": "empty", "next_action": "start_workflow"},
                    ensure_ascii=False,
                )
                return Command(update={"messages": [ToolMessage(content=content, tool_call_id=runtime.tool_call_id)]})
            updated = current
        elif normalized_action == "start":
            if current and current.get("stage") not in {"completed", "blocked"}:
                raise NIRWorkflowError("An active NIR workflow already exists; complete or reject it before starting another")
            if not task_type:
                raise NIRWorkflowError("task_type is required for start")
            updated = start_workflow(
                task_type=task_type,
                project_id=project_id,
                data_path=data_path,
                model_path=model_path,
                domain=domain,
                analyte=analyte,
                unit=unit,
                label_column=label_column,
                validation_goal=validation_goal,
                instrument=instrument,
                grouping_column=grouping_column,
                reference_method=reference_method,
                max_attempts=max_attempts,
            )
            if current:
                updated["revision"] = int(current.get("revision", 0)) + 1
        else:
            if current is None:
                raise NIRWorkflowError("No active NIR workflow; call action='start' first")
            if normalized_action == "approve" and not _latest_user_explicitly_approved(runtime):
                raise NIRWorkflowError("approve requires explicit authorization in the latest user message")
            updated = transition_workflow(
                current,
                action=normalized_action,
                data_path=data_path,
                model_path=model_path,
                metrics_path=metrics_path,
                domain=domain,
                analyte=analyte,
                unit=unit,
                label_column=label_column,
                validation_goal=validation_goal,
                instrument=instrument,
                grouping_column=grouping_column,
                reference_method=reference_method,
                audit_passed=audit_passed,
                attempt_passed=attempt_passed,
                grade=grade,
                missing_inputs=missing_inputs,
                evidence_ids=evidence_ids,
                retry_tool=retry_tool,
                retry_method=retry_method,
                retry_pipeline_steps=retry_pipeline_steps,
                retry_model_args=retry_model_args,
                retry_rationale=retry_rationale,
                expected_improvement=expected_improvement,
                notes=notes,
            )
        if updated is not current:
            updated = attach_workflow_runtime(
                updated,
                run_id=_runtime_identifier(runtime, "run_id"),
                trace_id=_runtime_identifier(runtime, "deerflow_trace_id"),
            )
        update: dict[str, Any] = {"messages": [ToolMessage(content=_tool_payload(updated), tool_call_id=runtime.tool_call_id)]}
        if updated is not current:
            update["nir_workflow"] = updated
        return Command(update=update)
    except NIRWorkflowError as exc:
        content = json.dumps(
            {"status": "error", "error": str(exc)},
            ensure_ascii=False,
        )
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        content=content,
                        tool_call_id=runtime.tool_call_id,
                        status="error",
                    )
                ]
            }
        )
