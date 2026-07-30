"""Deterministic control plane for NIR agent workflows.

The language model can choose domain actions, but it cannot skip required
inputs, retry budgets, or the approval gate encoded here. State is persisted in
``ThreadState.nir_workflow`` so a workflow survives multi-turn conversations.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import UTC, datetime
from typing import Any

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command

from deerflow.agents.thread_state import NIRWorkflowState
from deerflow.tools.types import Runtime
from deerflow.utils.messages import get_original_user_content_text

_SUPPORTED_TASK_TYPES = frozenset({"analysis", "calibration", "multi_modeling", "compare", "prediction", "inspection", "knowledge"})
_REQUIRED_INPUTS = {
    "analysis": ("data_path", "analyte", "unit", "domain"),
    "calibration": ("data_path", "analyte", "unit", "domain"),
    "multi_modeling": ("data_path", "analyte", "unit", "domain"),
    "compare": ("data_path", "analyte", "unit", "domain"),
    "prediction": ("data_path", "model_path"),
    "inspection": ("data_path",),
    "knowledge": (),
}
_HISTORY_LIMIT = 50
_TOOL_OBSERVATION_LIMIT = 100
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


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _missing_inputs(state: NIRWorkflowState) -> list[str]:
    required = _REQUIRED_INPUTS.get(state["task_type"], ())
    return [field for field in required if not state.get(field)]


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
    max_attempts: int = 3,
) -> NIRWorkflowState:
    """Create a new NIR workflow and determine its first executable stage."""
    normalized_task = task_type.strip().lower()
    if normalized_task not in _SUPPORTED_TASK_TYPES:
        raise NIRWorkflowError(f"Unsupported task_type {task_type!r}; expected one of {sorted(_SUPPORTED_TASK_TYPES)}")
    if not 1 <= max_attempts <= 10:
        raise NIRWorkflowError("max_attempts must be between 1 and 10")

    state: NIRWorkflowState = {
        "project_id": project_id or f"nir-{uuid.uuid4().hex[:12]}",
        "task_type": normalized_task,
        "stage": "intake",
        "revision": 1,
        "domain": domain,
        "analyte": analyte,
        "unit": unit,
        "data_path": data_path,
        "model_path": model_path,
        "metrics_path": None,
        "knowledge_evidence": [],
        "run_ids": [],
        "trace_ids": [],
        "tool_observations": [],
        "attempt": 0,
        "max_attempts": max_attempts,
        "missing_inputs": [],
        "approval_status": "not_required",
        "next_action": "collect_requirements",
        "history": [],
        "updated_at": _now(),
    }
    missing = _missing_inputs(state)
    state["missing_inputs"] = missing
    if missing:
        state["next_action"] = "collect_requirements"
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
    audit_passed: bool | None = None,
    attempt_passed: bool | None = None,
    grade: str | None = None,
    missing_inputs: list[str] | None = None,
    evidence_ids: list[str] | None = None,
    notes: str | None = None,
) -> NIRWorkflowState:
    """Apply one validated state transition to an existing workflow."""
    action = action.strip().lower()

    if action == "set_requirements":
        updated = _with_update(
            state,
            action=action,
            data_path=data_path,
            model_path=model_path,
            domain=domain,
            analyte=analyte,
            unit=unit,
        )
        missing = _missing_inputs(updated)
        updated["missing_inputs"] = missing
        updated["stage"] = "intake" if missing else "data_audit"
        updated["next_action"] = "collect_requirements" if missing else "inspect_data"
        return updated

    if action == "record_audit":
        if state["stage"] != "data_audit":
            raise NIRWorkflowError("record_audit is only allowed during data_audit")
        if audit_passed is None:
            raise NIRWorkflowError("audit_passed is required for record_audit")
        if not audit_passed:
            reasons = missing_inputs or ["data_quality_issue"]
            return _with_update(
                state,
                action=action,
                stage="blocked",
                next_action="resolve_data_issues",
                missing_inputs=reasons,
                event_details={"passed": False, "notes": notes},
            )
        return _with_update(
            state,
            action=action,
            stage="planning",
            next_action="prepare_analysis_plan",
            missing_inputs=[],
            event_details={"passed": True, "notes": notes},
        )

    if action == "plan_ready":
        if state["stage"] != "planning":
            raise NIRWorkflowError("plan_ready is only allowed during planning")
        return _with_update(
            state,
            action=action,
            stage="execution",
            next_action="execute_nir_tools",
            event_details={"notes": notes},
        )

    if action == "record_attempt":
        if state["stage"] not in {"execution", "evaluation"}:
            raise NIRWorkflowError("record_attempt requires execution or evaluation stage")
        if attempt_passed is None:
            raise NIRWorkflowError("attempt_passed is required for record_attempt")
        attempt = state["attempt"] + 1
        common = {
            "attempt": attempt,
            "model_path": model_path,
            "metrics_path": metrics_path,
            "event_details": {
                "attempt": attempt,
                "passed": attempt_passed,
                "grade": grade,
                "model_path": model_path,
                "metrics_path": metrics_path,
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
        if attempt < state["max_attempts"]:
            return _with_update(
                state,
                action=action,
                stage="knowledge",
                next_action="retrieve_evidence_for_retry",
                **common,
            )
        return _with_update(
            state,
            action=action,
            stage="blocked",
            next_action="report_best_effort",
            approval_status="not_required",
            **common,
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
        return _with_update(
            state,
            action=action,
            stage="planning",
            next_action="prepare_retry_plan",
            knowledge_evidence=_merge_evidence(state, evidence_ids),
            event_details={"notes": notes, "evidence_ids": evidence_ids},
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
        if state["stage"] not in {"approved", "registered", "execution"}:
            raise NIRWorkflowError("complete requires approved, registered, or execution stage")
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
            "workflow": state,
            "stage": state["stage"],
            "next_action": state["next_action"],
            "missing_inputs": state["missing_inputs"],
        },
        ensure_ascii=False,
    )


def registration_is_approved(state: dict | None) -> bool:
    """Return whether workflow state permits model registration."""
    return bool(state and state.get("approval_status") == "approved" and state.get("stage") in {"approved", "registered"})


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
    max_attempts: int = 3,
    audit_passed: bool | None = None,
    attempt_passed: bool | None = None,
    grade: str | None = None,
    missing_inputs: list[str] | None = None,
    evidence_ids: list[str] | None = None,
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
            plan_ready, record_attempt, knowledge_retrieved, approve, reject,
            registered, or complete.
        task_type: For start: analysis, calibration, multi_modeling, compare,
            prediction, inspection, or knowledge.
        project_id: Optional stable identifier for a new workflow.
        data_path: Input spectral data path.
        model_path: Model artifact path.
        metrics_path: Metrics artifact path for a completed attempt.
        domain: NIR application domain.
        analyte: Property being predicted or analysed.
        unit: Reference/prediction unit.
        max_attempts: Retry budget for a new workflow (1-10).
        audit_passed: Required by record_audit.
        attempt_passed: Required by record_attempt when used outside middleware.
        grade: Optional quality grade for a manually recorded attempt.
        missing_inputs: Missing or invalid fields found by data audit.
        evidence_ids: Stable document or source identifiers used for a knowledge retry.
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
                audit_passed=audit_passed,
                attempt_passed=attempt_passed,
                grade=grade,
                missing_inputs=missing_inputs,
                evidence_ids=evidence_ids,
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
