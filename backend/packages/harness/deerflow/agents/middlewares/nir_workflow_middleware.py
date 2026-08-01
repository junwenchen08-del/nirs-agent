"""Runtime enforcement and automatic advancement for NIR workflows."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from deerflow.agents.thread_state import NIRWorkflowState
from deerflow.community.nir.response_grounding import (
    render_grounded_nir_response,
    required_nir_workflow_action,
    validate_nir_response,
)
from deerflow.community.nir.workflow import (
    NIRWorkflowError,
    block_workflow_after_continuation_failure,
    block_workflow_after_retry_plan_mismatch,
    normalize_retry_model_method,
    record_response_guard,
    record_tool_observation,
    retry_execution_signature,
    transition_workflow,
)
from deerflow.tools.types import Runtime
from deerflow.trace_context import DEERFLOW_TRACE_METADATA_KEY
from deerflow.utils.messages import get_original_user_content_text, message_content_to_text

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ToolPolicy:
    stages: frozenset[str]
    task_types: frozenset[str]


_MODEL_TASKS = frozenset({"analysis", "calibration", "multi_modeling", "compare"})
_DATA_TASKS = frozenset({"analysis", "calibration", "multi_modeling", "compare", "prediction", "inspection"})
_EXECUTION_TASKS = frozenset({"analysis", "calibration", "multi_modeling", "compare", "prediction"})

_TOOL_POLICIES: dict[str, _ToolPolicy] = {
    "nir_load_data": _ToolPolicy(frozenset({"data_audit"}), _DATA_TASKS),
    "nir_inspect": _ToolPolicy(frozenset({"data_audit"}), _DATA_TASKS),
    "nir_preprocess": _ToolPolicy(frozenset({"execution"}), _EXECUTION_TASKS),
    "nir_train_auto_split_model": _ToolPolicy(frozenset({"execution"}), frozenset({"analysis", "calibration"})),
    "nir_train_model": _ToolPolicy(frozenset({"execution"}), frozenset({"analysis", "calibration"})),
    "nir_train_partitioned_model": _ToolPolicy(frozenset({"execution"}), frozenset({"analysis", "calibration"})),
    "nir_train_multi_model": _ToolPolicy(frozenset({"execution"}), frozenset({"multi_modeling"})),
    "nir_analyze": _ToolPolicy(frozenset({"execution"}), frozenset({"analysis", "calibration"})),
    "nir_analyze_collection": _ToolPolicy(frozenset({"execution"}), frozenset({"analysis", "calibration"})),
    "nir_compare": _ToolPolicy(frozenset({"execution"}), _MODEL_TASKS),
    "nir_predict": _ToolPolicy(frozenset({"execution"}), frozenset({"prediction"})),
    "nir_reflect": _ToolPolicy(frozenset({"evaluation"}), _MODEL_TASKS),
    "nir_search_knowledge": _ToolPolicy(frozenset({"execution", "knowledge"}), frozenset({"knowledge", *_MODEL_TASKS})),
    "nir_register_model": _ToolPolicy(frozenset({"approved"}), _MODEL_TASKS),
}
_OBSERVED_NIR_TOOLS = frozenset({*_TOOL_POLICIES, "nir_workflow"})

_MODELING_TOOLS = frozenset({"nir_train_auto_split_model", "nir_train_model", "nir_train_partitioned_model", "nir_train_multi_model", "nir_analyze", "nir_analyze_collection", "nir_compare"})
_INTERNAL_HOLDOUT_TOOLS = _MODELING_TOOLS - {"nir_train_partitioned_model"}
_EXTERNAL_VALIDATION_TOOLS = frozenset({"nir_train_partitioned_model"})
_TOOL_VALIDATION_PROTOCOL = {
    "nir_train_auto_split_model": ("deterministic_auto_split_holdout", "independent_holdout_not_external"),
    "nir_train_model": ("random_three_way_holdout", "independent_holdout_not_external"),
    "nir_train_partitioned_model": ("named_partition_external_validation", "independent_external_validation"),
    "nir_train_multi_model": ("multi_target_random_three_way_holdout", "independent_holdout_not_external"),
    "nir_analyze": ("automated_analysis_three_way_holdout", "independent_holdout_not_external"),
    "nir_analyze_collection": ("mat_collection_sequential_compact", "independent_holdout_not_external"),
    "nir_compare": ("preprocessing_comparison_three_way_holdout", "independent_holdout_not_external"),
}
_MAX_WORKFLOW_CONTINUATION_REMINDERS = 2
_GROUPING_SPLIT_CONFLICT_CODE = "nir_grouping_split_column_conflict"
_MODEL_SUBSTITUTION_CODE = "nir_model_substitution_requires_approval"
_VALIDATION_GOAL_CONFLICT_CODE = "nir_validation_goal_conflict"
_RETRY_PLAN_MISMATCH_CODE = "nir_retry_plan_mismatch"
_RETRY_PLAN_MISMATCH_EXHAUSTED_CODE = "nir_retry_plan_mismatch_exhausted"
_MODEL_ALIASES = {
    "cnn": ("cnn", "1dcnn", "一维卷积"),
    "mlp": ("mlp", "多层感知机"),
    "pls": ("pls", "偏最小二乘"),
    "pcr": ("pcr", "主成分回归"),
    "svr": ("svr", "支持向量回归"),
    "rf": ("rf", "随机森林"),
    "et": ("extratrees", "极端随机树"),
    "gbm": ("gbm", "梯度提升"),
    "ridge": ("ridge", "岭回归"),
    "lasso": ("lasso",),
    "elasticnet": ("elasticnet", "弹性网络"),
    "knn": ("knn",),
    "auto": ("auto", "自动选择"),
}
_TERMINAL_NIR_STAGES = frozenset({"completed", "blocked"})
_SCRIPT_SUFFIX_RE = re.compile(r"\.(?:py|ipynb|r|jl|m)(?:$|[?#])", re.IGNORECASE)
_RAW_NIR_DATA_SUFFIX_RE = re.compile(r"\.(?:csv|tsv|txt|mat|npz|npy|xlsx?|xls)(?:$|[?#])", re.IGNORECASE)
# Interpreters that execute analysis scripts (Python, pip, pytest, R, Julia,
# MATLAB). Matched only at a command-start position (after ^, a shell
# separator ;&|( or newline, or a known prefix sudo/time/exec/env/nohup) to
# avoid false positives like ``grep python file`` or ``echo R``.
# ``python\d*(?:\.\d+)*`` covers ``python``, ``python3`` and ``python3.10``.
_INTERPRETER_COMMAND_RE = re.compile(
    r"(?:^|[;&|(\n])\s*(?:(?:sudo|time|exec|env|nohup)\s+)?"
    r"(?:python\d*(?:\.\d+)*(?:\.exe)?|pip\d*|pytest|uv\s+run\s+python"
    r"|Rscript|R\b|julia|matlab)"
    r"(?:\s|$)",
    re.IGNORECASE,
)
# Direct script execution: a file with a script extension at a command-start
# position (e.g. ``./script.py``, ``/path/to/run.R``). Avoids false positives
# like ``cat script.py`` or ``grep pattern file.py`` where the script is merely
# an argument to a non-executing command. Interpreters (python/Rscript/julia/
# matlab) are covered by _INTERPRETER_COMMAND_RE above.
_SCRIPT_EXEC_RE = re.compile(
    r"(?:^|[;&|(\n])\s*(?:(?:sudo|time|exec|env|nohup)\s+)?"
    r"\S*\.(?:py|ipynb|r|jl|m)(?:\s|$)",
    re.IGNORECASE,
)
_ANALYSIS_CODE_RE = re.compile(r"\b(?:import|from)\s+(?:numpy|pandas|scipy|sklearn|nir_core)\b", re.IGNORECASE)


def _knowledge_evidence_ids(payload: Mapping[str, Any]) -> list[str]:
    results = payload.get("results")
    if not isinstance(results, list):
        return []
    evidence: list[str] = []
    for result in results:
        identifier: Any = result if isinstance(result, str) else None
        if isinstance(result, Mapping):
            for key in (
                "evidence_id",
                "chunk_id",
                "id",
                "doc_id",
                "document_id",
                "source",
                "path",
                "title",
            ):
                if result.get(key):
                    identifier = result[key]
                    break
        if identifier is not None:
            normalized = str(identifier).strip()
            if normalized and normalized not in evidence:
                evidence.append(normalized)
        if len(evidence) == 10:
            break
    return evidence


def _state_from_request(request: ToolCallRequest) -> Mapping[str, Any]:
    runtime = request.runtime
    if runtime is not None and isinstance(runtime.state, Mapping):
        return runtime.state
    return request.state if isinstance(request.state, Mapping) else {}


def _denied_message(
    request: ToolCallRequest,
    *,
    code: str,
    error: str,
    workflow: Mapping[str, Any] | None,
    policy: _ToolPolicy | None = None,
    details: Mapping[str, Any] | None = None,
) -> ToolMessage:
    tool_name = str(request.tool_call.get("name", "unknown_tool"))
    payload = {
        "status": "error",
        "code": code,
        "error": error,
        "tool": tool_name,
        "stage": workflow.get("stage") if workflow else None,
        "task_type": workflow.get("task_type") if workflow else None,
        "allowed_stages": sorted(policy.stages) if policy else [],
        "next_action": workflow.get("next_action") if workflow else "start_workflow",
        "missing_inputs": list(workflow.get("missing_inputs") or []) if workflow else [],
        "clarification_questions": list(workflow.get("clarification_questions") or []) if workflow else [],
    }
    if details:
        payload["details"] = dict(details)
    return ToolMessage(
        content=json.dumps(payload, ensure_ascii=False),
        tool_call_id=str(request.tool_call.get("id", "missing_id")),
        name=tool_name,
        status="error",
    )


def _requested_model_method(request: ToolCallRequest) -> str | None:
    tool_name = str(request.tool_call.get("name", ""))
    if tool_name not in _MODELING_TOOLS:
        return None
    args = request.tool_call.get("args")
    args = args if isinstance(args, Mapping) else {}
    return normalize_retry_model_method(tool_name, args.get("method"))


def _request_execution_signature(
    request: ToolCallRequest,
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    args = request.tool_call.get("args")
    args = args if isinstance(args, Mapping) else {}
    pipeline_steps = args.get("pipeline_steps")
    if pipeline_steps is None:
        pipeline_steps = ["tool_default"]
    return retry_execution_signature(
        tool_name=str(request.tool_call.get("name") or ""),
        method=_requested_model_method(request),
        pipeline_steps=pipeline_steps,
        model_args=args,
    )


def _retry_plan_mismatch_denial(
    request: ToolCallRequest,
    workflow: Mapping[str, Any],
    policy: _ToolPolicy,
    *,
    error: str,
    details: Mapping[str, Any] | None = None,
) -> ToolMessage:
    observations = workflow.get("tool_observations")
    latest = observations[-1] if isinstance(observations, list) and observations else None
    repeated = bool(isinstance(latest, Mapping) and latest.get("name") == request.tool_call.get("name") and latest.get("code") == _RETRY_PLAN_MISMATCH_CODE)
    denial_details = dict(details or {})
    if repeated:
        denial_details.update(
            {
                "action_required": "stop_retrying_and_report_runtime_blocker",
                "mismatch_count": 2,
            }
        )
    return _denied_message(
        request,
        code=_RETRY_PLAN_MISMATCH_EXHAUSTED_CODE if repeated else _RETRY_PLAN_MISMATCH_CODE,
        error=error,
        workflow={**workflow, "stage": "blocked", "next_action": "report_best_effort"} if repeated else workflow,
        policy=policy,
        details=denial_details,
    )


def _deny_retry_plan_mismatch(
    request: ToolCallRequest,
    workflow: Mapping[str, Any],
    policy: _ToolPolicy,
) -> ToolMessage | None:
    if str(request.tool_call.get("name") or "") not in _MODELING_TOOLS or int(workflow.get("attempt") or 0) < 1:
        return None
    plan = workflow.get("retry_plan")
    if not isinstance(plan, Mapping) or plan.get("status") != "ready":
        return _retry_plan_mismatch_denial(
            request,
            workflow=workflow,
            policy=policy,
            error="A ready retry plan bound to the latest failed attempt is required.",
            details={"action_required": "record_retry_plan_then_plan_ready"},
        )
    try:
        _steps, _model_args, signature = _request_execution_signature(request)
        _planned_steps, _planned_model_args, canonical_plan_signature = retry_execution_signature(
            tool_name=str(plan.get("tool_name") or ""),
            method=str(plan.get("method") or "auto"),
            pipeline_steps=plan.get("pipeline_steps"),
            model_args=plan.get("model_args"),
        )
    except NIRWorkflowError as exc:
        return _retry_plan_mismatch_denial(
            request,
            workflow=workflow,
            policy=policy,
            error=str(exc),
        )
    accepted_signatures = {plan.get("execution_signature"), canonical_plan_signature}
    if plan.get("source_attempt") == workflow.get("attempt") and plan.get("tool_name") == request.tool_call.get("name") and signature in accepted_signatures:
        return None
    return _retry_plan_mismatch_denial(
        request,
        workflow=workflow,
        policy=policy,
        error="The modeling call does not match the active retry plan.",
        details={
            "plan_id": plan.get("plan_id"),
            "planned_tool": plan.get("tool_name"),
            "planned_signature": canonical_plan_signature,
            "recorded_plan_signature": plan.get("execution_signature"),
            "actual_signature": signature,
            "action_required": "execute_exactly_the_recorded_retry_plan",
        },
    )


def _pending_failed_model(workflow: Mapping[str, Any]) -> str | None:
    observations = workflow.get("tool_observations")
    if not isinstance(observations, list):
        return None
    for observation in reversed(observations):
        if not isinstance(observation, Mapping) or observation.get("name") not in _MODELING_TOOLS:
            continue
        if observation.get("code") == _MODEL_SUBSTITUTION_CODE:
            continue
        method = str(observation.get("requested_method") or "").strip().lower()
        if not method or method == "auto":
            continue
        return method if observation.get("status") == "error" else None
    return None


def _latest_user_text(request: ToolCallRequest) -> str:
    messages = _state_from_request(request).get("messages")
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return get_original_user_content_text(message.content, message.additional_kwargs)
        if isinstance(message, Mapping) and str(message.get("type") or message.get("role") or "").lower() in {"human", "user"}:
            kwargs = message.get("additional_kwargs")
            return get_original_user_content_text(
                message.get("content"),
                kwargs if isinstance(kwargs, Mapping) else None,
            )
    return ""


def _latest_user_approved_substitution(request: ToolCallRequest, attempted_method: str) -> bool:
    text = _latest_user_text(request).lower()
    compact = re.sub(r"[\s_-]+", "", text)
    aliases = _MODEL_ALIASES.get(attempted_method, (attempted_method,))
    compact_aliases = tuple(re.sub(r"[\s_-]+", "", alias.lower()) for alias in aliases)
    target_mentioned = any(alias in compact for alias in compact_aliases)
    denial_prefix = r"(?:不要|不能|不准|禁止|别|拒绝|不想|donot|dont|never|decline|reject)"
    if any(re.search(rf"{denial_prefix}.{{0,16}}{re.escape(alias)}", compact) for alias in compact_aliases):
        return False
    approval_action = bool(
        re.search(
            r"(?:改用|换成|替换为|采用|使用|用|同意|可以)|"
            r"\b(?:use|switch(?:\s+to)?|replace(?:\s+with)?|fall\s*back\s+to|approve)\b",
            text,
            re.IGNORECASE,
        )
    )
    return target_mentioned and approval_action


def _deny_unapproved_model_substitution(
    request: ToolCallRequest,
    workflow: Mapping[str, Any],
    policy: _ToolPolicy,
) -> ToolMessage | None:
    failed_method = _pending_failed_model(workflow)
    attempted_method = _requested_model_method(request)
    if not failed_method or not attempted_method or attempted_method == failed_method:
        return None
    if _latest_user_approved_substitution(request, attempted_method):
        return None
    return _denied_message(
        request,
        code=_MODEL_SUBSTITUTION_CODE,
        error=(f"The explicitly requested model {failed_method!r} failed. Do not replace it with {attempted_method!r} until the user explicitly approves that model."),
        workflow=workflow,
        policy=policy,
        details={
            "failed_method": failed_method,
            "attempted_method": attempted_method,
            "action_required": "request_user_approval_for_model_substitution",
        },
    )


def _deny_validation_goal_conflict(
    request: ToolCallRequest,
    workflow: Mapping[str, Any],
    policy: _ToolPolicy,
) -> ToolMessage | None:
    tool_name = str(request.tool_call.get("name", ""))
    validation_goal = str(workflow.get("validation_goal") or "").strip().lower()
    if tool_name not in {*_MODELING_TOOLS, "nir_register_model"}:
        return None
    if not validation_goal:
        return None
    if validation_goal == "exploratory":
        return _denied_message(
            request,
            code=_VALIDATION_GOAL_CONFLICT_CODE,
            error=(f"Tool {tool_name!r} creates or registers a validation model, but validation_goal='exploratory' does not permit an independent holdout, external-validation, or deployable-model claim."),
            workflow=workflow,
            policy=policy,
            details={
                "validation_goal": validation_goal,
                "action_required": "report_exploratory_findings_or_request_validation_goal_change",
                "allowed_validation_goals_for_modeling": [
                    "internal_holdout",
                    "external_validation",
                    "production",
                ],
            },
        )
    if tool_name == "nir_register_model":
        return None
    if validation_goal == "internal_holdout" and tool_name in _INTERNAL_HOLDOUT_TOOLS:
        return None
    if validation_goal in {"external_validation", "production"} and tool_name in _EXTERNAL_VALIDATION_TOOLS:
        return None
    required_scope = "independent_external_validation" if validation_goal in {"external_validation", "production"} else "independent_holdout_not_external"
    allowed_tools = sorted(_EXTERNAL_VALIDATION_TOOLS) if validation_goal in {"external_validation", "production"} else sorted(_INTERNAL_HOLDOUT_TOOLS)
    return _denied_message(
        request,
        code=_VALIDATION_GOAL_CONFLICT_CODE,
        error=(f"Tool {tool_name!r} does not implement the validation protocol required by validation_goal={validation_goal!r}."),
        workflow=workflow,
        policy=policy,
        details={
            "validation_goal": validation_goal,
            "required_validation_scope": required_scope,
            "allowed_modeling_tools": allowed_tools,
            "action_required": "use_goal_compatible_validation_protocol_or_change_validation_goal",
        },
    )


def _deny_grouping_split_column_conflict(
    request: ToolCallRequest,
    workflow: Mapping[str, Any],
    policy: _ToolPolicy,
) -> ToolMessage | None:
    if str(request.tool_call.get("name", "")) != "nir_train_partitioned_model":
        return None
    args = request.tool_call.get("args")
    args = args if isinstance(args, Mapping) else {}
    split_column = str(args.get("split_col") or "").strip()
    grouping_column = str(workflow.get("grouping_column") or "").strip()
    if not split_column or split_column.casefold() != grouping_column.casefold():
        return None
    return _denied_message(
        request,
        code=_GROUPING_SPLIT_CONFLICT_CODE,
        error=("The named partition column and the sample-grouping column have different scientific roles and cannot be the same field."),
        workflow=workflow,
        policy=policy,
        details={
            "split_column": split_column,
            "grouping_column": grouping_column,
            "action_required": "record_distinct_sample_grouping_column",
        },
    )


def _deny_registration_evidence_mismatch(
    request: ToolCallRequest,
    workflow: Mapping[str, Any],
    policy: _ToolPolicy,
) -> ToolMessage | None:
    if str(request.tool_call.get("name", "")) != "nir_register_model":
        return None
    args = request.tool_call.get("args")
    args = args if isinstance(args, Mapping) else {}
    evidence = workflow.get("attempt_evidence")
    if not isinstance(evidence, Mapping):
        return _denied_message(
            request,
            code="nir_registration_evidence_mismatch",
            error="Registration requires evidence from the approved modeling attempt in this workflow.",
            workflow=workflow,
            policy=policy,
            details={"action_required": "rerun_modeling_and_approve_the_recorded_attempt"},
        )
    requested_model = str(args.get("model_path") or "").replace("\\", "/")
    requested_metrics = str(args.get("metrics_path") or "").replace("\\", "/")
    approved_model = str(evidence.get("model_path") or workflow.get("model_path") or "").replace("\\", "/")
    approved_metrics = str(evidence.get("metrics_path") or workflow.get("metrics_path") or "").replace("\\", "/")
    if requested_model == approved_model and requested_metrics == approved_metrics and requested_model and requested_metrics:
        return None
    return _denied_message(
        request,
        code="nir_registration_evidence_mismatch",
        error="The requested model and metrics are not the artifacts from the approved modeling attempt.",
        workflow=workflow,
        policy=policy,
        details={
            "requested_model_path": requested_model,
            "requested_metrics_path": requested_metrics,
            "approved_model_path": approved_model,
            "approved_metrics_path": approved_metrics,
            "action_required": "register_exactly_the_approved_attempt_artifacts",
        },
    )


def _authorize(request: ToolCallRequest) -> ToolMessage | None:
    tool_name = str(request.tool_call.get("name", ""))
    workflow = _state_from_request(request).get("nir_workflow")
    args = request.tool_call.get("args")
    args = args if isinstance(args, Mapping) else {}
    if isinstance(workflow, Mapping) and tool_name in {"read_file", "read_file_tool"}:
        requested_path = str(args.get("path") or args.get("file_path") or "").replace("\\", "/")
        workflow_path = str(workflow.get("data_path") or "").replace("\\", "/")
        is_workflow_input = bool(requested_path and workflow_path and requested_path == workflow_path)
        is_uploaded_raw_data = "/uploads/" in requested_path.lower() and bool(_RAW_NIR_DATA_SUFFIX_RE.search(requested_path))
        if is_workflow_input or is_uploaded_raw_data:
            return _denied_message(
                request,
                code="nir_raw_data_read_forbidden",
                error=("Raw NIR input data must not be copied into the model context. Use nir_inspect or another structured NIR tool and rely on its bounded summary."),
                workflow=workflow,
                details={
                    "path": requested_path,
                    "action_required": "use_nir_inspect_summary",
                },
            )
    if isinstance(workflow, Mapping) and str(workflow.get("stage", "")) not in _TERMINAL_NIR_STAGES:
        violation = False
        if tool_name in {"write_file", "write_file_tool"}:
            path = str(args.get("path") or args.get("file_path") or "")
            content = str(args.get("content") or "")
            # Only block files that are BOTH a script extension AND contain
            # analysis-style imports. This permits legitimate Python module
            # files (empty __init__.py, config.py without analysis imports)
            # while still blocking analysis scripts as defense-in-depth.
            # Active script execution is separately blocked in the bash branch.
            violation = bool(_SCRIPT_SUFFIX_RE.search(path) and _ANALYSIS_CODE_RE.search(content))
        elif tool_name in {"bash", "bash_tool"}:
            command = str(args.get("command") or args.get("cmd") or "")
            violation = bool(_INTERPRETER_COMMAND_RE.search(command) or _SCRIPT_EXEC_RE.search(command))
        if violation:
            return _denied_message(
                request,
                code="nir_code_execution_forbidden",
                error=("Python/R/MATLAB analysis scripts are forbidden during an active NIR workflow. Use nir_inspect, nir_load_data, nir_preprocess, and NIR modeling tools only."),
                workflow=workflow,
            )

    policy = _TOOL_POLICIES.get(tool_name)
    if policy is None:
        return None

    if not isinstance(workflow, Mapping):
        return _denied_message(
            request,
            code="nir_workflow_required",
            error="Start a NIR workflow with nir_workflow(action='start', ...) before calling NIR domain tools.",
            workflow=None,
            policy=policy,
        )

    if validation_denial := _deny_validation_goal_conflict(request, workflow, policy):
        return validation_denial
    if grouping_denial := _deny_grouping_split_column_conflict(
        request,
        workflow,
        policy,
    ):
        return grouping_denial
    stage = str(workflow.get("stage", ""))
    task_type = str(workflow.get("task_type", ""))
    if stage not in policy.stages:
        return _denied_message(
            request,
            code="nir_workflow_stage_denied",
            error=f"Tool {tool_name!r} is not allowed during NIR workflow stage {stage!r}.",
            workflow=workflow,
            policy=policy,
        )
    if task_type not in policy.task_types:
        return _denied_message(
            request,
            code="nir_workflow_task_denied",
            error=f"Tool {tool_name!r} is not compatible with NIR task type {task_type!r}.",
            workflow=workflow,
            policy=policy,
        )
    if retry_denial := _deny_retry_plan_mismatch(request, workflow, policy):
        return retry_denial
    if registration_denial := _deny_registration_evidence_mismatch(request, workflow, policy):
        return registration_denial
    if substitution_denial := _deny_unapproved_model_substitution(request, workflow, policy):
        return substitution_denial
    return None


def _tool_message(result: ToolMessage | Command) -> ToolMessage | None:
    if isinstance(result, ToolMessage):
        return result
    if not isinstance(result.update, Mapping):
        return None
    messages = result.update.get("messages")
    if not isinstance(messages, list):
        return None
    return next((message for message in reversed(messages) if isinstance(message, ToolMessage)), None)


def _success_payload(result: ToolMessage | Command) -> dict[str, Any] | None:
    message = _tool_message(result)
    if message is None or message.status == "error" or not isinstance(message.content, str):
        return None
    try:
        payload = json.loads(message.content)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("status") == "error":
        return None
    return payload


def _bind_reflection_request(request: ToolCallRequest) -> ToolCallRequest:
    """Bind reflection inputs to durable attempts instead of model-authored history."""

    if str(request.tool_call.get("name") or "") != "nir_reflect":
        return request
    workflow = _state_from_request(request).get("nir_workflow")
    if not isinstance(workflow, Mapping):
        return request
    attempts = workflow.get("attempts")
    attempts = attempts if isinstance(attempts, list) else []
    prior_attempts = attempts[:-1] if attempts else []
    history = [
        {
            "pipeline": attempt.get("pipeline_steps") or [],
            "metrics": attempt.get("metrics_summary") or {},
        }
        for attempt in prior_attempts
        if isinstance(attempt, Mapping)
    ]
    args = request.tool_call.get("args")
    args = dict(args) if isinstance(args, Mapping) else {}
    args.update(
        {
            "metrics_path": workflow.get("metrics_path"),
            "history": json.dumps(history, ensure_ascii=False),
            "domain": workflow.get("domain") or "default",
            "attempt": int(workflow.get("attempt") or 0),
            "max_retries": int(workflow.get("max_attempts") or 1),
        }
    )
    return replace(
        request,
        tool_call={
            **request.tool_call,
            "args": args,
        },
    )


def _model_attempt_update(
    request: ToolCallRequest,
    workflow: NIRWorkflowState,
    payload: Mapping[str, Any],
) -> NIRWorkflowState | None:
    tool_name = str(request.tool_call.get("name", ""))
    result = payload.get("best") if tool_name == "nir_compare" else payload
    if not isinstance(result, Mapping) or not isinstance(result.get("passed"), bool):
        return None

    model_path = payload.get("model_path") or payload.get("model")
    metrics_path = payload.get("metrics_path") or payload.get("metrics")
    if tool_name == "nir_compare":
        metrics_path = payload.get("all_metrics")
    protocol, validation_scope = _TOOL_VALIDATION_PROTOCOL[tool_name]
    emitted_evidence = payload.get("evidence")
    emitted_evidence = emitted_evidence if isinstance(emitted_evidence, Mapping) else {}
    digest_fields = {}
    for key in ("model_sha256", "metrics_sha256", "training_data_sha256"):
        value = str(emitted_evidence.get(key) or "").lower()
        if len(value) == 64 and all(character in "0123456789abcdef" for character in value):
            digest_fields[key] = value
    metric_keys = (
        "R2_val",
        "RPD",
        "RMSEC",
        "RMSECV",
        "RMSEP",
        "holdout",
        "external",
        "grade",
        "passed",
        "thresholds_used",
    )
    metrics_summary = {key: result.get(key) for key in metric_keys if result.get(key) is not None}
    if isinstance(result.get("overall"), Mapping):
        metrics_summary["overall"] = {key: result["overall"].get(key) for key in ("passed", "n_passed", "n_targets") if result["overall"].get(key) is not None}
    if isinstance(result.get("per_component"), list):
        metrics_summary["per_component"] = [
            {key: component.get(key) for key in ("name", "R2_val", "RPD", "RMSEP", "grade", "passed") if component.get(key) is not None} for component in result["per_component"][:20] if isinstance(component, Mapping)
        ]
        metrics_summary["component_count"] = len(result["per_component"])
    result_fact_keys = (
        "preprocessing",
        "wavelength_selection",
        "wavelength_selection_decision",
        "model_selection_decision",
        "candidate_results",
        "model_candidates",
        "method",
    )
    result_facts = {key: result.get(key) for key in result_fact_keys if result.get(key) is not None}
    try:
        pipeline_steps, model_args, execution_signature = _request_execution_signature(request)
    except NIRWorkflowError:
        logger.exception("Could not canonicalize NIR modeling execution")
        return None
    attempt_evidence = {
        "schema_version": 1,
        "tool_name": tool_name,
        "tool_call_id": str(request.tool_call.get("id")) if request.tool_call.get("id") else None,
        "run_id": _runtime_context_value(request, "run_id"),
        "trace_id": _runtime_context_value(request, DEERFLOW_TRACE_METADATA_KEY),
        "protocol": protocol,
        "validation_scope": validation_scope,
        "model_path": str(model_path) if model_path else None,
        "metrics_path": str(metrics_path) if metrics_path else None,
        "method": _requested_model_method(request),
        "pipeline_steps": pipeline_steps,
        "model_args": model_args,
        "execution_signature": execution_signature,
        **digest_fields,
        "metrics_summary": metrics_summary,
        **({"result_facts": result_facts} if result_facts else {}),
    }
    return transition_workflow(
        workflow,
        action="record_attempt",
        attempt_passed=result["passed"],
        grade=str(result.get("grade")) if result.get("grade") is not None else None,
        model_path=str(model_path) if model_path else None,
        metrics_path=str(metrics_path) if metrics_path else None,
        attempt_evidence=attempt_evidence,
        notes=f"Automatically recorded successful {tool_name} result.",
    )


def _data_audit_update(
    request: ToolCallRequest,
    workflow: NIRWorkflowState,
    payload: Mapping[str, Any],
) -> NIRWorkflowState:
    """Persist bounded structural facts needed by the final response."""

    tool_name = str(request.tool_call.get("name") or "")
    args = request.tool_call.get("args")
    args = args if isinstance(args, Mapping) else {}
    payload_fields = (
        "n_samples",
        "n_wavelengths",
        "raw_wavelength_range",
        "usable_wavelength_range",
        "constant_wavelength_count",
        "usable_wavelength_count",
        "wavelength_range_semantics",
        "y_names",
        "y_range",
        "y_ranges",
        "y_separated",
        "wv_separated",
    )
    evidence = {
        "source_tool": tool_name,
        "data_path": str(args.get("file_path") or args.get("data_path") or workflow.get("data_path") or ""),
        **{key: payload[key] for key in payload_fields if payload.get(key) is not None},
    }
    if "raw_wavelength_range" not in evidence and payload.get("wavelength_range") is not None:
        evidence["raw_wavelength_range"] = payload["wavelength_range"]
    for key in ("y_col", "y_cols", "x_cols", "wv_row", "x_var", "y_var", "wv_var", "subset", "transpose"):
        if args.get(key) is not None:
            evidence[key] = args[key]
    return transition_workflow(
        workflow,
        action="record_audit_evidence",
        audit_evidence=evidence,
        notes=f"Automatically recorded bounded {tool_name} evidence.",
    )


def _next_workflow(
    request: ToolCallRequest,
    workflow: NIRWorkflowState,
    payload: Mapping[str, Any],
) -> NIRWorkflowState | None:
    tool_name = str(request.tool_call.get("name", ""))
    if tool_name in {"nir_inspect", "nir_load_data"}:
        return _data_audit_update(request, workflow, payload)
    if tool_name in _MODELING_TOOLS:
        return _model_attempt_update(request, workflow, payload)
    if tool_name == "nir_reflect":
        return transition_workflow(
            workflow,
            action="record_reflection",
            reflection_result=dict(payload),
            notes="Reflection inputs were bound to the latest attempt evidence.",
        )
    if tool_name == "nir_search_knowledge":
        return transition_workflow(
            workflow,
            action="knowledge_retrieved",
            evidence_ids=_knowledge_evidence_ids(payload),
            notes="Knowledge retrieval completed successfully.",
        )
    if tool_name == "nir_register_model" and payload.get("status") == "registered":
        return transition_workflow(
            workflow,
            action="registered",
            notes=f"Registered model {payload.get('model_id', '')}".strip(),
        )
    return None


def _attach_workflow_update(result: ToolMessage | Command, workflow: NIRWorkflowState) -> ToolMessage | Command:
    if isinstance(result, ToolMessage):
        return Command(update={"nir_workflow": workflow, "messages": [result]})
    if isinstance(result.update, Mapping):
        return replace(result, update={**result.update, "nir_workflow": workflow})
    logger.warning("Could not persist automatic NIR workflow update: Command.update is not a mapping")
    return result


def _runtime_context_value(request: ToolCallRequest, key: str) -> str | None:
    runtime = request.runtime
    context = runtime.context if runtime is not None else None
    if not isinstance(context, Mapping):
        return None
    value = context.get(key)
    return str(value) if value else None


def _record_observation(request: ToolCallRequest, result: ToolMessage | Command) -> ToolMessage | Command:
    tool_name = str(request.tool_call.get("name", ""))
    if tool_name not in _OBSERVED_NIR_TOOLS:
        return result

    current = _state_from_request(request).get("nir_workflow")
    if not isinstance(current, dict):
        return result

    workflow = current
    if isinstance(result, Command) and isinstance(result.update, Mapping):
        candidate = result.update.get("nir_workflow")
        if isinstance(candidate, dict):
            workflow = candidate
    message = _tool_message(result)
    payload: dict[str, Any] = {}
    if message is not None and isinstance(message.content, str):
        try:
            decoded = json.loads(message.content)
            if isinstance(decoded, dict):
                payload = decoded
        except (TypeError, ValueError):
            pass
    status = "error" if message is not None and (message.status == "error" or payload.get("status") == "error") else "success"
    if payload.get("code") == _RETRY_PLAN_MISMATCH_EXHAUSTED_CODE:
        details = payload.get("details")
        plan_id = details.get("plan_id") if isinstance(details, Mapping) else None
        workflow = block_workflow_after_retry_plan_mismatch(
            workflow,
            plan_id=str(plan_id) if plan_id else None,
        )
    observed = record_tool_observation(
        workflow,
        name=tool_name,
        status=status,
        stage_before=str(current.get("stage")) if current.get("stage") else None,
        stage_after=str(workflow.get("stage")) if workflow.get("stage") else None,
        code=str(payload["code"]) if payload.get("code") else None,
        requested_method=_requested_model_method(request),
        call_id=str(request.tool_call.get("id")) if request.tool_call.get("id") else None,
        run_id=_runtime_context_value(request, "run_id"),
        trace_id=_runtime_context_value(request, DEERFLOW_TRACE_METADATA_KEY),
    )
    return _attach_workflow_update(result, observed)


def _advance(request: ToolCallRequest, result: ToolMessage | Command) -> ToolMessage | Command:
    payload = _success_payload(result)
    if payload is None:
        return result
    workflow = _state_from_request(request).get("nir_workflow")
    if not isinstance(workflow, dict):
        return result
    try:
        updated = _next_workflow(request, workflow, payload)
    except NIRWorkflowError:
        logger.exception("Automatic NIR workflow transition failed")
        return result
    return _attach_workflow_update(result, updated) if updated is not None else result


def _workflow_completion_reminder(
    required_action: str,
    workflow: Mapping[str, Any],
) -> str:
    instructions = {
        "inspect_data": ("Call nir_inspect or the required structured NIR audit tool before forming a conclusion."),
        "prepare_analysis_plan": ("Record the analysis plan with nir_workflow(action='plan_ready') and execute it before reporting results."),
        "execute_nir_tools": ("Execute the authorized NIR tool for the current plan before reporting results."),
        "reflect_on_attempt": ("Call nir_reflect now. Its inputs will be bound to the latest failed attempt by the runtime. Then follow the persisted reflection decision; do not report a final result yet."),
        "retrieve_evidence_for_retry": ("Retrieve the evidence requested by the latest reflection, record it, and continue to retry planning."),
        "prepare_retry_plan": ("Call nir_workflow(action='record_retry_plan', ...) with a materially changed execution, then call plan_ready and execute that exact plan."),
    }
    instruction = instructions.get(
        required_action,
        f"Complete the required workflow action {required_action!r}.",
    )
    return (
        "Your previous response attempted to end an active NIR workflow before "
        "its mandatory next action was completed. This control message is "
        "hidden from the user. "
        f"Current stage={workflow.get('stage')!r}, "
        f"required next_action={required_action!r}. {instruction}"
    )


class NIRWorkflowMiddleware(AgentMiddleware[AgentState]):
    """Enforce NIR workflow policy and persist deterministic tool outcomes."""

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        denied = _authorize(request)
        if denied is not None:
            return _record_observation(request, denied)
        bound_request = _bind_reflection_request(request)
        return _record_observation(bound_request, _advance(bound_request, handler(bound_request)))

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        denied = _authorize(request)
        if denied is not None:
            return _record_observation(request, denied)
        bound_request = _bind_reflection_request(request)
        return _record_observation(
            bound_request,
            _advance(bound_request, await handler(bound_request)),
        )

    def _guard_final_response(self, state: AgentState) -> dict | None:
        messages = state.get("messages") or []
        if not messages or not isinstance(messages[-1], AIMessage):
            return None
        message = messages[-1]
        if message.tool_calls:
            return None
        workflow = state.get("nir_workflow")
        if not isinstance(workflow, dict):
            return None
        required_action = required_nir_workflow_action(workflow)
        if required_action:
            violation = f"workflow_incomplete:{required_action}"
            guarded_workflow = record_response_guard(
                workflow,
                violations=[violation],
                message_id=str(message.id) if message.id else None,
            )
            reminder_count = sum(violation in event.get("violations", []) for event in guarded_workflow.get("response_guard_events", []) if isinstance(event, Mapping))
            if reminder_count <= _MAX_WORKFLOW_CONTINUATION_REMINDERS:
                reminder = HumanMessage(
                    name="nir_workflow_completion_reminder",
                    content=_workflow_completion_reminder(
                        required_action,
                        workflow,
                    ),
                    additional_kwargs={
                        "hide_from_ui": True,
                        "nir_required_action": required_action,
                    },
                )
                logger.warning(
                    "Re-engaging NIR agent after premature final response",
                    extra={
                        "project_id": workflow.get("project_id"),
                        "stage": workflow.get("stage"),
                        "required_action": required_action,
                        "reminder_count": reminder_count,
                    },
                )
                return {
                    "messages": [reminder],
                    "nir_workflow": guarded_workflow,
                    "jump_to": "model",
                }

            blocked_workflow = block_workflow_after_continuation_failure(
                guarded_workflow,
                required_action=required_action,
            )
            additional_kwargs = dict(message.additional_kwargs or {})
            additional_kwargs["nir_response_grounding"] = {
                "passed": False,
                "violations": [violation],
                "replacement": "blocked_best_effort_evidence_summary",
            }
            replacement = message.model_copy(
                update={
                    "content": render_grounded_nir_response(
                        blocked_workflow,
                        violations=(violation,),
                    ),
                    "additional_kwargs": additional_kwargs,
                }
            )
            return {
                "messages": [replacement],
                "nir_workflow": blocked_workflow,
            }
        response_text = message_content_to_text(message.content)
        verdict = validate_nir_response(response_text, workflow)
        if verdict.passed:
            return None

        grounding_metadata = {
            "passed": False,
            "violations": list(verdict.violations),
            "replacement": "current_attempt_evidence_summary",
        }
        additional_kwargs = dict(message.additional_kwargs or {})
        additional_kwargs["nir_response_grounding"] = grounding_metadata
        replacement = message.model_copy(
            update={
                "content": render_grounded_nir_response(
                    workflow,
                    violations=verdict.violations,
                ),
                "additional_kwargs": additional_kwargs,
            }
        )
        guarded_workflow = record_response_guard(
            workflow,
            violations=list(verdict.violations),
            message_id=str(message.id) if message.id else None,
        )
        logger.warning(
            "Replaced unsupported NIR final response claims",
            extra={
                "project_id": workflow.get("project_id"),
                "stage": workflow.get("stage"),
                "violations": list(verdict.violations),
            },
        )
        return {
            "messages": [replacement],
            "nir_workflow": guarded_workflow,
        }

    @hook_config(can_jump_to=["model"])
    @override
    def after_model(self, state: AgentState, runtime: Runtime) -> dict | None:  # noqa: ARG002
        return self._guard_final_response(state)

    @hook_config(can_jump_to=["model"])
    @override
    async def aafter_model(self, state: AgentState, runtime: Runtime) -> dict | None:  # noqa: ARG002
        return self._guard_final_response(state)
