"""Runtime enforcement and automatic advancement for NIR workflows."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware, hook_config
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from deerflow.agents.thread_state import NIRWorkflowState
from deerflow.community.nir.decision_facts import model_result_facts
from deerflow.community.nir.model_autosave import failed_model_save, save_registered_model
from deerflow.community.nir.process import bind_execution, finish_execution, reset_execution, start_execution
from deerflow.community.nir.response_grounding import (
    render_grounded_nir_response,
    required_nir_workflow_action,
    validate_nir_response,
)
from deerflow.community.nir.user_wait import begin_user_wait, question_for_user, resume_user_wait, waiting_for_user, waiting_message
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


_MODEL_TASKS = frozenset({"analysis", "calibration", "classification", "multi_modeling", "compare"})
_DATA_TASKS = frozenset({"analysis", "calibration", "classification", "multi_modeling", "compare", "prediction", "inspection"})
_EXECUTION_TASKS = frozenset({"analysis", "calibration", "classification", "multi_modeling", "compare", "prediction"})

_CHEMOTOOLS_READ_TOOLS = frozenset(
    {
        "chemotools_health",
        "chemotools_list_capabilities",
        "chemotools_describe_capability",
        "chemotools_validate_operation",
        "chemotools_get_artifact_metadata",
    }
)
_CHEMOTOOLS_EXECUTION_TOOLS = frozenset(
    {
        "chemotools_fit_estimator",
        "chemotools_apply_estimator",
        "chemotools_call_function",
        "chemotools_render_plot",
        "chemotools_run_inspector",
    }
)
_DATASET_LIBRARY_TOOLS = frozenset(
    {
        "nir_dataset_list",
        "nir_dataset_get",
        "nir_dataset_history",
        "nir_dataset_save",
        "nir_dataset_attach",
    }
)
_MODEL_REUSE_TOOLS = frozenset({"nir_model_list", "nir_model_get", "nir_model_attach"})
_DATASET_READ_STAGES = frozenset({"clarification", "data_audit", "planning", "execution", "evaluation", "knowledge", "review", "approved", "registered", "completed", "blocked"})
_ALL_NIR_TASKS = frozenset({*_DATA_TASKS, "knowledge"})

_TOOL_POLICIES: dict[str, _ToolPolicy] = {
    "nir_decision_context": _ToolPolicy(frozenset({"execution", "decision_review", "budget_exhausted"}), frozenset({"analysis", "calibration"})),
    "nir_submit_decision": _ToolPolicy(frozenset({"execution", "decision_review", "budget_exhausted", "awaiting_final_confirmation", "frozen", "review"}), frozenset({"analysis", "calibration"})),
    "nir_decision_history": _ToolPolicy(frozenset({"execution", "decision_review", "initial_result", "awaiting_final_confirmation", "budget_exhausted"}), frozenset({"analysis", "calibration"})),
    "nir_execute_decision": _ToolPolicy(frozenset({"execution", "initial_result"}), frozenset({"analysis", "calibration"})),
    "nir_prepare_search": _ToolPolicy(frozenset({"data_audit", "planning", "execution"}), frozenset({"analysis", "calibration"})),
    "nir_search_model": _ToolPolicy(frozenset({"execution"}), frozenset({"analysis", "calibration"})),
    "nir_freeze_search": _ToolPolicy(frozenset({"execution", "awaiting_final_confirmation", "frozen"}), frozenset({"analysis", "calibration"})),
    "nir_finalize_search": _ToolPolicy(frozenset({"frozen", "review"}), frozenset({"analysis", "calibration"})),
    "nir_load_data": _ToolPolicy(frozenset({"data_audit"}), _DATA_TASKS),
    "nir_inspect": _ToolPolicy(frozenset({"data_audit"}), _DATA_TASKS),
    "nir_preprocess": _ToolPolicy(frozenset({"execution"}), _EXECUTION_TASKS),
    "nir_align_wavelengths": _ToolPolicy(frozenset({"execution"}), _EXECUTION_TASKS),
    "nir_train_auto_split_model": _ToolPolicy(frozenset({"execution"}), frozenset({"analysis", "calibration"})),
    "nir_train_model": _ToolPolicy(frozenset({"execution"}), frozenset({"analysis", "calibration"})),
    "nir_train_classifier": _ToolPolicy(frozenset({"execution"}), frozenset({"classification"})),
    "nir_train_partitioned_model": _ToolPolicy(frozenset({"execution"}), frozenset({"analysis", "calibration"})),
    "nir_train_multi_model": _ToolPolicy(frozenset({"execution"}), frozenset({"multi_modeling"})),
    "nir_analyze": _ToolPolicy(frozenset({"execution"}), frozenset({"analysis", "calibration"})),
    "nir_analyze_collection": _ToolPolicy(frozenset({"execution"}), frozenset({"analysis", "calibration"})),
    "nir_compare": _ToolPolicy(frozenset({"execution"}), _MODEL_TASKS),
    "nir_predict": _ToolPolicy(frozenset({"execution"}), frozenset({"prediction"})),
    "nir_reflect": _ToolPolicy(frozenset({"evaluation"}), _MODEL_TASKS),
    "nir_search_knowledge": _ToolPolicy(frozenset({"execution", "knowledge"}), frozenset({"knowledge", *_MODEL_TASKS})),
    "nir_search_method_knowledge": _ToolPolicy(_DATASET_READ_STAGES, _ALL_NIR_TASKS),
    "nir_register_model": _ToolPolicy(frozenset({"approved"}), _MODEL_TASKS),
    "nir_dataset_list": _ToolPolicy(_DATASET_READ_STAGES, _ALL_NIR_TASKS),
    "nir_dataset_get": _ToolPolicy(_DATASET_READ_STAGES, _ALL_NIR_TASKS),
    "nir_dataset_history": _ToolPolicy(_DATASET_READ_STAGES, _ALL_NIR_TASKS),
    "nir_dataset_save": _ToolPolicy(frozenset({"clarification", "data_audit"}), _DATA_TASKS),
    "nir_dataset_attach": _ToolPolicy(frozenset({"clarification"}), _DATA_TASKS),
    "nir_model_list": _ToolPolicy(_DATASET_READ_STAGES, _ALL_NIR_TASKS),
    "nir_model_get": _ToolPolicy(_DATASET_READ_STAGES, _ALL_NIR_TASKS),
    "nir_model_attach": _ToolPolicy(frozenset({"clarification"}), frozenset({"prediction"})),
    "nir_model_promote": _ToolPolicy(frozenset({"registered", "completed"}), _MODEL_TASKS),
    **{
        name: _ToolPolicy(
            frozenset({"planning", "execution", "evaluation", "knowledge", "approved"}),
            frozenset({*_EXECUTION_TASKS, "knowledge"}),
        )
        for name in _CHEMOTOOLS_READ_TOOLS
    },
    **{
        name: _ToolPolicy(
            frozenset({"execution", "evaluation"}),
            _EXECUTION_TASKS,
        )
        for name in _CHEMOTOOLS_EXECUTION_TOOLS
    },
}
_OBSERVED_NIR_TOOLS = frozenset({*_TOOL_POLICIES, "nir_workflow"})

_MODELING_TOOLS = frozenset({"nir_train_auto_split_model", "nir_train_model", "nir_train_classifier", "nir_train_partitioned_model", "nir_train_multi_model", "nir_analyze", "nir_analyze_collection", "nir_compare"})
_PREPROCESSING_DECISION_TOOLS = frozenset({*_MODELING_TOOLS, "nir_preprocess", "nir_search_model"})
_CHEMOTOOLS_CATALOG_TOOL = "chemotools_list_capabilities"
_INTERNAL_HOLDOUT_TOOLS = _MODELING_TOOLS
_EXTERNAL_VALIDATION_TOOLS = frozenset({"nir_train_partitioned_model"})
_TOOL_VALIDATION_PROTOCOL = {
    "nir_train_auto_split_model": ("deterministic_auto_split_holdout", "independent_holdout_not_external"),
    "nir_train_model": ("random_three_way_holdout", "independent_holdout_not_external"),
    "nir_train_classifier": ("classification_three_way_holdout", "independent_holdout_not_external"),
    "nir_train_partitioned_model": ("named_partition_external_validation", "independent_external_validation"),
    "nir_train_multi_model": ("multi_target_random_three_way_holdout", "independent_holdout_not_external"),
    "nir_analyze": ("automated_analysis_three_way_holdout", "independent_holdout_not_external"),
    "nir_analyze_collection": ("mat_collection_sequential_compact", "independent_holdout_not_external"),
    "nir_compare": ("preprocessing_comparison_three_way_holdout", "independent_holdout_not_external"),
}
_PARTITIONED_PROTOCOLS = {
    ("named_partition_external_validation", "independent_external_validation"),
    ("named_partition_independent_holdout", "independent_holdout_not_external"),
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


def _deny_invalid_dataset_start_binding(request: ToolCallRequest) -> ToolMessage | None:
    if str(request.tool_call.get("name", "")) != "nir_workflow":
        return None
    args = request.tool_call.get("args")
    args = args if isinstance(args, Mapping) else {}
    if str(args.get("action", "")).strip().lower() != "start":
        return None
    lineage_keys = ("dataset_id", "dataset_profile_id", "dataset_sha256", "dataset_attachment_id")
    requested_lineage = any(args.get(key) is not None for key in lineage_keys)
    latest_attachment: Mapping[str, Any] | None = None
    messages = _state_from_request(request).get("messages")
    if isinstance(messages, list):
        for message in reversed(messages):
            if not isinstance(message, ToolMessage) or message.name != "nir_dataset_attach":
                continue
            try:
                payload = json.loads(message_content_to_text(message.content))
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(payload, Mapping) and payload.get("status") == "ok" and payload.get("attachment_status") == "attached":
                latest_attachment = payload
                break
    requested_path = str(args.get("data_path") or "")
    uses_latest_path = bool(latest_attachment and requested_path == str(latest_attachment.get("virtual_path") or ""))
    if not requested_lineage and not uses_latest_path:
        return None
    if latest_attachment is None:
        return _denied_message(
            request,
            code="nir_dataset_binding_invalid",
            error="Saved Dataset lineage must come from a successful nir_dataset_attach result in this thread.",
            workflow=None,
            details={"action_required": "attach_dataset_before_starting_workflow"},
        )
    expected = {
        "project_id": latest_attachment.get("workflow_project_id"),
        "data_path": latest_attachment.get("virtual_path"),
        "dataset_id": latest_attachment.get("dataset_id"),
        "dataset_profile_id": latest_attachment.get("profile_id"),
        "dataset_sha256": latest_attachment.get("source_sha256"),
        "dataset_attachment_id": latest_attachment.get("attachment_id"),
    }
    if any(args.get(key) != value for key, value in expected.items()):
        return _denied_message(
            request,
            code="nir_dataset_binding_invalid",
            error="Workflow Dataset lineage does not match the latest verified attachment.",
            workflow=None,
            details={"action_required": "copy_all_lineage_fields_from_nir_dataset_attach"},
        )
    return None


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
    if tool_name == "nir_train_partitioned_model":
        args = request.tool_call.get("args")
        args = args if isinstance(args, Mapping) else {}
        requested_scope = str(args.get("validation_scope") or "independent_external_validation").strip().lower()
        required_scope = "independent_external_validation" if validation_goal in {"external_validation", "production"} else "independent_holdout_not_external"
        if requested_scope == required_scope:
            return None
        return _denied_message(
            request,
            code=_VALIDATION_GOAL_CONFLICT_CODE,
            error=(f"Tool {tool_name!r} requested validation_scope={requested_scope!r}, but validation_goal={validation_goal!r} requires {required_scope!r}."),
            workflow=workflow,
            policy=policy,
            details={
                "validation_goal": validation_goal,
                "requested_validation_scope": requested_scope,
                "required_validation_scope": required_scope,
                "allowed_modeling_tools": ["nir_train_partitioned_model"],
                "action_required": "retry_partitioned_model_with_required_validation_scope",
            },
        )
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


def _chemotools_catalog_attempted(workflow: Mapping[str, Any]) -> bool:
    """Return whether this workflow tried the authoritative MCP catalog.

    A failed catalog call is still an attempt: it authorizes the documented
    native fallback while keeping the outage visible in the durable trace.
    """

    observations = workflow.get("tool_observations")
    if not isinstance(observations, list):
        return False
    return any(isinstance(observation, Mapping) and observation.get("name") == _CHEMOTOOLS_CATALOG_TOOL and observation.get("status") in {"success", "error"} for observation in observations)


def _deny_missing_chemotools_catalog(
    request: ToolCallRequest,
    workflow: Mapping[str, Any],
    policy: _ToolPolicy,
) -> ToolMessage | None:
    tool_name = str(request.tool_call.get("name", ""))
    if tool_name not in _PREPROCESSING_DECISION_TOOLS or _chemotools_catalog_attempted(workflow):
        return None
    return _denied_message(
        request,
        code="nir_chemotools_catalog_required",
        error=("Query the Chemotools MCP capability catalog before selecting or executing a preprocessing or modeling path. If the catalog call fails, the recorded failure permits the governed native fallback."),
        workflow=workflow,
        policy=policy,
        details={
            "required_tool": _CHEMOTOOLS_CATALOG_TOOL,
            "action_required": "query_chemotools_catalog_before_preprocessing_decision",
            "fallback_policy": "native_only_after_recorded_mcp_failure_or_no_equivalent_capability",
        },
    )


def _deny_plan_ready_without_chemotools_catalog(
    request: ToolCallRequest,
    workflow: Mapping[str, Any],
    args: Mapping[str, Any],
) -> ToolMessage | None:
    if str(request.tool_call.get("name", "")) != "nir_workflow":
        return None
    if str(args.get("action", "")).strip().lower() != "plan_ready":
        return None
    if workflow.get("task_type") not in _MODEL_TASKS:
        return None
    if workflow.get("validation_goal") == "exploratory":
        return None
    if _chemotools_catalog_attempted(workflow):
        return None
    return _denied_message(
        request,
        code="nir_chemotools_catalog_required",
        error=("Query the Chemotools MCP capability catalog before finalizing the preprocessing and modeling plan."),
        workflow=workflow,
        details={
            "required_tool": _CHEMOTOOLS_CATALOG_TOOL,
            "action_required": "query_chemotools_catalog_before_preprocessing_decision",
            "fallback_policy": "native_only_after_recorded_mcp_failure_or_no_equivalent_capability",
        },
    )


def _authorize(request: ToolCallRequest) -> ToolMessage | None:
    tool_name = str(request.tool_call.get("name", ""))
    workflow = _state_from_request(request).get("nir_workflow")
    args = request.tool_call.get("args")
    args = args if isinstance(args, Mapping) else {}
    if waiting_for_user(workflow):
        own_question = tool_name == "ask_clarification" and request.tool_call.get("id") == workflow["user_question"].get("tool_call_id")
        if own_question:
            return None
        if not own_question and not (tool_name == "nir_workflow" and args.get("action") == "status"):
            return _denied_message(request, code="nir_waiting_for_user", error="正在等待你的回复，确认前不会继续建模。", workflow=workflow)
    # Honor the user's explicit strategy request before a workflow exists.
    # This rejects a downgrade; it does not invent a modeling decision.
    from deerflow.community.nir.web_entry import web_request

    web = web_request(getattr(getattr(request, "runtime", None), "context", None))
    from deerflow.community.nir.web_entry import completed_intake_workflow, unified_entry

    unified = unified_entry(getattr(getattr(request, "runtime", None), "context", None))
    if unified and (not workflow or completed_intake_workflow(workflow)) and tool_name in (_MODELING_TOOLS | _CHEMOTOOLS_EXECUTION_TOOLS | {"nir_search_model", "nir_preprocess"}):
        return _denied_message(request, code="nir_unified_start_required", error="Interpret the task and start the owned NIR workflow before any modeling. Ordinary questions do not start training.", workflow=workflow)
    if (
        web
        and isinstance(workflow, Mapping)
        and not all(
            workflow.get(key) == value
            for key, value in {
                "search_protocol": "sealed_holdout",
                "decision_strategy": "llm_autonomous",
                "autonomous_rounds": True,
                "autonomous_delivery": True,
            }.items()
        )
    ):
        if not (tool_name == "nir_workflow" and args.get("action") == "status"):
            return _denied_message(
                request, code="nir_web_mode_conflict", error="This chat already has a workflow with different mode settings. Use a fresh chat; existing sessions cannot be upgraded or reset by a web mode selection.", workflow=workflow
            )
    if (web or not unified and _delivery_requested(_latest_user_text(request))) and not (isinstance(workflow, Mapping) and workflow.get("search_protocol") == "sealed_holdout"):
        required = {"search_protocol": "sealed_holdout", "decision_strategy": "llm_autonomous", "autonomous_rounds": True, "autonomous_delivery": True}
        allowed_read = tool_name in _CHEMOTOOLS_READ_TOOLS or tool_name in {"nir_search_method_knowledge", "nir_list_preprocessing_methods", "nir_describe_preprocessing_method"}
        starting = tool_name == "nir_workflow" and args.get("action") == "start" and all(args.get(key) == value for key, value in required.items())
        if starting and web:
            starting = args.get("budget_profile", web["budget_profile"]) == web["budget_profile"] and args.get("task_budget", web["task_budget"]) == web["task_budget"]
        if not starting and not allowed_read and not (tool_name == "nir_workflow" and args.get("action") == "status"):
            return _denied_message(
                request,
                code="nir_delivery_mode_required",
                error="The user requested autonomous delivery. Start with all required mode flags; do not fall back to legacy.",
                workflow=workflow,
                details={"required_start_fields": required, "validation_goal": "internal_holdout", "budget_arguments": "task_budget JSON only"},
            )
    if isinstance(workflow, Mapping) and workflow.get("search_protocol") == "sealed_holdout":
        if (workflow.get("delivery_policy") or {}).get("automatic_delivery"):
            if tool_name == "nir_prepare_search" and workflow.get("search_session_id"):
                return _denied_message(request, code="nir_prepare_already_bound", error="Preparation is already persisted. Continue with the current session; no new partition or budget is allowed.", workflow=workflow)
            if tool_name == "nir_decision_context" and workflow.get("decision_context_id"):
                return _denied_message(
                    request, code="nir_context_already_available", error="The unchanged context is already available. Submit against its context_id/revision/hash or read detailed history; do not request it again.", workflow=workflow
                )
        allowed = {
            "nir_prepare_search",
            "nir_search_model",
            "nir_freeze_search",
            "nir_finalize_search",
            "nir_search_method_knowledge",
            "nir_describe_preprocessing_method",
            "nir_list_preprocessing_methods",
            "chemotools_health",
            "chemotools_list_capabilities",
            "chemotools_describe_capability",
        }
        workflow_read = tool_name == "nir_workflow" and args.get("action") == "status"
        if workflow.get("decision_strategy") == "llm_autonomous":
            allowed -= {"nir_search_model", "nir_freeze_search", "nir_finalize_search"}
            allowed |= {"nir_decision_context", "nir_submit_decision", "nir_execute_decision", "nir_decision_history"}
            if workflow.get("autonomous_delivery"):
                if workflow.get("stage") not in {"blocked", "completed", "registered"}:
                    allowed |= {"ask_clarification"}
                if not (workflow.get("delivery_policy") or {}).get("automatic_delivery"):
                    allowed |= {"nir_freeze_search", "nir_finalize_search"}
                if not workflow.get("search_session_id") and workflow.get("stage") in {"intake", "clarification"}:
                    workflow_read = tool_name == "nir_workflow" and args.get("action") in {"status", "set_requirements"}
                    allowed |= {"ask_clarification"}
                if workflow.get("stage") in {"review", "approved", "registered"}:
                    workflow_read = tool_name == "nir_workflow" and args.get("action") in {"status", "approve", "reject", "complete"}
                    allowed |= {"nir_register_model"}
        if tool_name not in allowed and not workflow_read:
            return _denied_message(request, code="nir_sealed_search_tool_denied", error="The sealed regression protocol permits only governed search/finalization and data-free catalog tools.", workflow=workflow)
    if dataset_binding_denial := _deny_invalid_dataset_start_binding(request):
        return dataset_binding_denial
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
                error=("Python/R/MATLAB analysis scripts are forbidden during an active NIR workflow. Use nir_inspect, nir_load_data, nir_preprocess, nir_align_wavelengths, and NIR modeling tools only."),
                workflow=workflow,
            )

    if isinstance(workflow, Mapping):
        if catalog_plan_denial := _deny_plan_ready_without_chemotools_catalog(
            request,
            workflow,
            args,
        ):
            return catalog_plan_denial

    policy = _TOOL_POLICIES.get(tool_name)
    if policy is None:
        return None

    if not isinstance(workflow, Mapping):
        # Library discovery/save/attach precedes workflow start: attach must
        # produce the virtual path that nir_workflow(action="start") binds.
        if tool_name in _DATASET_LIBRARY_TOOLS or tool_name in _MODEL_REUSE_TOOLS:
            return None
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
    if catalog_denial := _deny_missing_chemotools_catalog(request, workflow, policy):
        return catalog_denial
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
    if tool_name == "nir_train_partitioned_model":
        emitted_protocol = str(result.get("protocol") or "")
        emitted_scope = str(result.get("validation_scope") or "")
        if (emitted_protocol, emitted_scope) in _PARTITIONED_PROTOCOLS:
            protocol, validation_scope = emitted_protocol, emitted_scope
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
    result_facts = model_result_facts(result)
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
        "axis_first",
        "axis_last",
        "axis_direction",
        "constant_wavelength_count",
        "usable_wavelength_count",
        "wavelength_range_semantics",
        "y_names",
        "y_range",
        "y_ranges",
        "y_separated",
        "wv_separated",
        "categorical_columns",
        "classification_hint",
        "format",
        "structure",
        "value_range",
        "has_nan",
        "labeled_matrix",
        "layout_pattern",
        "auto_load_supported",
        "target_columns",
        "metadata_columns",
        "has_reference",
        "has_wavelength_axis",
    )
    evidence = {
        "source_tool": tool_name,
        "data_path": str(args.get("file_path") or args.get("data_path") or workflow.get("data_path") or ""),
        **{key: payload[key] for key in payload_fields if payload.get(key) is not None},
    }
    if "raw_wavelength_range" not in evidence and payload.get("wavelength_range") is not None:
        evidence["raw_wavelength_range"] = payload["wavelength_range"]
    if "n_samples" not in evidence and payload.get("estimated_samples") is not None:
        evidence["n_samples"] = payload["estimated_samples"]
    if "n_wavelengths" not in evidence and payload.get("estimated_wavelengths") is not None:
        evidence["n_wavelengths"] = payload["estimated_wavelengths"]
    for key in ("y_col", "y_cols", "x_cols", "wv_row", "x_var", "y_var", "wv_var", "subset", "transpose"):
        if args.get(key) is not None:
            evidence[key] = args[key]
    updated = transition_workflow(
        workflow,
        action="record_audit_evidence",
        audit_evidence=evidence,
        notes=f"Automatically recorded bounded {tool_name} evidence.",
    )
    if workflow.get("task_type") == "inspection" and tool_name == "nir_inspect":
        return transition_workflow(
            updated,
            action="complete",
            notes="Inspection completed successfully; no modeling was requested.",
        )
    return updated


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
        registered = transition_workflow(
            workflow,
            action="registered",
            notes=f"Registered model {payload.get('model_id', '')}".strip(),
        )
        registered["registered_model"] = {"model_id": payload.get("model_id"), "version": payload.get("version")}
        return registered
    if tool_name == "nir_model_promote" and payload.get("library_status") == "ready":
        return {**workflow, "model_library": {"status": "saved", "model_id": payload.get("model_id"), "version": payload.get("version")}}
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


def _registration_save_context(request: ToolCallRequest, result: ToolMessage | Command) -> tuple[dict, dict] | None:
    if request.tool_call.get("name") != "nir_register_model" or not isinstance(result, Command) or not isinstance(result.update, Mapping):
        return None
    workflow, payload = result.update.get("nir_workflow"), _success_payload(result)
    if not isinstance(workflow, dict) or workflow.get("stage") != "registered" or payload is None or payload.get("status") != "registered":
        return None
    return workflow, payload


def _with_model_library_save(result: Command, workflow: dict, payload: dict, outcome: dict) -> Command:
    message = _tool_message(result)
    if message is None:
        return result
    updated_message = message.model_copy(update={"content": json.dumps({**payload, "model_library": outcome}, ensure_ascii=False)})
    return replace(
        result,
        update={
            **result.update,
            "nir_workflow": {**workflow, "model_library": outcome},
            "messages": [updated_message if item is message else item for item in result.update.get("messages", [])],
        },
    )


def _workflow_completion_reminder(
    required_action: str,
    workflow: Mapping[str, Any],
) -> str:
    instructions = {
        "read_decision_context": "Read nir_decision_context for persisted experiment feedback and remaining cumulative budget. Independently compare next actions, including STOP; quality and knowledge rank do not dictate your choice.",
        "read_final_selection_context": "Search budget is exhausted. Read nir_decision_context for existing Cal/Val results and choose a successful retained experiment using RETAIN, zero fits. Stop source remains budget.",
        "submit_final_selection": "Submit phase=final_selection/action=RETAIN with successful retained_experiment_id, stop_evidence, null execution plan, expected_fits=0 and concise evidence. Never automatically select the lowest RMSE.",
        "freeze_selected_model": "Call nir_freeze_search with exactly the retained_experiment_id in the accepted terminal decision. No refit or new parameter search.",
        "submit_decision": "Submit your selected action with alternatives, reasons for rejecting them, uncertainty, hypotheses, concrete configuration and fit estimate. STOP must name your chosen successful retained experiment.",
        "execute_decision": "Execute only the accepted decision_id, then read a fresh decision context for the next LLM action. Never read Test or call legacy training.",
        "read_initial_context": "Call nir_decision_context; choose your initial plan from Cal features, actual capabilities and sourced method evidence.",
        "submit_initial_decision": "Submit your structured initial alternatives and selected pipeline using nir_submit_decision. Do not call direct search or legacy fitting.",
        "execute_initial_decision": "Call nir_execute_decision with the accepted decision_id, return its Cal CV/Val evidence, then wait. No subsequent loop is enabled.",
        "prepare_sealed_search": "Call nir_prepare_search to ingest the bound source and seal Test. Do not use legacy inspection/modeling tools.",
        "search_or_freeze": "Use nir_search_model on explicit pipelines or nir_freeze_search with a selected successful experiment. Search metrics only describe Cal/Val.",
        "finalize_search": "Call nir_finalize_search once for the frozen model; never reopen search after Test evaluation.",
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


def _delivery_requested(text):
    return bool(re.search(r"\bautonomous_delivery\s*[=:]\s*true\b", text, re.I))


def _user_task_message(message):
    return isinstance(message, HumanMessage) and not any(message.additional_kwargs.get(key) for key in ("slash_skill_activation", "durable_context_data", "dynamic_context_reminder"))


class NIRWorkflowMiddleware(AgentMiddleware[AgentState]):
    """Enforce NIR workflow policy and persist deterministic tool outcomes."""

    @staticmethod
    def _restore_unified_binding(state, runtime):
        from types import SimpleNamespace

        from deerflow.community.nir.process import runtime_store
        from deerflow.community.nir.web_entry import stored_workflow

        config = getattr(runtime, "config", None)
        if not config:
            from langgraph.config import get_config

            try:
                config = get_config()
            except RuntimeError:
                return None
        stored = stored_workflow(runtime_store(SimpleNamespace(config=config, context=runtime.context)))
        if stored:
            if stored.get("search_session_id"):
                from deerflow.community.nir.dataset_session import SearchSessionStore

                service_runtime = SimpleNamespace(config=config, context=runtime.context, state={**state, "nir_workflow": stored})
                search = SearchSessionStore(runtime_store(service_runtime))
                with search.process._connection() as db:
                    session = search._session(db, stored["search_session_id"])
                auto = session.get("automatic_delivery") or {}
                if auto.get("workflow"):
                    stored = auto["workflow"] if auto["workflow"].get("revision", 0) > stored.get("revision", 0) else stored
                if not waiting_for_user(stored) and auto.get("status") in {"queued", "failed"} and not auto.get("lease") and session["stage"] in {"stopped", "frozen", "finalized"}:
                    from deerflow.community.nir.automatic_delivery import complete_terminal_delivery
                    from deerflow.community.nir.web_entry import persist_unified_workflow

                    pending = {**stored, "retained_experiment_id": session["retained_experiment_id"], "stop_decision_id": session["stop_decision_id"], "stop_source": session["stop_source"]}
                    try:
                        _, stored = complete_terminal_delivery(service_runtime, search, pending)
                        persist_unified_workflow(service_runtime, stored)
                    except Exception:
                        logger.warning("Unable to resume deterministic terminal publication; state remains durable", exc_info=True)
            current = state.get("nir_workflow") or {}
            if not current or current.get("revision", 0) < stored.get("revision", 0) or any(current.get(key) != stored.get(key) for key in ("project_id", "search_protocol", "decision_strategy", "delivery_policy", "search_session_id")):
                return {"nir_workflow": stored}
        return None

    @staticmethod
    def _deterministic_delivery_response(request):
        workflow = request.state.get("nir_workflow") or {}
        if not (workflow.get("delivery_policy") or {}).get("automatic_delivery") or workflow.get("stage") != "review":
            return None
        # A separate human approval/rejection is still handled by the original
        # tools. Only the same run's completed terminal submission is narrated.
        last = next((message for message in reversed(request.messages) if isinstance(message, (HumanMessage, ToolMessage))), None)
        if not isinstance(last, ToolMessage) or last.name != "nir_submit_decision":
            return None
        try:
            value = json.loads(last.content)
        except (ValueError, TypeError):
            return None
        if (value.get("automatic_delivery") or {}).get("status") != "delivered":
            return None
        paths = value["automatic_delivery"]
        text = render_grounded_nir_response(workflow, include_downloads=False) + "\n\n报告已生成，可在下方查看或下载。"
        text += "\n\n" + " · ".join(f"[{label}]({paths[key]})" for label, key in (("模型文件", "model"), ("查看报告", "report_html"), ("下载全部文件", "bundle")))
        return ModelResponse(result=[AIMessage(content=text, additional_kwargs={"nir_deterministic_delivery": True, "nir_delivery_session_id": workflow["search_session_id"]})])

    @staticmethod
    def _persist_question(state, runtime, workflow):
        from types import SimpleNamespace

        from deerflow.community.nir.web_entry import persist_unified_workflow

        config = getattr(runtime, "config", None)
        if not config:
            from langgraph.config import get_config

            try:
                config = get_config()
            except RuntimeError:
                return
        persist_unified_workflow(SimpleNamespace(state=state, context=getattr(runtime, "context", None), config=config), workflow)

    def before_agent(self, state, runtime):
        restored = self._restore_unified_binding(state, runtime)
        workflow = (restored or {}).get("nir_workflow") or state.get("nir_workflow") or {}
        resumed = resume_user_wait(workflow, state.get("messages") or [])
        if resumed:
            self._persist_question(state, runtime, resumed)
            return {**(restored or {}), "nir_workflow": resumed}
        return restored

    async def abefore_agent(self, state, runtime):
        return await asyncio.to_thread(self.before_agent, state, runtime)

    @staticmethod
    def _with_unified_intake(request):
        from deerflow.community.nir.web_entry import completed_intake_workflow, unified_entry

        workflow = request.state.get("nir_workflow")
        if not unified_entry(getattr(getattr(request, "runtime", None), "context", None)) or (workflow and not completed_intake_workflow(workflow)):
            return request
        from langchain_core.messages import SystemMessage

        instruction = (
            "\nFor a new NIR single-target regression execution request, interpret the actual user requirement and uploaded data, then call nir_workflow(action=start, task_type=calibration). "
            "The trusted server supplies the scientific protocol and resource policy; omit all mode/budget arguments. "
            "Owned numeric MAT uploads are supported; a completed non-modeling inspection may start regression in this thread with its evidence archived. "
            "An omitted validation_goal means internal_holdout; explicit external validation cannot use this protocol. "
            "If using an attached dataset, copy every lineage field from the successful nir_dataset_attach result exactly. "
            "Conceptual questions (e.g. what is SNV), comparisons or unclear execution intent remain ordinary QA; ask clarification when data/target are missing. "
            "Do not train from keyword matches, invent target units, or use legacy/raw/MCP fits for this task. Other task types keep their existing tools."
        )
        tools = []
        hidden = {"search_protocol", "decision_strategy", "autonomous_rounds", "autonomous_delivery", "max_attempts", "max_fits", "max_decisions", "budget_profile", "task_budget"}
        for item in request.tools:
            if getattr(item, "name", None) == "nir_workflow":
                schema = {"type": "object", "properties": {key: value for key, value in item.args.items() if key not in hidden}, "required": ["action"], "additionalProperties": False}
                item = item.model_copy(update={"args_schema": schema})
            tools.append(item)
        original = message_content_to_text(request.system_message.content) if request.system_message else ""
        from deerflow.community.nir.web_entry import unified_policy

        settings = dict(request.model_settings)
        output_cap = unified_policy()["max_output_tokens"]
        settings["max_tokens"] = min(settings.get("max_tokens") or getattr(request.model, "max_tokens", None) or output_cap, output_cap)
        return request.override(tools=tools, model_settings=settings, system_message=SystemMessage(content=original + instruction))

    @staticmethod
    def _with_delivery_contract(request):
        workflow = request.state.get("nir_workflow") or {}
        from deerflow.community.nir.web_entry import web_request

        runtime_context = getattr(getattr(request, "runtime", None), "context", None)
        web = web_request(runtime_context)
        from deerflow.community.nir.web_entry import unified_entry

        if not workflow and not web and unified_entry(runtime_context):
            return request
        requested = any(_user_task_message(message) and _delivery_requested(get_original_user_content_text(message.content, message.additional_kwargs)) for message in request.messages)
        if unified_entry(runtime_context):
            requested = False  # Task intent comes from the actual start call.
        if not workflow.get("autonomous_delivery") and not requested and not web:
            return request
        from deerflow.community.nir.decision_schema import bound_tool_schema, proposal_schema

        context = None
        for message in reversed(request.messages):
            if isinstance(message, ToolMessage) and message.name == "nir_decision_context":
                try:
                    value = json.loads(message.content)
                    if value.get("context_id") == workflow.get("decision_context_id"):
                        if not value.get("proposal_schema") and value.get("proposal_schema_source") == "nir_submit_decision.current_tool_schema":
                            value = {**value, "proposal_schema": proposal_schema(value["phase"])}
                        if value.get("proposal_schema"):
                            context = value
                            break
                except (TypeError, ValueError):
                    pass
        stage = workflow.get("stage")
        if not workflow.get("autonomous_delivery"):
            names = {"nir_workflow"}
        elif waiting_for_user(workflow) or stage == "blocked":
            names = {"nir_workflow"}
        elif not workflow.get("search_session_id") and workflow.get("missing_inputs"):
            names = {"nir_workflow", "ask_clarification"}
        elif not workflow.get("search_session_id"):
            names = {"nir_prepare_search"}
        elif stage == "awaiting_final_confirmation":
            names = {"nir_freeze_search"}
        elif stage == "frozen":
            names = {"nir_finalize_search"}
        elif stage in {"review", "registered", "completed", "blocked"}:
            names = {"nir_workflow"}
        elif stage == "approved":
            names = {"nir_workflow", "nir_register_model"}
        elif workflow.get("active_decision_id"):
            names = {"nir_execute_decision"}
        elif context:
            names = {"nir_submit_decision", "nir_decision_history"}
        else:
            names = {"nir_decision_context"}
        if workflow.get("autonomous_delivery") and not waiting_for_user(workflow) and stage not in {"blocked", "completed", "registered"}:
            names.add("ask_clarification")
        tools = []
        for item in request.tools:
            name = getattr(item, "name", None)
            if name not in names:
                continue
            schema = None
            if name == "nir_submit_decision" and context:
                schema = bound_tool_schema(context)
            elif name == "nir_workflow" and workflow.get("autonomous_delivery"):
                if not workflow.get("search_session_id") and workflow.get("missing_inputs"):
                    fields = {"action", "data_path", "domain", "analyte", "unit", "validation_goal"}
                    properties = {key: value for key, value in item.args.items() if key in fields}
                    properties["action"] = {"type": "string", "enum": ["status", "set_requirements"]}
                    schema = {"type": "object", "properties": properties, "required": ["action"], "additionalProperties": False}
                else:
                    actions = ["status", "approve", "reject"] if stage == "review" else ["status", "complete"] if stage in {"approved", "registered"} else ["status"]
                    schema = {"type": "object", "properties": {"action": {"type": "string", "enum": actions}, "notes": {"type": "string"}}, "required": ["action"], "additionalProperties": False}
            elif name == "nir_workflow" and not workflow.get("autonomous_delivery"):
                allowed = {
                    "action",
                    "task_type",
                    "data_path",
                    "analyte",
                    "unit",
                    "domain",
                    "validation_goal",
                    "search_protocol",
                    "decision_strategy",
                    "autonomous_rounds",
                    "autonomous_delivery",
                    "budget_profile",
                    "task_budget",
                    "model_constraint",
                    "pipeline_constraint",
                }
                properties = {key: value for key, value in item.args.items() if key in allowed}
                constants = {
                    "action": "start",
                    "task_type": "calibration",
                    "validation_goal": "internal_holdout",
                    "search_protocol": "sealed_holdout",
                    "decision_strategy": "llm_autonomous",
                    "autonomous_rounds": True,
                    "autonomous_delivery": True,
                }
                for key, value in constants.items():
                    properties[key] = {"const": value, "type": "boolean" if isinstance(value, bool) else "string"}
                if web:
                    for key in ("budget_profile", "task_budget"):
                        properties[key] = {"const": web[key], "type": "string"}
                schema = {"type": "object", "properties": properties, "required": [*constants, "data_path", "analyte", "unit", "domain"], "additionalProperties": False}
            tools.append(item.model_copy(update={"args_schema": schema}) if schema else item)
        messages = request.messages
        if context and any(item.name == "nir_submit_decision" for item in tools):
            from deerflow.community.nir.decision_provider import context_message_view

            messages = list(messages)
            for index, message in enumerate(messages):
                if isinstance(message, ToolMessage) and message.name == "nir_decision_context":
                    try:
                        if json.loads(message.content).get("context_id") == context["context_id"]:
                            messages[index] = context_message_view(message, context)
                    except (TypeError, ValueError):
                        pass
        from deerflow.community.nir.decision_provider import autonomous_skill_message_view

        messages = [autonomous_skill_message_view(message) for message in messages]
        return request.override(tools=tools, messages=messages)

    @staticmethod
    def _with_delivery_state(request):
        workflow = request.state.get("nir_workflow") or {}
        if not workflow.get("autonomous_delivery"):
            return request
        from langchain_core.messages import SystemMessage

        view = {key: workflow.get(key) for key in ("stage", "search_session_id", "decision_context_id", "active_decision_id", "retained_experiment_id", "stop_source", "user_question")}
        view["required_workflow_action"] = required_nir_workflow_action(workflow)
        instruction = (
            "\n<current_sealed_workflow>" + json.dumps(view) + "</current_sealed_workflow>\n"
            "This is the current authoritative process state. Do not repeat preparation after a session is bound. "
            "Use the current context to submit your own decision; this state does not select a model or optimization action."
            " Write user-facing answers for ordinary Chinese-speaking users: begin with whether the task finished, the result and one useful next step. "
            "Default to a short Chinese answer. Explain R² as 越接近 1 越好 and RMSE as 越小越好, rounding displayed metrics to four decimal places. "
            "For internal holdout results, explain: 本次测试使用从这份数据中单独留出的样本，尚未用另外采集的数据验证。 "
            "Keep protocol names, hashes, internal stages, evidence/quality gates, freezing, raw tool names and experiment ledgers in the report; show technical details only when the user asks. "
            "Say saved to the model library only after the trusted model_library outcome confirms saved; approval and registration alone do not confirm library storage."
            " Ask the user only when their input is necessary, using ask_clarification. Once you ask a question or offer a choice, wait for their reply; never ask and execute a choice in the same turn. "
            "A new reply does not approve registration or select a model automatically. If the reply does not answer the pending question, clarify again rather than choose for the user."
        )
        if (workflow.get("delivery_policy") or {}).get("automatic_delivery"):
            instruction += (
                " Cumulative tokens are statistics only. After your legal STOP/RETAIN and named successful experiment, the server automatically freezes, evaluates Test once and publishes deterministic reports. "
                "Do not call freeze/finalize or request more model narration for delivery."
            )
        original = message_content_to_text(request.system_message.content) if request.system_message else ""
        return request.override(system_message=SystemMessage(content=original + instruction))

    @staticmethod
    def _prepare_budget_call(request):
        workflow = request.state.get("nir_workflow") or {}
        from deerflow.community.nir.web_entry import web_request

        web = web_request(getattr(getattr(request, "runtime", None), "context", None))
        from deerflow.community.nir.web_entry import unified_entry

        attempted_unified = unified_entry(getattr(getattr(request, "runtime", None), "context", None)) and any(
            isinstance(message, AIMessage) and any(call["name"] == "nir_workflow" and call["args"].get("action") == "start" and call["args"].get("task_type") in {"calibration", "analysis"} for call in message.tool_calls)
            for message in request.messages
        )
        requested = any(_user_task_message(message) and _delivery_requested(get_original_user_content_text(message.content, message.additional_kwargs)) for message in request.messages)
        if unified_entry(getattr(getattr(request, "runtime", None), "context", None)):
            requested = False
        if not workflow.get("autonomous_delivery") and not requested and not web and not attempted_unified:
            return request, None
        from types import SimpleNamespace

        from langgraph.config import get_config

        from deerflow.community.nir.dataset_session import SearchSessionStore
        from deerflow.community.nir.decision_budget import reserve_call
        from deerflow.community.nir.process import runtime_store

        search = SearchSessionStore(runtime_store(SimpleNamespace(config=get_config(), context=request.runtime.context)))
        sid = workflow.get("search_session_id")
        if sid:
            with search.process._connection() as db:
                session = search._session(db, sid)
            if session.get("export"):
                return request, None  # Existing approval dialogue uses normal runtime limits.
        # Conservative byte-based admission estimate includes tool schemas. It is
        # explicitly not an exact tokenizer or a guarantee about provider billing.
        estimate = NIRWorkflowMiddleware._admission_estimate(request)
        if not sid:
            from deerflow.community.nir.decision_budget import PROFILES, SERVER_DEFAULTS, reserve_intake_call

            policy = workflow.get("delivery_policy") or (web or {}).get("delivery_policy")
            if not policy and attempted_unified:
                from deerflow.community.nir.web_entry import unified_policy

                policy = unified_policy()
            if not policy:
                import os

                policy = {key: min(value, int(os.environ.get("NIR_AUTONOMOUS_" + key.upper(), SERVER_DEFAULTS[key]))) for key, value in PROFILES["development"].items()}
            try:
                initial_receipts = None
                if policy.get("profile") == "unified":
                    with search.process._connection() as db:
                        initial_receipts = [json.loads(row[0]) for row in db.execute("SELECT payload FROM model_issued_calls WHERE tool='__model_response__'")]
                    run_id = (request.runtime.context or {}).get("run_id")
                    initial_receipts = [receipt for receipt in initial_receipts if receipt.get("agent_run_id") == run_id and not receipt.get("intake_lease_id") and not receipt.get("session_id")]
                lease = reserve_intake_call(search.process, policy, estimate, initial_receipts=initial_receipts)
            except ValueError as exc:
                return request, ModelResponse(
                    result=[
                        AIMessage(
                            content="任务准备阶段的资源或安全额度已关闭；未训练模型、未使用 Test。原因：" + str(exc),
                            additional_kwargs={"nir_program_termination": {"stage": "blocked", "stop_source": "budget" if "budget:" in str(exc) else "safety", "reason": str(exc)}},
                        )
                    ]
                )
            settings = dict(request.model_settings)
            settings["max_tokens"] = min(settings.get("max_tokens") or getattr(request.model, "max_tokens", None) or policy["max_output_tokens"], policy["max_output_tokens"])
            return request.override(model_settings=settings), (search.process, lease)
        try:
            lease = reserve_call(search, sid, estimate, defer_finalization=True)
            if lease is None:
                # Admission closed search after the old feedback was displayed.
                # Rebuild the provider's state/tools before spending a final call;
                # do not allow an old TUNE/EXPLORE to consume the RETAIN reserve.
                updated = {**workflow, "stage": "budget_exhausted", "stop_source": "budget", "decision_context_id": None, "active_decision_id": None, "next_action": "read_final_selection_context"}
                request = request.override(state={**request.state, "nir_workflow": updated})
                request = NIRWorkflowMiddleware._compact_decision_messages(request)
                request = NIRWorkflowMiddleware._with_delivery_state(request)
                from deerflow.community.nir.decision_tools import nir_decision_context_tool

                request = request.override(tools=[nir_decision_context_tool])
                estimate = NIRWorkflowMiddleware._admission_estimate(request)
                lease = reserve_call(search, sid, estimate)
        except ValueError as exc:
            with search.process._connection() as db:
                current = search._session(db, sid)
            return request, ModelResponse(
                result=[
                    AIMessage(
                        content="资源或安全约束已关闭当前建模流程；没有自动选择模型或重新使用 Test。原因：" + str(exc),
                        additional_kwargs={"nir_program_termination": {"stage": current["stage"], "stop_source": current.get("stop_source", "safety"), "reason": str(exc)}},
                    )
                ]
            )
        settings = dict(request.model_settings)
        settings["max_tokens"] = min(settings.get("max_tokens") or getattr(request.model, "max_tokens", None) or session["delivery_policy"]["max_output_tokens"], session["delivery_policy"]["max_output_tokens"])
        return request.override(model_settings=settings), (search, sid, lease)

    @staticmethod
    def _finish_budget_call(binding, response, seconds, *, failed=False, failure_reason=None):
        if binding is None:
            return
        from deerflow.community.nir.decision_budget import finish_call, finish_intake_call

        messages = getattr(response, "result", [response]) if response else []
        usage = next((message.usage_metadata for message in messages if isinstance(message, AIMessage) and message.usage_metadata), None)
        finish = finish_intake_call if len(binding) == 2 else finish_call
        finish(*binding, usage, seconds, failed=failed, **({"failure_reason": failure_reason} if len(binding) == 3 else {}))

    @staticmethod
    def _capture_model_calls(request, response, elapsed_seconds=None):
        workflow = request.state.get("nir_workflow") or {}
        response_messages = getattr(response, "result", [response])
        start_call = next(
            (call for message in response_messages if isinstance(message, AIMessage) for call in message.tool_calls if call["name"] == "nir_workflow" and call["args"].get("action") == "start"),
            None,
        )
        if workflow.get("decision_strategy") != "llm_autonomous" and not start_call:
            return
        from types import SimpleNamespace

        from langgraph.config import get_config

        from deerflow.community.nir.decision_origin import record_model_response
        from deerflow.community.nir.process import runtime_store

        config = get_config()
        runtime = SimpleNamespace(state=request.state, context=request.runtime.context, config=config)
        context = request.runtime.context
        process = runtime_store(runtime)
        if process is None:
            return
        from deerflow.community.nir.decision_budget import INTAKE_ID

        with process._connection() as db:
            row = db.execute("SELECT payload FROM model_issued_calls WHERE id=?", (INTAKE_ID,)).fetchone()
        intake_lease = (json.loads(row[0])["budget"].get("inflight") or {}) if row else {}
        record_model_response(
            process,
            request.model,
            response_messages,
            run_id=context.get("run_id") if isinstance(context, dict) else config.get("metadata", {}).get("run_id"),
            session_id=workflow.get("search_session_id"),
            elapsed_seconds=elapsed_seconds,
            project_id=workflow.get("project_id") if workflow.get("autonomous_delivery") else None,
            workflow_start_call_id=workflow.get("workflow_start_call_id") or (start_call["id"] if start_call else None),
            estimated_tokens=NIRWorkflowMiddleware._admission_estimate(request),
            request_profile=NIRWorkflowMiddleware._request_cost_profile(request),
            intake_lease_id=intake_lease.get("id"),
        )

    @override
    def wrap_model_call(self, request: ModelRequest, handler):
        workflow = request.state.get("nir_workflow") or {}
        if waiting_for_user(workflow):
            return ModelResponse(result=[AIMessage(content=waiting_message(workflow), id="nir-waiting:" + str(workflow["user_question"]["message_id"]), additional_kwargs={"nir_waiting_for_user": True})])
        if workflow.get("runtime_termination"):
            return ModelResponse(result=[AIMessage(content=render_grounded_nir_response(workflow), additional_kwargs={"nir_runtime_stopped": True})])
        completed = self._deterministic_delivery_response(request)
        if completed is not None:
            return completed
        request = self._with_unified_intake(request)
        request = self._without_autonomous_memory(request)
        request = self._compact_decision_messages(request)
        request = self._with_delivery_state(request)
        request = self._with_delivery_contract(request)
        request, binding = self._prepare_budget_call(request)
        if isinstance(binding, ModelResponse):
            return binding
        import time

        started = time.monotonic()
        try:
            response = handler(request)
        except BaseException as exc:
            self._finish_budget_call(binding, None, time.monotonic() - started, failed=True, failure_reason={"exception_type": type(exc).__name__, "http_status": getattr(exc, "status_code", None)})
            raise
        self._capture_model_calls(request, response, time.monotonic() - started)
        self._finish_budget_call(binding, response, time.monotonic() - started)
        return response

    @override
    async def awrap_model_call(self, request: ModelRequest, handler):
        workflow = request.state.get("nir_workflow") or {}
        if waiting_for_user(workflow):
            return ModelResponse(result=[AIMessage(content=waiting_message(workflow), id="nir-waiting:" + str(workflow["user_question"]["message_id"]), additional_kwargs={"nir_waiting_for_user": True})])
        if workflow.get("runtime_termination"):
            return ModelResponse(result=[AIMessage(content=render_grounded_nir_response(workflow), additional_kwargs={"nir_runtime_stopped": True})])
        completed = self._deterministic_delivery_response(request)
        if completed is not None:
            return completed
        request = self._with_unified_intake(request)
        request = self._without_autonomous_memory(request)
        request = self._compact_decision_messages(request)
        request = self._with_delivery_state(request)
        request = self._with_delivery_contract(request)
        request, binding = await asyncio.to_thread(self._prepare_budget_call, request)
        if isinstance(binding, ModelResponse):
            return binding
        import time

        started = time.monotonic()
        try:
            response = await handler(request)
        except BaseException as exc:
            await asyncio.shield(asyncio.to_thread(self._finish_budget_call, binding, None, time.monotonic() - started, failed=True, failure_reason={"exception_type": type(exc).__name__, "http_status": getattr(exc, "status_code", None)}))
            raise
        await asyncio.to_thread(self._capture_model_calls, request, response, time.monotonic() - started)
        await asyncio.to_thread(self._finish_budget_call, binding, response, time.monotonic() - started)
        return response

    @staticmethod
    def _compact_decision_messages(request):
        if not (request.state.get("nir_workflow") or {}).get("autonomous_rounds"):
            return request
        messages = request.messages
        # Prune only complete governed call/response pairs preceding the newest
        # successful context. The checkpoint, immutable decisions and receipts
        # remain untouched. Never manufacture a history summary.
        context_indexes = []
        for index, message in enumerate(messages):
            if isinstance(message, ToolMessage) and message.name == "nir_decision_context":
                try:
                    if json.loads(message.content).get("context_id"):
                        context_indexes.append(index)
                except (TypeError, ValueError):
                    pass
        wf = request.state.get("nir_workflow") or {}
        closing = wf.get("autonomous_delivery") and (
            wf.get("stage") in {"awaiting_final_confirmation", "frozen", "review", "approved", "registered", "completed"} or wf.get("stage") == "budget_exhausted" and not wf.get("decision_context_id")
        )
        if not context_indexes and not closing:
            return request
        boundary = len(messages) if closing else context_indexes[-1]
        newest_call = None if closing else messages[boundary].tool_call_id
        governed = {"nir_prepare_search", "nir_decision_context", "nir_submit_decision", "nir_execute_decision", "nir_decision_history"}
        if (request.state.get("nir_workflow") or {}).get("autonomous_delivery"):
            governed.remove("nir_prepare_search")  # Keep the immutable session-binding receipt.
        terminal_calls = set()
        if closing:
            for message in messages:
                if isinstance(message, ToolMessage) and message.name == "nir_submit_decision":
                    try:
                        if json.loads(message.content).get("action") in {"STOP", "RETAIN"}:
                            terminal_calls.add(message.tool_call_id)
                    except (TypeError, ValueError):
                        pass
        completed = {message.tool_call_id for message in messages[:boundary] if isinstance(message, ToolMessage)}
        removable = set()
        discarded = set()
        for index, message in enumerate(messages[:boundary]):
            if isinstance(message, AIMessage) and message.tool_calls and all(call["name"] in governed and call["id"] in completed and call["id"] != newest_call and call["id"] not in terminal_calls for call in message.tool_calls):
                discarded.add(index)
                removable.update(call["id"] for call in message.tool_calls)
        retained = [message for index, message in enumerate(messages) if index not in discarded and not (isinstance(message, ToolMessage) and message.tool_call_id in removable)]
        # Malformed arguments are not an executed experiment or a valid tool
        # call. Keep the original checkpoint/receipt, but do not repeatedly send
        # that large invalid JSON back to the provider as a tool call.
        retained = [
            message.model_copy(
                update={
                    "content": "Previous tool response contained invalid JSON and was not executed.",
                    "invalid_tool_calls": [],
                    "additional_kwargs": {key: value for key, value in message.additional_kwargs.items() if key != "tool_calls"},
                }
            )
            if isinstance(message, AIMessage) and message.invalid_tool_calls and not message.tool_calls
            else message
            for message in retained
        ]
        return request.override(messages=retained)

    @staticmethod
    def _without_autonomous_memory(request):
        from deerflow.community.nir.web_entry import web_request

        active = (request.state.get("nir_workflow") or {}).get("decision_strategy") == "llm_autonomous"
        active = active or bool(web_request(getattr(getattr(request, "runtime", None), "context", None)))
        requested = any(
            _user_task_message(message) and re.search(r"llm_autonomous|LLM.{0,15}自主|自主.{0,12}(?:首轮|初始)|autonomous initial", get_original_user_content_text(message.content, message.additional_kwargs), re.I)
            for message in request.messages
        )
        if active or requested:
            messages = [message for message in request.messages if not (message.additional_kwargs.get("dynamic_context_reminder") and "<memory>" in message_content_to_text(message.content))]
            return request.override(messages=messages)
        return request

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
        execution = self._start_process(bound_request)
        token = bind_execution(execution)
        try:
            result = _record_observation(bound_request, _advance(bound_request, handler(bound_request)))
            context = _registration_save_context(bound_request, result)
            if context is not None:
                workflow, payload = context
                try:
                    asyncio.get_running_loop()
                except RuntimeError:
                    outcome = asyncio.run(save_registered_model(bound_request.runtime, workflow, payload))
                else:
                    outcome = failed_model_save("async_runtime_required")
                result = _with_model_library_save(result, workflow, payload, outcome)
            self._finish_process(bound_request, execution, result)
            return result
        except BaseException as exc:
            self._finish_process(bound_request, execution, None, status="cancelled" if isinstance(exc, asyncio.CancelledError) else "failed" if isinstance(exc, Exception) else "interrupted")
            raise
        finally:
            reset_execution(token)

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
        execution = await asyncio.to_thread(self._start_process, bound_request)
        token = bind_execution(execution)
        try:
            result = _record_observation(bound_request, _advance(bound_request, await handler(bound_request)))
            context = _registration_save_context(bound_request, result)
            if context is not None:
                workflow, payload = context
                outcome = await save_registered_model(bound_request.runtime, workflow, payload)
                result = _with_model_library_save(result, workflow, payload, outcome)
            await asyncio.to_thread(self._finish_process, bound_request, execution, result)
            return result
        except BaseException as exc:
            status = "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed" if isinstance(exc, Exception) else "interrupted"
            await asyncio.shield(asyncio.to_thread(self._finish_process, bound_request, execution, None, status=status))
            raise
        finally:
            reset_execution(token)

    @staticmethod
    def _start_process(request):
        try:
            return start_execution(request.runtime, str(request.tool_call.get("name")), str(request.tool_call.get("id")))
        except Exception:
            logger.exception("Unable to record NIR process start")
            return None

    @staticmethod
    def _finish_process(request, execution, result, *, status=None):
        payload = _success_payload(result) if result is not None else None
        workflow = _state_from_request(request).get("nir_workflow") or {}
        if isinstance(result, Command) and isinstance(result.update, Mapping):
            workflow = result.update.get("nir_workflow") or workflow
        from deerflow.community.nir.web_entry import persist_unified_workflow

        persist_unified_workflow(request.runtime, workflow)
        finish_execution(request.runtime, execution, str(request.tool_call.get("name")), str(request.tool_call.get("id")), payload, workflow, status or ("succeeded" if payload else "failed"))

    def _guard_final_response(self, state: AgentState, runtime=None) -> dict | None:
        messages = state.get("messages") or []
        if not messages or not isinstance(messages[-1], AIMessage):
            return None
        message = messages[-1]
        termination = message.additional_kwargs.get("nir_program_termination")
        if termination and isinstance(state.get("nir_workflow"), dict):
            workflow = state["nir_workflow"]
            stopped = {
                **workflow,
                "stage": "blocked",
                "next_action": "report_safety_termination",
                "stop_source": termination["stop_source"],
                "runtime_termination": {"source": termination["stop_source"], "reason": termination.get("reason")},
                "revision": workflow["revision"] + 1,
            }
            self._persist_question(state, runtime, stopped)
            return {"nir_workflow": stopped, "messages": [message.model_copy(update={"content": render_grounded_nir_response(stopped)})], "jump_to": "end"}
        workflow = state.get("nir_workflow")
        if not isinstance(workflow, dict):
            return None
        if waiting_for_user(workflow):
            return {"jump_to": "end"}
        termination = message.additional_kwargs.get("agent_termination") or {}
        if workflow.get("autonomous_delivery") and workflow.get("stage") not in {"blocked", "completed"} and not message.invalid_tool_calls:
            ask = next((call for call in message.tool_calls or termination.get("tool_calls", []) if call["name"] == "ask_clarification"), None)
            text = message_content_to_text(message.content)
            if termination.get("reason"):
                text = text.replace(termination["reason"], "").strip()
            question = str(ask["args"].get("question") or "").strip() if ask else question_for_user(text)
            if question:
                paused = begin_user_wait(workflow, messages, question, str(message.id), tool_call_id=ask["id"] if ask else None)
                if not validate_nir_response(waiting_message(paused), paused).passed:
                    paused["user_question"]["question"] = "请告诉我你希望下一步如何处理。"
                    paused["user_question"]["sanitized_display"] = True
                if ask and not validate_nir_response(question + "\n" + str(ask["args"].get("options") or []), paused).passed:
                    paused["user_question"]["sanitized_display"] = True
                self._persist_question(state, runtime, paused)
                additional = {key: value for key, value in message.additional_kwargs.items() if key != "tool_calls"}
                additional["nir_waiting_for_user"] = True
                replacement = message.model_copy(update={"content": "" if ask else waiting_message(paused), "tool_calls": [ask] if ask else [], "additional_kwargs": additional})
                return {"nir_workflow": paused, "messages": [replacement], **({} if ask else {"jump_to": "end"})}
        if termination and workflow.get("autonomous_delivery"):
            # A runtime stop is authoritative. Never send it through the
            # unfinished-workflow retry path or choose a model on its behalf.
            reason = str(termination["source"]) + "_limit"
            stopped = {**workflow, "stage": "blocked", "next_action": "report_safety_termination", "revision": workflow["revision"] + 1, "runtime_termination": termination, "stop_source": workflow.get("stop_source") or "tool_failure"}
            if workflow.get("search_session_id") and runtime is not None:
                from types import SimpleNamespace

                from langgraph.config import get_config

                from deerflow.community.nir.decision_budget import close_incomplete_workflow
                from deerflow.community.nir.search_tools import _service

                config = getattr(runtime, "config", None) or get_config()
                trusted = SimpleNamespace(state=state, context=getattr(runtime, "context", None), config=config)
                session = close_incomplete_workflow(_service(trusted), workflow["search_session_id"], invalid_json=False, reason=reason)
                stopped["stop_source"] = session.get("stop_source")
            self._persist_question(state, runtime, stopped)
            replacement = message.model_copy(update={"content": render_grounded_nir_response(stopped), "tool_calls": [], "additional_kwargs": {**message.additional_kwargs, "nir_runtime_stopped": True}})
            return {"nir_workflow": stopped, "messages": [replacement], "jump_to": "end"}
        if message.tool_calls:
            return None
        required_action = required_nir_workflow_action(workflow)
        if required_action:
            invalid_calls = message.invalid_tool_calls if workflow.get("decision_strategy") == "llm_autonomous" else []
            syntax_errors = []
            for call in invalid_calls:
                try:
                    json.loads(call.get("args") or "")
                except (ValueError, TypeError) as exc:
                    syntax_errors.append(f"{call.get('name') or 'tool'}: {str(exc)[:200]}")
                else:
                    syntax_errors.append("Invalid tool envelope; arguments must be a JSON object under proposal.")
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
                    )
                    + (
                        " Your attempted tool call was NOT executed because its JSON could not be parsed: " + "; ".join(syntax_errors[:3]) + ". Resubmit valid JSON to the current tool schema; do not narrate a promise to resubmit. "
                        "Use concise observations and comparisons, omit optional defaults, keep expected_outcome at proposal root, "
                        "and close all objects. The output cap is unchanged; keep scientific reasons brief. "
                        if syntax_errors
                        else ""
                    ),
                    additional_kwargs={
                        "hide_from_ui": True,
                        "nir_required_action": required_action,
                        **({"nir_invalid_tool_call": True} if invalid_calls else {}),
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
                    # Retain the raw provider output for audit and repair, but
                    # do not show an unfinished draft as a final chat answer.
                    "messages": [message.model_copy(update={"additional_kwargs": {**message.additional_kwargs, "hide_from_ui": True, "nir_unfinished_response": True}}), reminder],
                    "nir_workflow": guarded_workflow,
                    "jump_to": "model",
                }

            blocked_workflow = block_workflow_after_continuation_failure(
                guarded_workflow,
                required_action=required_action,
            )
            if workflow.get("autonomous_delivery") and workflow.get("search_session_id") and runtime is not None:
                from types import SimpleNamespace

                from langgraph.config import get_config

                from deerflow.community.nir.decision_budget import close_incomplete_workflow
                from deerflow.community.nir.search_tools import _service
                from deerflow.community.nir.web_entry import persist_unified_workflow

                trusted = SimpleNamespace(state=state, context=getattr(runtime, "context", None), config=get_config())
                session = close_incomplete_workflow(_service(trusted), workflow["search_session_id"], invalid_json=bool(invalid_calls))
                blocked_workflow.update(stop_source=session.get("stop_source"), continuation_failure={"required_action": required_action, "invalid_json": bool(invalid_calls), "reason": "workflow_continuation_limit"})
                persist_unified_workflow(trusted, blocked_workflow)
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

    @hook_config(can_jump_to=["model", "end"])
    @override
    def after_model(self, state: AgentState, runtime: Runtime) -> dict | None:  # noqa: ARG002
        return self._guard_final_response(state, runtime)

    @hook_config(can_jump_to=["model", "end"])
    @override
    async def aafter_model(self, state: AgentState, runtime: Runtime) -> dict | None:  # noqa: ARG002
        return self._guard_final_response(state, runtime)

    @staticmethod
    def _admission_estimate(request):
        return NIRWorkflowMiddleware._request_cost_profile(request)["input_utf8_bytes"]

    @staticmethod
    def _request_cost_profile(request):
        from deerflow.community.nir.decision_provider import request_cost_profile

        return request_cost_profile(request)
