"""Runtime enforcement and automatic advancement for NIR workflows."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from deerflow.agents.thread_state import NIRWorkflowState
from deerflow.community.nir.workflow import NIRWorkflowError, record_tool_observation, transition_workflow
from deerflow.trace_context import DEERFLOW_TRACE_METADATA_KEY
from deerflow.utils.messages import get_original_user_content_text

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
    "nir_reflect": _ToolPolicy(frozenset({"knowledge", "review"}), _MODEL_TASKS),
    "nir_search_knowledge": _ToolPolicy(frozenset({"execution", "knowledge"}), frozenset({"knowledge", *_MODEL_TASKS})),
    "nir_register_model": _ToolPolicy(frozenset({"approved"}), _MODEL_TASKS),
}
_OBSERVED_NIR_TOOLS = frozenset({*_TOOL_POLICIES, "nir_workflow"})

_MODELING_TOOLS = frozenset({"nir_train_auto_split_model", "nir_train_model", "nir_train_partitioned_model", "nir_train_multi_model", "nir_analyze", "nir_analyze_collection", "nir_compare"})
_MODEL_DEFAULTS = {
    "nir_train_auto_split_model": "auto",
    "nir_train_model": "pls",
    "nir_train_partitioned_model": "auto",
    "nir_train_multi_model": "pls",
    "nir_analyze": "auto",
    "nir_analyze_collection": "auto",
    "nir_compare": "pls",
}
_MODEL_SUBSTITUTION_CODE = "nir_model_substitution_requires_approval"
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
    raw = args.get("method", _MODEL_DEFAULTS[tool_name])
    method = str(raw or _MODEL_DEFAULTS[tool_name]).strip().lower()
    return "cnn" if method in {"1d-cnn", "1d_cnn"} else method


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


def _authorize(request: ToolCallRequest) -> ToolMessage | None:
    tool_name = str(request.tool_call.get("name", ""))
    workflow = _state_from_request(request).get("nir_workflow")
    if isinstance(workflow, Mapping) and str(workflow.get("stage", "")) not in _TERMINAL_NIR_STAGES:
        args = request.tool_call.get("args")
        args = args if isinstance(args, Mapping) else {}
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


def _model_attempt_update(
    tool_name: str,
    workflow: NIRWorkflowState,
    payload: Mapping[str, Any],
) -> NIRWorkflowState | None:
    result = payload.get("best") if tool_name == "nir_compare" else payload
    if not isinstance(result, Mapping) or not isinstance(result.get("passed"), bool):
        return None

    model_path = payload.get("model_path") or payload.get("model")
    metrics_path = payload.get("metrics_path") or payload.get("metrics")
    if tool_name == "nir_compare":
        metrics_path = payload.get("all_metrics")
    return transition_workflow(
        workflow,
        action="record_attempt",
        attempt_passed=result["passed"],
        grade=str(result.get("grade")) if result.get("grade") is not None else None,
        model_path=str(model_path) if model_path else None,
        metrics_path=str(metrics_path) if metrics_path else None,
        notes=f"Automatically recorded successful {tool_name} result.",
    )


def _next_workflow(
    tool_name: str,
    workflow: NIRWorkflowState,
    payload: Mapping[str, Any],
) -> NIRWorkflowState | None:
    if tool_name in _MODELING_TOOLS:
        return _model_attempt_update(tool_name, workflow, payload)
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
        updated = _next_workflow(str(request.tool_call.get("name", "")), workflow, payload)
    except NIRWorkflowError:
        logger.exception("Automatic NIR workflow transition failed")
        return result
    return _attach_workflow_update(result, updated) if updated is not None else result


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
        return _record_observation(request, _advance(request, handler(request)))

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        denied = _authorize(request)
        if denied is not None:
            return _record_observation(request, denied)
        return _record_observation(request, _advance(request, await handler(request)))
