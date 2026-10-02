"""Bounded Agent tools for the owner-scoped persistent NIR model library."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

from langchain.tools import tool
from langchain_core.messages import HumanMessage

from deerflow.config import get_app_config
from deerflow.config.paths import get_paths
from deerflow.persistence.engine import get_session_factory
from deerflow.runtime.user_context import resolve_runtime_user_id
from deerflow.tools.types import Runtime
from deerflow.utils.messages import get_original_user_content_text

from ._common import _err, _ok
from .datasets.sandbox_sync import sync_attachment_to_sandbox
from .models.service import ModelError, ModelService

_PROMOTION_AUTHORIZATION_PATTERNS = (
    r"(?:保存|持久化|加入).{0,16}模型(?:库)?",
    r"模型.{0,16}(?:长期保存|跨会话|以后使用)",
    r"(?:长期|跨会话).{0,16}保存.{0,16}模型",
    r"\bpromote.{0,24}model\b",
    r"\bsave.{0,24}model.{0,24}(?:library|later|future)\b",
)

_PROMOTION_NEGATION_PATTERN = re.compile(
    r"(?:不要|不再|别|取消|拒绝|禁止|无需|不需要).{0,24}(?:保存|持久化|加入|promote|save|模型库|跨会话)",
    flags=re.IGNORECASE,
)
_PROMOTION_POSITIVE_ACTION_PATTERN = re.compile(r"(?:保存|持久化|加入|promote|save)", flags=re.IGNORECASE)
_PROMOTION_MODEL_PATTERN = re.compile(r"(?:模型|model)", flags=re.IGNORECASE)
_PROMOTION_REUSE_SCOPE_PATTERN = re.compile(
    r"(?:模型库|跨会话|长期|以后(?:使用)?|library|later|future)",
    flags=re.IGNORECASE,
)


def _service() -> ModelService:
    config = get_app_config()
    if not config.nir_library.enabled:
        raise ModelError("feature_disabled", "NIR model library is disabled")
    session_factory = get_session_factory()
    if session_factory is None:
        raise ModelError("database_unavailable", "Persistent database is unavailable")
    return ModelService(session_factory, get_paths(), config.nir_library)


def _thread_id(runtime: Runtime) -> str | None:
    context = runtime.context if isinstance(runtime.context, dict) else {}
    value = context.get("thread_id")
    if value:
        return str(value)
    config = getattr(runtime, "config", None) or {}
    configurable = config.get("configurable", {}) if isinstance(config, dict) else {}
    value = configurable.get("thread_id") if isinstance(configurable, dict) else None
    return str(value) if value else None


def _run_id(runtime: Runtime) -> str | None:
    context = runtime.context if isinstance(runtime.context, dict) else {}
    value = context.get("run_id")
    return str(value) if value else None


def _latest_user_authorized_promotion(runtime: Runtime) -> bool:
    state = runtime.state if isinstance(runtime.state, Mapping) else {}
    messages = state.get("messages")
    if not isinstance(messages, list):
        return False
    for message in reversed(messages):
        if not isinstance(message, HumanMessage) or message.additional_kwargs.get("hide_from_ui"):
            continue
        text = get_original_user_content_text(message.content, message.additional_kwargs)
        if _PROMOTION_NEGATION_PATTERN.search(text):
            return False
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in _PROMOTION_AUTHORIZATION_PATTERNS):
            return True
        return bool(_PROMOTION_POSITIVE_ACTION_PATTERN.search(text) and _PROMOTION_MODEL_PATTERN.search(text) and _PROMOTION_REUSE_SCOPE_PATTERN.search(text))
    return False


def _tool_error(exc: ModelError) -> str:
    return _err(str(exc), code=exc.code)


@tool("nir_model_list", parse_docstring=True)
async def nir_model_list_tool(runtime: Runtime, include_archived: bool = False) -> str:
    """List reusable model versions owned by the current user.

    Args:
        include_archived: Include archived model metadata when true.
    """
    try:
        rows = await _service().list_models(resolve_runtime_user_id(runtime), include_archived=include_archived)
        return _ok({"status": "ok", "models": rows, "count": len(rows)})
    except ModelError as exc:
        return _tool_error(exc)


@tool("nir_model_get", parse_docstring=True)
async def nir_model_get_tool(runtime: Runtime, model_id: str, version: str | None = None) -> str:
    """Get bounded metadata for one owned model version.

    Args:
        model_id: Logical model ID returned by nir_model_list.
        version: Exact version; omit to get the latest non-deleted version.
    """
    try:
        row = await _service().get_model(resolve_runtime_user_id(runtime), model_id, version)
        return _ok({"status": "ok", "model": row})
    except ModelError as exc:
        return _tool_error(exc)


@tool("nir_model_promote", parse_docstring=True)
async def nir_model_promote_tool(
    runtime: Runtime,
    model_id: str,
    version: str,
    idempotency_key: str | None = None,
) -> str:
    """Persist the current thread's approved registered model for later reuse.

    Args:
        model_id: Exact model ID used by nir_register_model.
        version: Exact version returned by nir_register_model.
        idempotency_key: Optional stable retry key.
    """
    if not _latest_user_authorized_promotion(runtime):
        return _err(
            "Persistent cross-session model storage requires an explicit request in the latest user message.",
            code="confirmation_required",
        )
    thread_id = _thread_id(runtime)
    if not thread_id:
        return _err("Current thread identity is unavailable.", code="thread_not_owned")
    state = runtime.state if isinstance(runtime.state, Mapping) else {}
    workflow = state.get("nir_workflow")
    if not isinstance(workflow, Mapping):
        return _err("Registered NIR workflow state is unavailable.", code="registration_required")
    try:
        result = await _service().promote_registered(
            resolve_runtime_user_id(runtime),
            thread_id,
            model_id,
            version,
            workflow=workflow,
            source_run_id=_run_id(runtime),
            idempotency_key=idempotency_key,
        )
        library_status = result.pop("status", None)
        return _ok({**result, "library_status": library_status, "status": "ok"})
    except ModelError as exc:
        return _tool_error(exc)


@tool("nir_model_attach", parse_docstring=True)
async def nir_model_attach_tool(runtime: Runtime, model_id: str, version: str) -> str:
    """Attach one verified saved model to the current thread for prediction.

    Args:
        model_id: Logical model ID returned by nir_model_list.
        version: Exact ready model version to attach.
    """
    thread_id = _thread_id(runtime)
    if not thread_id:
        return _err("Current thread identity is unavailable.", code="thread_not_owned")
    owner = resolve_runtime_user_id(runtime)

    async def sync(virtual_path: str, local_path: Path) -> None:
        await sync_attachment_to_sandbox(owner, thread_id, virtual_path, local_path)

    try:
        result = await _service().attach_to_thread(
            owner,
            model_id,
            version,
            thread_id,
            sync_callback=sync,
        )
        attachment_status = result.pop("status", None)
        return _ok(
            {
                **result,
                "attachment_status": attachment_status,
                "status": "ok",
                "next_action": "Start nir_workflow(task_type='prediction', model_path from this result), inspect the new input data, then call nir_predict with this exact model_path.",
            }
        )
    except ModelError as exc:
        return _tool_error(exc)
