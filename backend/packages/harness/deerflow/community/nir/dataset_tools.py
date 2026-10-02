"""Bounded Agent tools for the opt-in, owner-scoped NIR dataset library."""

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
from .datasets.service import DatasetError, DatasetService

_SAVE_AUTHORIZATION_PATTERNS = (
    r"保存.{0,12}(数据集库|以后|长期|跨会话)",
    r"(长期|跨会话).{0,12}保存",
    r"save.{0,24}(dataset library|for later|across chats)",
    r"keep.{0,24}(dataset|file).{0,24}(later|future)",
)


def _service() -> DatasetService:
    config = get_app_config()
    if not config.nir_library.enabled:
        raise DatasetError("feature_disabled", "NIR dataset library is disabled")
    session_factory = get_session_factory()
    if session_factory is None:
        raise DatasetError("database_unavailable", "Persistent database is unavailable")
    return DatasetService(session_factory, get_paths(), config.nir_library)


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


def _latest_user_authorized_save(runtime: Runtime) -> bool:
    state = runtime.state if isinstance(runtime.state, Mapping) else {}
    messages = state.get("messages")
    if not isinstance(messages, list):
        return False
    for message in reversed(messages):
        if not isinstance(message, HumanMessage) or message.additional_kwargs.get("hide_from_ui"):
            continue
        text = get_original_user_content_text(message.content, message.additional_kwargs)
        return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in _SAVE_AUTHORIZATION_PATTERNS)
    return False


def _tool_error(exc: DatasetError) -> str:
    return _err(str(exc), code=exc.code)


@tool("nir_dataset_list", parse_docstring=True)
async def nir_dataset_list_tool(runtime: Runtime, include_archived: bool = False) -> str:
    """List the current user's saved NIR datasets without returning raw spectra.

    Args:
        include_archived: Include archived metadata entries when true.
    """
    try:
        rows = await _service().list_datasets(resolve_runtime_user_id(runtime))
        if not include_archived:
            rows = [row for row in rows if row.get("status") == "ready"]
        return _ok({"status": "ok", "datasets": rows, "count": len(rows)})
    except DatasetError as exc:
        return _tool_error(exc)


@tool("nir_dataset_get", parse_docstring=True)
async def nir_dataset_get_tool(runtime: Runtime, dataset_id: str) -> str:
    """Get bounded metadata and Profile summaries for one owned dataset.

    Args:
        dataset_id: Exact dataset ID returned by nir_dataset_list.
    """
    try:
        service = _service()
        owner = resolve_runtime_user_id(runtime)
        dataset = await service.get_dataset(owner, dataset_id)
        profiles = await service.list_profiles(owner, dataset_id)
        summaries = [{key: value for key, value in profile.items() if key != "mapping"} for profile in profiles]
        return _ok({"status": "ok", "dataset": dataset, "profiles": summaries})
    except DatasetError as exc:
        return _tool_error(exc)


@tool("nir_dataset_history", parse_docstring=True)
async def nir_dataset_history_tool(runtime: Runtime, dataset_id: str) -> str:
    """List successful attachments for one owned dataset.

    Args:
        dataset_id: Exact dataset ID returned by nir_dataset_list.
    """
    try:
        uses = await _service().list_uses(resolve_runtime_user_id(runtime), dataset_id)
        return _ok({"status": "ok", "dataset_id": dataset_id, "uses": uses, "count": len(uses)})
    except DatasetError as exc:
        return _tool_error(exc)


@tool("nir_dataset_save", parse_docstring=True)
async def nir_dataset_save_tool(runtime: Runtime, virtual_path: str, name: str) -> str:
    """Save a current-thread CSV/TXT/MAT upload only after the user explicitly asks.

    Args:
        virtual_path: Exact /mnt/user-data/uploads filename from the current thread.
        name: Human-readable dataset name.
    """
    if not _latest_user_authorized_save(runtime):
        return _err(
            "Saving for cross-session reuse requires an explicit request in the latest user message.",
            code="confirmation_required",
        )
    thread_id = _thread_id(runtime)
    if not thread_id:
        return _err("Current thread identity is unavailable.", code="thread_not_owned")
    try:
        result = await _service().save_from_thread_upload(
            resolve_runtime_user_id(runtime),
            thread_id,
            virtual_path,
            name,
            save_confirmed=True,
        )
        dataset_status = result.pop("status", None)
        return _ok({**result, "dataset_status": dataset_status, "status": "ok"})
    except DatasetError as exc:
        return _tool_error(exc)


@tool("nir_dataset_attach", parse_docstring=True)
async def nir_dataset_attach_tool(
    runtime: Runtime,
    dataset_id: str,
    profile_id: str | None = None,
    desired_filename: str | None = None,
    idempotency_key: str | None = None,
) -> str:
    """Attach a verified saved dataset to the current thread for fresh inspection.

    Args:
        dataset_id: Exact dataset ID returned by nir_dataset_list.
        profile_id: Optional confirmed Profile ID. Omit to attach raw bytes for re-inspection.
        desired_filename: Optional basename retaining the source extension.
        idempotency_key: Optional retry key; a stable dataset/profile key is used when omitted.
    """
    thread_id = _thread_id(runtime)
    if not thread_id:
        return _err("Current thread identity is unavailable.", code="thread_not_owned")
    owner = resolve_runtime_user_id(runtime)
    current_run_id = _run_id(runtime)
    key = idempotency_key or f"agent:{current_run_id or 'no-run'}:{dataset_id}:{profile_id or 'raw'}"

    async def sync(virtual_path: str, local_path: Path) -> None:
        await sync_attachment_to_sandbox(owner, thread_id, virtual_path, local_path)

    try:
        result = await _service().attach_to_thread(
            owner,
            dataset_id,
            thread_id,
            profile_id=profile_id,
            idempotency_key=key,
            desired_filename=desired_filename,
            run_id=current_run_id,
            prepare_workflow=True,
            sync_callback=sync,
        )
        attachment_status = result.pop("status", None)
        return _ok(
            {
                **result,
                "attachment_status": attachment_status,
                "status": "ok",
                "next_action": ("Start nir_workflow with project_id, dataset/profile/hash/attachment lineage from this result, then call nir_inspect on virtual_path; prior inspection or training conclusions are not inherited."),
            }
        )
    except DatasetError as exc:
        return _tool_error(exc)
