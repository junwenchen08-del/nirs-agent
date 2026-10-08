"""Persist exactly the model just registered by the approved agent workflow."""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping

from langchain_core.messages import HumanMessage

from deerflow.config import get_app_config
from deerflow.runtime.user_context import resolve_runtime_user_id
from deerflow.tools.types import Runtime
from deerflow.utils.messages import get_original_user_content_text

logger = logging.getLogger(__name__)

_DECLINED_STORAGE = re.compile(
    r"(?:不要|不再|别|取消|拒绝|禁止|无需|不需要)\s*(?:自动|长期|再)?\s*(?:持久化|保存|入库|加入模型库)"
    r"|\b(?:do not|don't|never|no need to)\s+(?:automatically\s+)?(?:save|persist|promote|store)\b.{0,80}\b(?:model|library)\b",
    flags=re.IGNORECASE,
)


def _storage_declined(runtime: Runtime) -> bool:
    state = runtime.state if isinstance(runtime.state, Mapping) else {}
    for message in reversed(state.get("messages") or []):
        if isinstance(message, HumanMessage) and not message.additional_kwargs.get("hide_from_ui"):
            text = get_original_user_content_text(message.content, message.additional_kwargs)
            return bool(_DECLINED_STORAGE.search(text))
    return False


def failed_model_save(code: str) -> dict:
    messages = {
        "quota_exceeded": "模型库容量不足，请清理后重试保存。",
        "insufficient_free_space": "磁盘可用空间不足，请清理后重试保存。",
        "version_limit": "该模型的可用版本已达上限，请归档旧版本后重试。",
        "artifact_unverified": "模型文件或指标未通过完整性校验，未保存至模型库。",
        "idempotency_conflict": "模型库中已有同名版本对应其他模型，未覆盖原记录。",
    }
    return {"status": "failed", "error_code": code, "message": messages.get(code, "注册已完成，但模型库保存失败，请重试保存。")}


async def save_registered_model(runtime: Runtime, workflow: Mapping, registration: Mapping) -> dict:
    """Apply trusted configuration, then reuse full owner/integrity/lineage checks.

    This is an agent registration hook, not an intent-authorized general save
    tool. It cannot promote arbitrary uploads or unapproved modeling attempts.
    """
    from .model_tools import _run_id, _service, _thread_id
    from .models.service import ModelError

    try:
        policy = get_app_config().nir_library
        if not policy.enabled or not policy.auto_save_registered_models:
            return {"status": "disabled", "message": "未启用注册模型自动保存；本次仅完成对话内注册。"}
        if _storage_declined(runtime):
            return {"status": "skipped", "reason": "user_declined", "message": "按你的要求，仅注册模型，未保存至模型库。"}
        if workflow.get("stage") != "registered" or workflow.get("approval_status") != "approved" or registration.get("status") != "registered":
            return failed_model_save("registration_required")
        thread_id = _thread_id(runtime)
        if not thread_id:
            return failed_model_save("thread_not_owned")
        model_id, version = registration.get("model_id"), registration.get("version")
        if not isinstance(model_id, str) or not isinstance(version, str) or not model_id or not version:
            return failed_model_save("invalid_identifier")
        saved = await _service().promote_registered(
            resolve_runtime_user_id(runtime),
            thread_id,
            model_id,
            version,
            workflow=workflow,
            source_run_id=_run_id(runtime),
        )
        if saved.get("status") != "ready":
            return failed_model_save("model_not_ready")
        return {
            "status": "reused" if saved.get("reused_existing") else "saved",
            "model_id": saved["model_id"],
            "version": saved["version"],
            "message": "已保存至模型库，可在左侧模型库查看并跨会话使用。",
        }
    except ModelError as exc:
        return failed_model_save(exc.code)
    except Exception:
        logger.exception("Registered model could not be saved to the persistent library")
        return failed_model_save("storage_error")
