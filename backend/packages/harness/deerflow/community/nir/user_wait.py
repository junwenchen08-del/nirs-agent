"""A durable user-question boundary, separate from model/experiment decisions."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from datetime import UTC, datetime

from langchain_core.messages import HumanMessage

from deerflow.utils.messages import get_original_user_content_text

_QUESTION = re.compile(
    r"(?:你|您)(?:是|更)?(?:希望|想让|想要|想|要我|选择|倾向|决定|愿意|是否)[^。！？?\n]{0,600}[？?]"
    r"|请(?:你|您)?(?:先)?(?:确认|选择|决定|告诉我|回复|回答|告知|提供|补充|输入)[^。！？?\n]{0,600}"
    r"|(?:等待|等到|等|收到)(?:你|您)的(?:回复|回答|确认|选择)[^。！？?\n]{0,600}"
    r"|(?:是否需要我|要不要我|需不需要我|需要我)[^。！？?\n]{0,600}[？?]"
    r"|(?:would\s+you\s+(?:like|prefer)|do\s+you\s+want\s+me\s+to|shall\s+i|which\s+(?:option\s+)?do\s+you\s+prefer)[^.?!\n]{0,600}\?",
    # Explicit answer requests need no question mark, and often end in a
    # colon before numbered choices. Never treat that punctuation as consent.
    re.IGNORECASE,
)


def question_for_user(text: str) -> str | None:
    text = re.sub(r"```[^`]*```", "", text, flags=re.S)
    english = re.search(r"\bplease\s+(?:reply|respond|choose|confirm|select|tell\s+me)\b[^\n]{0,600}", text, re.I)
    for match in _QUESTION.finditer(text):
        if re.search(r"如果\s*$|若\s*$|\bif\s*$", text[max(0, match.start() - 8) : match.start()], re.I):
            continue
        # Preserve the options and answer instructions below the request.
        return text[match.start() :].strip()[:4000]
    return text[english.start() :].strip()[:4000] if english else None


def waiting_for_user(workflow) -> bool:
    pending = workflow.get("user_question") if isinstance(workflow, Mapping) else None
    return isinstance(pending, Mapping) and pending.get("status") == "waiting"


def latest_user_message(messages):
    """Return the latest genuine user reply, excluding injected runtime context."""
    for message in reversed(messages):
        if not isinstance(message, HumanMessage):
            continue
        flags = message.additional_kwargs or {}
        if any(flags.get(key) for key in ("hide_from_ui", "slash_skill_activation", "durable_context_data", "dynamic_context_reminder")):
            continue
        if message.name in {"summary", "memory", "nir_workflow_completion_reminder"}:
            continue
        text = get_original_user_content_text(message.content, flags).strip()
        if not text or text.startswith(("<memory>", "<current_sealed_workflow>")):
            continue
        return message
    return None


def _latest_user(messages):
    message = latest_user_message(messages)
    if message is None:
        return None, None
    text = get_original_user_content_text(message.content, message.additional_kwargs).strip()
    return message, hashlib.sha256(text.encode()).hexdigest()


def begin_user_wait(workflow, messages, question, message_id, *, tool_call_id=None):
    user, digest = _latest_user(messages)
    revision = workflow.get("revision", 0) + 1
    return {
        **workflow,
        "revision": revision,
        "user_question": {
            "status": "waiting",
            "question": question[:4000],
            "message_id": message_id,
            "tool_call_id": tool_call_id,
            "user_message_id": user.id if user else None,
            "user_message_sha256": digest,
            "asked_at": datetime.now(UTC).isoformat(),
        },
        "history": [*(workflow.get("history") or []), {"action": "await_user_reply", "message_id": message_id, "revision": revision}][-50:],
    }


def resume_user_wait(workflow, messages):
    if not waiting_for_user(workflow):
        return None
    user, digest = _latest_user(messages)
    pending = workflow["user_question"]
    if user is None:
        return None
    if user.id and pending.get("user_message_id"):
        if user.id == pending["user_message_id"]:
            return None
    elif digest == pending.get("user_message_sha256"):
        return None
    revision = workflow.get("revision", 0) + 1
    return {
        **workflow,
        "revision": revision,
        "user_question": {**pending, "status": "reply_received", "reply_message_id": user.id, "replied_at": datetime.now(UTC).isoformat()},
        "history": [*(workflow.get("history") or []), {"action": "user_reply_received", "reply_message_id": user.id, "revision": revision}][-50:],
    }


def waiting_message(workflow) -> str:
    return "需要你确认下一步。收到你的回复后，我再继续。\n\n" + workflow["user_question"]["question"]
