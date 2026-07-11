"""Middleware for automatic thread title generation."""

import logging
import re
from typing import TYPE_CHECKING, Any, NotRequired, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langgraph.config import get_config
from langgraph.constants import TAG_NOSTREAM
from langgraph.runtime import Runtime

from deerflow.agents.middlewares.dynamic_context_middleware import is_dynamic_context_reminder
from deerflow.config.title_config import get_title_config
from deerflow.models import create_chat_model
from deerflow.utils.llm_text import strip_think_blocks

if TYPE_CHECKING:
    from deerflow.config.app_config import AppConfig
    from deerflow.config.title_config import TitleConfig

logger = logging.getLogger(__name__)

# Tag names injected by upstream middlewares that must be stripped from user
# text before it is used for title generation. Kept in sync with
# input_sanitization_middleware._BLOCKED_TAG_NAMES and the tag list in
# frontend/src/core/messages/utils.ts.
_INTERNAL_TAG_NAMES = (
    "uploaded_files",
    "slash_skill_activation",
    "system-reminder",
    "memory",
    "current_date",
    "think",
    "analysis",
)
# Pattern: <tag ...> ... </tag>  (non-greedy, multiline, case-insensitive).
# Group 1 captures the opening tag name so the closing tag is matched via
# backreference, ensuring only paired open/close tags are removed.
_INTERNAL_TAG_RE = re.compile(
    r"<\s*(" + "|".join(_INTERNAL_TAG_NAMES) + r")\b[^>]*>[\s\S]*?<\s*/\s*\1\s*>",
    re.IGNORECASE,
)
# Orphan tags that survive the paired-tag regex: self-closing <tag/>, a
# dangling opening <tag> whose close was truncated at context-window
# limits, or an orphan closing </tag>. Requires a closing ">" so a
# truncated "<tag" with no ">" (very rare, mid-stream truncation) is
# left untouched rather than eating arbitrary trailing content.
_INTERNAL_TAG_ORPHAN_RE = re.compile(
    r"<\s*/?\s*(" + "|".join(_INTERNAL_TAG_NAMES) + r")\b[^>]*>",
    re.IGNORECASE,
)


class TitleMiddlewareState(AgentState):
    """Compatible with the `ThreadState` schema."""

    title: NotRequired[str | None]


class TitleMiddleware(AgentMiddleware[TitleMiddlewareState]):
    """Automatically generate a title for the thread after the first user message."""

    state_schema = TitleMiddlewareState

    def __init__(self, *, app_config: "AppConfig | None" = None, title_config: "TitleConfig | None" = None):
        super().__init__()
        self._app_config = app_config
        self._title_config = title_config

    def _get_title_config(self):
        if self._title_config is not None:
            return self._title_config
        if self._app_config is not None:
            return self._app_config.title
        return get_title_config()

    def _normalize_content(self, content: object) -> str:
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts = [self._normalize_content(item) for item in content]
            return "\n".join(part for part in parts if part)

        if isinstance(content, dict):
            text_value = content.get("text")
            if isinstance(text_value, str):
                return text_value

            nested_content = content.get("content")
            if nested_content is not None:
                return self._normalize_content(nested_content)

        return ""

    @staticmethod
    def _message_type(message: object) -> str | None:
        message_type = getattr(message, "type", None)
        if message_type is None and isinstance(message, dict):
            message_type = message.get("type") or message.get("role")
        if message_type == "user":
            return "human"
        if message_type == "assistant":
            return "ai"
        return message_type if isinstance(message_type, str) else None

    @staticmethod
    def _message_content(message: object) -> object:
        if isinstance(message, dict):
            return message.get("content", "")
        return getattr(message, "content", "")

    @staticmethod
    def _is_dynamic_context_reminder_message(message: object) -> bool:
        if is_dynamic_context_reminder(message):
            return True
        if isinstance(message, dict):
            additional_kwargs = message.get("additional_kwargs")
            return isinstance(additional_kwargs, dict) and bool(additional_kwargs.get("dynamic_context_reminder"))
        return False

    @staticmethod
    def _is_user_message_for_title(message: object) -> bool:
        return TitleMiddleware._message_type(message) == "human" and not TitleMiddleware._is_dynamic_context_reminder_message(message)

    def _get_title_user_message(self, state: TitleMiddlewareState) -> str:
        messages = state.get("messages") or []
        for m in messages:
            if not self._is_user_message_for_title(m):
                continue
            cleaned = self._strip_internal_tags(self._normalize_content(self._message_content(m)))
            if cleaned:
                return cleaned
            # User submitted a message with only an attachment (no text). Use
            # the filename(s) from additional_kwargs.files as the title seed
            # so the sidebar shows something more useful than "New Conversation".
            filenames = self._get_uploaded_filenames(m)
            if filenames:
                return ", ".join(filenames)
        return ""

    @staticmethod
    def _get_uploaded_filenames(message: object) -> list[str]:
        """Extract uploaded filenames from a message's additional_kwargs.files.

        Tolerates both dict-shape and object-shape messages, and skips entries
        that do not look like files (no ``name`` field).
        """
        if isinstance(message, dict):
            additional = message.get("additional_kwargs") or {}
        else:
            additional = getattr(message, "additional_kwargs", None) or {}
        if not isinstance(additional, dict):
            return []
        files = additional.get("files")
        if not isinstance(files, list):
            return []
        names: list[str] = []
        for f in files:
            if isinstance(f, dict):
                name = f.get("name") or f.get("filename")
            else:
                name = getattr(f, "name", None) or getattr(f, "filename", None)
            if isinstance(name, str) and name.strip():
                names.append(name.strip())
        return names

    @staticmethod
    def _strip_internal_tags(text: str) -> str:
        """Remove framework-injected context blocks from user text before using it for title generation.

        Upstream middlewares prepend XML-like blocks to HumanMessage content
        (uploaded_files, slash_skill_activation, system-reminder, etc.). If we
        pass that raw text to the title model or the local fallback, the
        resulting thread title becomes a truncated tag prefix and is useless
        in the sidebar.
        """
        if not text:
            return ""
        cleaned = _INTERNAL_TAG_RE.sub("", text)
        # Second pass: strip any remaining opening/closing/self-closing
        # tags from the known list that survived the paired-tag regex
        # (self-closing <tag/>, or a dangling <tag> whose close was
        # truncated upstream).
        cleaned = _INTERNAL_TAG_ORPHAN_RE.sub("", cleaned)
        # Collapse leftover blank lines and trim.
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()

    def _should_generate_title(self, state: TitleMiddlewareState, *, allow_partial_exchange: bool = False) -> bool:
        """Check if we should generate a title for this thread."""
        config = self._get_title_config()
        if not config.enabled:
            return False

        # Check if thread already has a title in state
        if state.get("title"):
            return False

        # Check if this is the first turn (has at least one user message and one assistant response).
        # Defensively coerce a None ``messages`` channel (possible when reading a
        # partially-initialized checkpoint) into an empty list so ``len()`` is safe.
        messages = state.get("messages") or []
        min_messages = 1 if allow_partial_exchange else 2
        if len(messages) < min_messages:
            return False

        # Count user and assistant messages
        user_messages = [m for m in messages if self._is_user_message_for_title(m)]
        assistant_messages = [m for m in messages if self._message_type(m) == "ai"]

        # Normal path: title only after first complete exchange. Interrupted path
        # (``allow_partial_exchange=True``) accepts a lone first-turn user message
        # so a fallback title can still be persisted when the run is cancelled
        # before any AI chunk reaches the checkpoint.
        return len(user_messages) == 1 and (len(assistant_messages) >= 1 or allow_partial_exchange)

    def _build_title_prompt(self, state: TitleMiddlewareState) -> tuple[str, str]:
        """Extract user/assistant messages and build the title prompt.

        Returns (prompt_string, user_msg) so callers can use user_msg as fallback.
        """
        config = self._get_title_config()
        messages = state.get("messages") or []

        assistant_msg_content = next((self._message_content(m) for m in messages if self._message_type(m) == "ai"), "")

        user_msg = self._get_title_user_message(state)
        assistant_msg = strip_think_blocks(self._normalize_content(assistant_msg_content))

        prompt = config.prompt_template.format(
            max_words=config.max_words,
            user_msg=user_msg[:500],
            assistant_msg=assistant_msg[:500],
        )
        return prompt, user_msg

    def _parse_title(self, content: object) -> str:
        """Normalize model output into a clean title string."""
        config = self._get_title_config()
        title_content = self._normalize_content(content)
        title_content = strip_think_blocks(title_content)
        title = title_content.strip().strip('"').strip("'")
        return title[: config.max_chars] if len(title) > config.max_chars else title

    def _fallback_title(self, user_msg: str) -> str:
        config = self._get_title_config()
        fallback_chars = min(config.max_chars, 50)
        if len(user_msg) > fallback_chars:
            return user_msg[:fallback_chars].rstrip() + "..."
        return user_msg if user_msg else "New Conversation"

    def _get_runnable_config(self) -> dict[str, Any]:
        """Inherit the parent RunnableConfig and add middleware tag.

        This ensures RunJournal identifies LLM calls from this middleware
        as ``middleware:title`` instead of ``lead_agent``.
        """
        try:
            parent = get_config()
        except Exception:
            parent = {}
        config = {**parent}
        config["run_name"] = "title_agent"
        config["tags"] = [
            *(config.get("tags") or []),
            "middleware:title",
            TAG_NOSTREAM,
        ]
        return config

    def _generate_title_result(self, state: TitleMiddlewareState, *, allow_partial_exchange: bool = False) -> dict | None:
        """Generate a local fallback title without blocking on an LLM call."""
        if not self._should_generate_title(state, allow_partial_exchange=allow_partial_exchange):
            return None

        user_msg = self._get_title_user_message(state)
        return {"title": self._fallback_title(user_msg)}

    async def _agenerate_title_result(self, state: TitleMiddlewareState) -> dict | None:
        """Generate a configured LLM title asynchronously and fall back locally."""
        if not self._should_generate_title(state):
            return None

        config = self._get_title_config()
        if not config.model_name:
            user_msg = self._get_title_user_message(state)
            return {"title": self._fallback_title(user_msg)}

        user_msg = self._get_title_user_message(state)

        try:
            prompt, user_msg = self._build_title_prompt(state)
            # attach_tracing=False because ``_get_runnable_config()`` inherits
            # the graph-level RunnableConfig (set in ``_make_lead_agent``) whose
            # callbacks already carry tracing handlers; binding them again at
            # the model level would emit duplicate spans.
            model_kwargs = {"thinking_enabled": False, "attach_tracing": False}
            if self._app_config is not None:
                model_kwargs["app_config"] = self._app_config
            model = create_chat_model(name=config.model_name, **model_kwargs)
            response = await model.ainvoke(prompt, config=self._get_runnable_config())
            title = self._parse_title(response.content)
            if title:
                return {"title": title}
        except Exception:
            logger.debug("Failed to generate async title; falling back to local title", exc_info=True)
        return {"title": self._fallback_title(user_msg)}

    @override
    def after_model(self, state: TitleMiddlewareState, runtime: Runtime) -> dict | None:
        return self._generate_title_result(state)

    @override
    async def aafter_model(self, state: TitleMiddlewareState, runtime: Runtime) -> dict | None:
        return await self._agenerate_title_result(state)
