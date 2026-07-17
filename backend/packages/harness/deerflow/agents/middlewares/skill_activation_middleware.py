"""Middleware for explicit slash and deterministic routed skill activation."""

from __future__ import annotations

import asyncio
import hashlib
import html
import logging
import re
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, override

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, HumanMessage

from deerflow.runtime.secret_context import ACTIVE_SECRETS_CONTEXT_KEY, extract_request_secrets
from deerflow.skills.slash import parse_slash_skill_reference, resolve_slash_skill
from deerflow.skills.storage import get_or_new_skill_storage
from deerflow.skills.storage.skill_storage import SkillStorage
from deerflow.skills.types import SKILL_MD_FILE, SecretRequirement
from deerflow.utils.messages import get_original_user_content_text

if TYPE_CHECKING:
    from deerflow.config.app_config import AppConfig

logger = logging.getLogger(__name__)

_SLASH_SKILL_ACTIVATION_KEY = "slash_skill_activation"
_SLASH_SKILL_ACTIVATION_TARGET_ID_KEY = "slash_skill_activation_target_id"
_SUMMARY_MESSAGE_NAME = "summary"


@dataclass(frozen=True, slots=True)
class _Activation:
    skill_name: str
    category: str
    container_file_path: str
    skill_content: str
    content_hash: str
    remaining_text: str
    mode: str = "explicit"
    required_secrets: tuple[SecretRequirement, ...] = ()


@dataclass(frozen=True, slots=True)
class _ActivationResolution:
    activation: _Activation | None = None
    failure_message: str | None = None


@dataclass(frozen=True, slots=True)
class SkillAutoRoute:
    """Deterministic text patterns that activate one installed skill."""

    skill_name: str
    patterns: tuple[str, ...]
    state_key: str | None = None
    state_status_field: str = "stage"
    terminal_statuses: tuple[str, ...] = ()

    def matches(self, text: str, state: dict | None = None) -> bool:
        if self.state_key and state:
            route_state = state.get(self.state_key)
            if route_state:
                status = route_state.get(self.state_status_field) if isinstance(route_state, dict) else None
                if status not in self.terminal_statuses:
                    return True
        return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in self.patterns)


def is_slash_skill_activation_reminder(message: object) -> bool:
    """Return whether a message is hidden slash-skill activation context."""
    return isinstance(message, HumanMessage) and bool(message.additional_kwargs.get(_SLASH_SKILL_ACTIVATION_KEY))


def _is_user_activation_target(message: object) -> bool:
    if not isinstance(message, HumanMessage):
        return False
    if message.name == _SUMMARY_MESSAGE_NAME:
        return False
    if message.additional_kwargs.get("hide_from_ui"):
        return False
    return True


class SkillActivationMiddleware(AgentMiddleware):
    """Inject full SKILL.md content for explicit or deterministic activation."""

    def __init__(
        self,
        *,
        available_skills: set[str] | None = None,
        app_config: AppConfig | None = None,
        auto_routes: tuple[SkillAutoRoute, ...] = (),
    ) -> None:
        super().__init__()
        self._available_skills = set(available_skills) if available_skills is not None else None
        self._app_config = app_config
        self._auto_routes = auto_routes

    def _storage(self) -> SkillStorage:
        if self._app_config is not None:
            return get_or_new_skill_storage(app_config=self._app_config)
        return get_or_new_skill_storage()

    @staticmethod
    def _read_skill_content(skill_file: Path, skills_root: Path) -> str:
        if skill_file.name != SKILL_MD_FILE:
            raise ValueError(f"Expected {SKILL_MD_FILE}, got {skill_file.name}")
        resolved_root = skills_root.resolve()
        resolved_file = skill_file.resolve()
        try:
            resolved_file.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError("Resolved skill file must stay within the configured skills root.") from exc
        if not resolved_file.is_file():
            raise FileNotFoundError(resolved_file)
        return resolved_file.read_text(encoding="utf-8")

    def _resolve_activation(self, text: str, state: dict | None = None) -> _ActivationResolution | None:
        reference = parse_slash_skill_reference(text)
        mode = "explicit"
        if reference is None:
            route = next(
                (candidate for candidate in self._auto_routes if candidate.matches(text, state)),
                None,
            )
            if route is None:
                return None
            skill_name = route.skill_name
            remaining_text = text
            mode = "automatic"
        else:
            skill_name = reference.name
            remaining_text = reference.remaining_text

        storage = self._storage()
        skills = storage.load_skills(enabled_only=False)
        skill = next((candidate for candidate in skills if candidate.name == skill_name), None)
        if skill is None:
            if mode == "automatic":
                return None
            return _ActivationResolution(failure_message=f"Skill `/{skill_name}` is not installed.")
        if not skill.enabled:
            if mode == "automatic":
                return None
            return _ActivationResolution(failure_message=f"Skill `/{skill_name}` is installed but disabled. Enable it before using slash activation.")
        if self._available_skills is not None and skill_name not in self._available_skills:
            if mode == "automatic":
                return None
            return _ActivationResolution(failure_message=f"Skill `/{skill_name}` is not available for this agent.")

        if mode == "explicit":
            resolved = resolve_slash_skill(
                text,
                skills,
                available_skills=self._available_skills,
                container_base_path=storage.get_container_root(),
            )
            if resolved is None:
                return _ActivationResolution(failure_message=f"Skill `/{skill_name}` could not be resolved.")
            container_file_path = resolved.container_file_path
            remaining_text = resolved.remaining_text
        else:
            container_file_path = skill.get_container_file_path(storage.get_container_root())

        try:
            skill_content = self._read_skill_content(skill.skill_file, storage.get_skills_root_path())
        except (OSError, ValueError):
            logger.exception("Failed to read activated skill %s", skill.name)
            if mode == "automatic":
                return None
            return _ActivationResolution(failure_message=f"Skill `/{skill_name}` could not be loaded safely. Please check the skill installation.")

        content_hash = hashlib.sha256(skill_content.encode("utf-8")).hexdigest()
        return _ActivationResolution(
            activation=_Activation(
                skill_name=skill.name,
                category=str(skill.category),
                container_file_path=container_file_path,
                skill_content=skill_content,
                content_hash=content_hash,
                remaining_text=remaining_text,
                mode=mode,
                required_secrets=tuple(skill.required_secrets or ()),
            )
        )

    @staticmethod
    def _build_activation_reminder(activation: _Activation) -> str:
        user_request = activation.remaining_text or ("No additional task text was provided after the slash skill command. Ask the user what they want to do with this skill if the next step is unclear.")
        escaped_user_request = html.escape(user_request, quote=False)
        escaped_skill_content = html.escape(activation.skill_content, quote=False)
        escaped_skill_name = html.escape(activation.skill_name, quote=True)
        escaped_category = html.escape(activation.category, quote=True)
        escaped_path = html.escape(activation.container_file_path, quote=True)
        escaped_content_hash = html.escape(activation.content_hash, quote=True)
        activation_reason = (
            f"The user explicitly activated the `{activation.skill_name}` skill for this turn." if activation.mode == "explicit" else f"The runtime deterministically routed this request to the `{activation.skill_name}` domain skill."
        )
        return f"""<slash_skill_activation mode="{activation.mode}">
{activation_reason}
Treat the task text as:
<user_request>
{escaped_user_request}
</user_request>

Follow this skill before choosing a general workflow. Load supporting resources from the same skill directory only when needed.

<skill name="{escaped_skill_name}" category="{escaped_category}" path="{escaped_path}" sha256="{escaped_content_hash}">
<skill_content encoding="xml-escaped">
{escaped_skill_content}
</skill_content>
</skill>
</slash_skill_activation>"""

    @staticmethod
    def _has_existing_activation_for_target(messages: list, target_index: int, target: HumanMessage) -> bool:
        if target_index <= 0:
            return False

        if target.id:
            for previous in messages[:target_index]:
                if not is_slash_skill_activation_reminder(previous):
                    continue
                target_id = previous.additional_kwargs.get(_SLASH_SKILL_ACTIVATION_TARGET_ID_KEY)
                if target_id == target.id or previous.id == f"{target.id}__slash_activation":
                    return True

        previous = messages[target_index - 1]
        return is_slash_skill_activation_reminder(previous)

    def _find_activation_target(
        self,
        messages: list,
        state: dict | None = None,
    ) -> tuple[int, HumanMessage, _ActivationResolution] | None:
        if not messages:
            return None

        target_index = next((idx for idx in range(len(messages) - 1, -1, -1) if _is_user_activation_target(messages[idx])), None)
        if target_index is None:
            return None

        target = messages[target_index]
        if target is None:
            return None
        if self._has_existing_activation_for_target(messages, target_index, target):
            return None

        content = get_original_user_content_text(target.content, target.additional_kwargs)
        resolution = self._resolve_activation(content, state)
        if resolution is None:
            return None
        return target_index, target, resolution

    @staticmethod
    def _record_activation(request: ModelRequest, activation: _Activation, *, hook: str) -> None:
        runtime = getattr(request, "runtime", None)
        context = getattr(runtime, "context", None)
        journal = context.get("__run_journal") if isinstance(context, dict) else None
        if journal is None:
            return
        try:
            changes = {
                "skill_name": activation.skill_name,
                "category": activation.category,
                "path": activation.container_file_path,
                "content_hash": activation.content_hash,
            }
            if activation.mode != "explicit":
                changes["mode"] = activation.mode
            journal.record_middleware(
                "skill_activation",
                name="SkillActivationMiddleware",
                hook=hook,
                action="activate",
                changes=changes,
            )
        except Exception:
            logger.debug("Failed to record slash skill activation audit event", exc_info=True)

    def _prepare_model_request(self, request: ModelRequest, *, hook: str) -> ModelRequest | AIMessage | None:
        target_and_resolution = self._find_activation_target(list(request.messages), request.state)
        if target_and_resolution is None:
            return None

        target_index, target, resolution = target_and_resolution
        if resolution.failure_message:
            return AIMessage(content=resolution.failure_message)

        activation = resolution.activation
        if activation is None:
            return None

        logger.info(
            "SkillActivationMiddleware: activating skill %s mode=%s category=%s path=%s hash=%s",
            activation.skill_name,
            activation.mode,
            activation.category,
            activation.container_file_path,
            activation.content_hash,
        )
        self._record_activation(request, activation, hook=hook)
        self._apply_skill_secrets(request, activation)
        activation_msg = self._make_activation_message(target, self._build_activation_reminder(activation))
        messages = list(request.messages)
        messages.insert(target_index, activation_msg)
        return request.override(messages=messages)

    @staticmethod
    def _apply_skill_secrets(request: ModelRequest, activation: _Activation) -> None:
        """Resolve the activated skill's declared secrets into the per-run injection
        set (binding point A, issue #3861).

        For each declared secret present in the request's ``context.secrets``,
        record its value in the injection set stored under
        ``ACTIVE_SECRETS_CONTEXT_KEY`` on the shared run context, so the bash tool
        can build the subprocess env for this turn. The injected value always comes
        from the caller's request — never from the host environment, which is
        scrubbed of secret-looking names by ``env_policy.build_sandbox_env`` before
        injection. A skill can therefore never harvest a host platform credential
        (it only ever receives what the caller explicitly supplied), so a declared
        name that also exists in the host env is fine: the caller's value wins and
        the host value is dropped. Secret *values* are never logged.
        """
        runtime = getattr(request, "runtime", None)
        context = getattr(runtime, "context", None)
        if not isinstance(context, dict):
            return
        # Unconditionally clear any active-secret set a previous activation in
        # the same run may have written, before this turn's resolution decides
        # what (if anything) to install. Otherwise a later skill that declares
        # no secrets, or whose required secrets the caller did not supply, would
        # inherit the previous skill's injection set and the bash tool would
        # inject those values into a subprocess that never declared them (#3861).
        context.pop(ACTIVE_SECRETS_CONTEXT_KEY, None)
        if not activation.required_secrets:
            return

        request_secrets = extract_request_secrets(context)
        injected: dict[str, str] = {}
        missing: list[str] = []
        for req in activation.required_secrets:
            if req.name in request_secrets:
                injected[req.name] = request_secrets[req.name]
            elif not req.optional:
                missing.append(req.name)

        if injected:
            context[ACTIVE_SECRETS_CONTEXT_KEY] = injected
        if missing:
            logger.warning(
                "Skill %s activated but required secrets are missing from the request context: %s",
                activation.skill_name,
                ", ".join(sorted(missing)),
            )

    @staticmethod
    def _make_activation_message(target: HumanMessage, activation_content: str) -> HumanMessage:
        stable_id = target.id or str(uuid.uuid4())
        additional_kwargs = {
            "hide_from_ui": True,
            _SLASH_SKILL_ACTIVATION_KEY: True,
        }
        if target.id:
            additional_kwargs[_SLASH_SKILL_ACTIVATION_TARGET_ID_KEY] = target.id
        return HumanMessage(
            content=activation_content,
            id=f"{stable_id}__slash_activation",
            additional_kwargs=additional_kwargs,
        )

    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | AIMessage:
        prepared = self._prepare_model_request(request, hook="wrap_model_call")
        if prepared is None:
            return handler(request)
        if isinstance(prepared, AIMessage):
            return prepared
        return handler(prepared)

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | AIMessage:
        prepared = await asyncio.to_thread(self._prepare_model_request, request, hook="awrap_model_call")
        if prepared is None:
            return await handler(request)
        if isinstance(prepared, AIMessage):
            return prepared
        return await handler(prepared)
