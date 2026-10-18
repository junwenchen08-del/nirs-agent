"""Provider input views and size audit; durable context/decisions stay intact."""

import copy
import hashlib
import html
import json
import re

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool


def malformed_call_message_view(messages):
    """Omit invalid-only calls and their recovery results in the provider copy.

    DanglingToolCallMiddleware may already have inserted error ToolMessages for
    unparseable calls. Removing only the assistant's calls leaves orphan results
    that strict providers reject. Scope removal to that assistant's batch so a
    reused transport ID in a different valid batch keeps its actual result.
    The checkpoint, parser correction and durable model receipts are untouched.
    """
    projected = []
    invalid_ids = set()
    changed = False
    for message in messages:
        if isinstance(message, AIMessage):
            invalid_ids = set()
            if message.invalid_tool_calls and not message.tool_calls:
                calls = [*message.invalid_tool_calls, *(message.additional_kwargs.get("tool_calls") or [])]
                invalid_ids = {call["id"] for call in calls if isinstance(call, dict) and call.get("id")}
                message = message.model_copy(
                    update={
                        "content": "Previous tool response contained invalid JSON and was not executed.",
                        "invalid_tool_calls": [],
                        "additional_kwargs": {key: value for key, value in message.additional_kwargs.items() if key != "tool_calls"},
                    }
                )
                changed = True
        elif isinstance(message, ToolMessage) and message.tool_call_id in invalid_ids:
            changed = True
            continue
        projected.append(message)
    return projected if changed else messages


def runtime_provider_cost(runtime):
    """Read actual all-caller usage from the existing trusted RunJournal.

    Never merge message estimates with provider usage, or publish journal text.
    A missing journal stays unavailable rather than being invented as zero.
    """
    from deerflow.runtime.journal import RunJournal

    config = getattr(runtime, "config", None) or {}
    callbacks = config.get("callbacks")
    handlers = callbacks if isinstance(callbacks, list) else getattr(callbacks, "handlers", [])
    run_id = (getattr(runtime, "context", None) or {}).get("run_id")
    thread_id = config.get("configurable", {}).get("thread_id")
    for handler in handlers:
        if isinstance(handler, RunJournal) and handler.run_id == run_id and handler.thread_id == thread_id:
            value = handler.get_completion_data()
            keys = ("total_input_tokens", "total_output_tokens", "total_tokens", "llm_call_count", "lead_agent_tokens", "middleware_tokens", "subagent_tokens", "token_usage_by_model")
            return {"source": "trusted_run_journal", "scope": "current_run_until_deterministic_publication", "run_id": run_id, "usage_kind": "actual_provider_usage", **{key: value[key] for key in keys}}
    return None


def autonomous_skill_message_view(message):
    """Omit only the verified skill's legacy section in the provider copy."""
    if not (isinstance(message, HumanMessage) and message.additional_kwargs.get("slash_skill_activation") and isinstance(message.content, str)):
        return message
    match = re.search(r'(<skill name="nir-coordinator"[^>]*sha256="([a-f0-9]{64})"[^>]*>\s*<skill_content encoding="xml-escaped">\n)(.*?)(\n</skill_content>)', message.content, re.S)
    if not match:
        return message
    source = html.unescape(match[3])
    if hashlib.sha256(source.encode("utf-8")).hexdigest() != match[2]:
        return message
    start = source.find("\n## Phase-")
    end = source.find("\n# NIR 光谱分析协调器")
    if start < 0 or end <= start:
        return message
    selected = source[start:end]
    if not all(heading in selected for heading in ("Phase-1", "Phase-2", "Phase-3", "Phase-4", "Runtime validation evidence")):
        return message
    projected = '<autonomous_skill_projection source="original_skill_sha256" omitted="legacy_sections_and_frontmatter_tool_list">\n' + html.escape(selected, quote=False) + "\n</autonomous_skill_projection>"
    content = message.content[: match.start(3)] + projected + message.content[match.end(3) :]
    return message.model_copy(update={"content": content})


def context_message_view(message, context):
    value = copy.deepcopy(context)
    value.pop("proposal_schema", None)
    value["proposal_schema_source"] = "nir_submit_decision.current_tool_schema"
    manifest = value.get("evaluation", {}).get("cv_manifest")
    if manifest and "folds" in manifest:
        value["evaluation"]["cv_manifest"] = {key: item for key, item in manifest.items() if key != "folds"}
        value["evaluation"]["cv_manifest"]["fold_sizes"] = [{"train": len(fold["train"]), "validation": len(fold["validation"])} for fold in manifest["folds"]]
    value["provider_projection"] = {
        "source": "immutable_context_identified_by_context_id_revision_sha256",
        "omitted": ["duplicate_proposal_schema", "Cal_row_membership_indices"],
        "cv_membership_source": "persisted_frozen_cv_manifest",
    }
    return message.model_copy(update={"content": json.dumps(value, ensure_ascii=False, separators=(",", ":"))})


def request_cost_profile(request):
    """Exact serialized UTF-8 size proxy; never describe this as token usage."""

    def size(value):
        return len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"))

    payload = {
        "messages": [message.model_dump() for message in request.messages],
        "system": request.system_message.model_dump() if request.system_message else None,
        "tools": [convert_to_openai_tool(item) for item in request.tools],
    }
    components = {key: size(value) for key, value in payload.items()}
    total = size(payload)
    components["serialization_overhead"] = total - sum(components.values())
    message_groups = {"task_messages": 0, "decision_context": 0, "tool_results_and_corrections": 0, "assistant_history": 0}
    contexts = []
    for message in request.messages:
        group = "task_messages" if isinstance(message, HumanMessage) else "tool_results_and_corrections" if isinstance(message, ToolMessage) else "assistant_history"
        if isinstance(message, ToolMessage) and message.name == "nir_decision_context":
            group = "decision_context"
            try:
                context = json.loads(message.content)
                contexts.append(
                    {
                        "context_id": context.get("context_id"),
                        "phase": context.get("phase"),
                        "knowledge_bytes": size(context.get("knowledge", {})),
                        "history_bytes": size({key: context.get(key) for key in ("history", "history_state")}),
                        "schema_bytes": size(context.get("proposal_schema")) if context.get("proposal_schema") else 0,
                    }
                )
            except (TypeError, ValueError):
                pass
        message_groups[group] += size(message.model_dump())
    return {
        "schema_version": 1,
        "measurement": "utf8_bytes_proxy_not_tokenizer",
        "input_utf8_bytes": total,
        "components_bytes": components,
        "message_groups_bytes": message_groups,
        "context_subsets_non_additive": contexts,
        "tool_count": len(request.tools),
        "message_count": len(request.messages),
        "actual_tokens_source": "provider_usage_metadata_separately_recorded",
    }
