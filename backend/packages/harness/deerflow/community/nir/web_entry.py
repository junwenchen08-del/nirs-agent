"""Trusted web mode binding; uses the existing immutable delivery policy."""

import json
import os

from .decision_budget import SERVER_DEFAULTS, resolve_policy

WEB_BUDGET_FIELDS = frozenset({"max_experiments", "max_fits", "max_decisions", "max_wall_seconds"})
UNIFIED_BINDING_ID = "unified_workflow_binding"


def unified_entry(context):
    return (context or {}).get("nir_unified_entry") == {"schema_version": 1, "intent": "pending"}


def completed_intake_workflow(workflow):
    """Only a completed, non-modeling inspection/knowledge task can hand off.

    A search, failed modeling task or bound resource policy is never restarted.
    The persisted receipt/session checks at start are authoritative as well.
    """
    return bool(
        workflow
        and workflow.get("stage") == "completed"
        and workflow.get("task_type") in {"inspection", "knowledge"}
        and workflow.get("search_protocol", "legacy") == "legacy"
        and not any(
            workflow.get(key)
            for key in (
                "search_session_id",
                "autonomous_delivery",
                "delivery_policy",
                "attempt",
                "attempts",
                "attempt_evidence",
                "model_path",
                "metrics_path",
            )
        )
    )


def unified_policy():
    return resolve_policy("unified")


def stored_workflow(process):
    """The existing receipt index owns the binding even before preparation."""
    if process is None or not process.db.exists():
        return None
    with process._connection() as db:
        row = db.execute("SELECT payload FROM model_issued_calls WHERE id=?", (UNIFIED_BINDING_ID,)).fetchone()
    return json.loads(row[0])["workflow"] if row else None


def persist_unified_workflow(runtime, workflow, *, starting=False, previous=None):
    from .process import _digest, _json, runtime_store

    if (workflow.get("delivery_policy") or {}).get("profile") != "unified":
        return
    process = runtime_store(runtime)
    if process is None:
        raise ValueError("Unified modeling requires an owned thread")
    binding = {key: workflow.get(key) for key in ("project_id", "search_protocol", "decision_strategy", "autonomous_rounds", "autonomous_delivery", "delivery_policy", "workflow_start_call_id")}
    with process._connection(write=True) as db:
        row = db.execute("SELECT payload FROM model_issued_calls WHERE id=?", (UNIFIED_BINDING_ID,)).fetchone()
        if previous is not None:
            if not starting or not completed_intake_workflow(previous) or row or any(db.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone() for table in ("search_sessions", "initial_decisions", "attempts")):
                raise ValueError("Only an unmodeled completed inspection can hand off; existing searches and budgets cannot restart")
            archive_id = "completed_intake_" + _digest(previous)
            archive = {"workflow": previous, "source": "completed_non_modeling_workflow"}
            db.execute("INSERT OR IGNORE INTO model_issued_calls VALUES (?, '__completed_intake__', ?, ?)", (archive_id, _digest(previous), _json(archive)))
            workflow["previous_workflow_id"] = archive_id
        if row:
            old = json.loads(row[0])
            if starting or old["binding"] != binding or old["workflow"].get("search_session_id") and old["workflow"]["search_session_id"] != workflow.get("search_session_id"):
                raise ValueError("This thread's modeling protocol is already bound; use a new thread for a new task")
        payload = {"binding": binding, "workflow": workflow}
        db.execute("INSERT OR REPLACE INTO model_issued_calls VALUES (?, '__unified_binding__', ?, ?)", (UNIFIED_BINDING_ID, _digest(binding), _json(payload)))


def validate_unified_start(runtime, workflow):
    """Semantic task recognition belongs to the actual Lead tool call.

    No keyword classifier starts training. A model-issued calibration/analysis
    start is the explicit execution intent; missing requirements stay in intake.
    The server independently binds policy and checks the owned upload path.
    """
    from pathlib import PurePosixPath

    from langchain_core.messages import AIMessage

    from .decision_origin import issued_origin
    from .process import runtime_store

    calls = [call for message in (runtime.state or {}).get("messages", []) if isinstance(message, AIMessage) for call in message.tool_calls if call["id"] == runtime.tool_call_id]
    if len(calls) != 1:
        raise ValueError("Unified task recognition requires the actual Lead start call")
    call = calls[0]
    issued_origin(runtime_store(runtime), runtime.tool_call_id, "nir_workflow", call["args"])
    for key in ("task_type", "data_path", "analyte", "unit", "domain", "model_constraint", "pipeline_constraint"):
        if key in call["args"] and key != "pipeline_constraint" and workflow.get(key) != (call["args"][key].strip() if isinstance(call["args"][key], str) else call["args"][key]):
            raise ValueError("Start intent differs from the model-issued task: " + key)
    path = workflow.get("data_path")
    if path:
        virtual = PurePosixPath(path)
        if not str(virtual).startswith("/mnt/user-data/uploads/") or ".." in virtual.parts or virtual.suffix.lower() not in {".npz", ".csv", ".txt", ".tsv", ".mat"}:
            raise ValueError("Unified regression requires an owned numeric upload")
        process = runtime_store(runtime)
        uploads = process.root.parent / "user-data" / "uploads"
        source = uploads.joinpath(*virtual.parts[4:])
        if not source.is_file() or any(parent.is_symlink() for parent in (source, *source.parents)) or not source.resolve().is_relative_to(uploads.resolve()):
            raise ValueError("Unified regression requires an existing, owned, non-symlink upload")


def web_request(context):
    return (context or {}).get("nir_web_request")


def bind_web_request(config, body):
    """Only validated body.context selections may create this runtime binding.

    Gateway owner authorization is unchanged. Neither run input nor arbitrary
    RunnableConfig overrides can manufacture the internal request.
    """
    for container in ("context", "configurable"):
        if isinstance(config.get(container), dict):
            config[container].pop("nir_web_request", None)
            config[container].pop("nir_autonomous", None)
            config[container].pop("nir_unified_entry", None)
    if "nir_workflow" in (getattr(body, "input", None) or {}):
        raise ValueError("nir_workflow is server-owned; cannot be submitted by the web client")
    selected = (getattr(body, "context", None) or {}).get("nir_autonomous")
    if any(config.get(container, {}).get("agent_name") for container in ("context", "configurable")):
        # Custom agents retain their established entry contract.
        return
    if getattr(body, "assistant_id", "__internal__") in {"lead_agent", None} and isinstance(getattr(body, "input", None), dict) and body.input.get("messages"):
        # Default Gateway chat has one entry. A stale/forged old UI selection
        # cannot choose a different protocol or grant numerical resources.
        config.setdefault("context", {})["nir_unified_entry"] = {"schema_version": 1, "intent": "pending"}
        return
    if selected is None:
        # This is an intent boundary, not a modeling classification. The actual
        # Lead chooses start/clarification/ordinary QA from the user task. Only
        # the server-side workflow start can bind the scientific protocol.
        if getattr(body, "assistant_id", None) == "lead_agent" and isinstance(getattr(body, "input", None), dict) and body.input.get("messages"):
            config.setdefault("context", {})["nir_unified_entry"] = {"schema_version": 1, "intent": "pending"}
        return
    if not isinstance(selected, dict) or set(selected) - {"enabled", "profile", "budget"} or selected.get("enabled") is not True:
        raise ValueError("nir_autonomous: expected enabled=true, profile and optional budget")
    profile = selected.get("profile", "development")
    if profile not in {"development", "autonomous", "custom"}:
        raise ValueError("nir_autonomous.profile: development, autonomous or custom required")
    requested = selected.get("budget", {})
    if not isinstance(requested, dict) or set(requested) - WEB_BUDGET_FIELDS:
        raise ValueError("nir_autonomous.budget: only experiments, fits, decisions and wall seconds are writable")
    # Custom numerical limits belong to autonomous search, not to the smaller
    # development LLM allowance. Hidden fields still come only from server policy.
    resolved_profile = "autonomous" if profile == "custom" else profile
    if profile == "custom":
        server_defaults = {key: min(default, int(os.environ.get("NIR_AUTONOMOUS_" + key.upper(), default))) for key, default in SERVER_DEFAULTS.items()}
        requested = server_defaults | requested
    policy = resolve_policy(resolved_profile, requested)
    bound = {
        "search_protocol": "sealed_holdout",
        "decision_strategy": "llm_autonomous",
        "autonomous_rounds": True,
        "autonomous_delivery": True,
        "budget_profile": resolved_profile,
        "task_budget": json.dumps({key: policy[key] for key in SERVER_DEFAULTS}, sort_keys=True),
        "delivery_policy": policy,
        "ui_profile": profile,
    }
    config.setdefault("context", {})["nir_web_request"] = bound
    # Autonomous work executes on the Lead; generic chat mode cannot dispatch
    # subagents or a planning agent around its sealed tool policy.
    for container in ("context", "configurable"):
        config.setdefault(container, {}).update(is_plan_mode=False, subagent_enabled=False)
