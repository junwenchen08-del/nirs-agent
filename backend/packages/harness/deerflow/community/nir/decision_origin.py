"""Server-issued receipts captured at the real Lead Agent model boundary."""

import json

from langchain_core.messages import AIMessage

from .process import _digest, _json, _now

DECISION_CALLS = {"nir_submit_decision", "nir_execute_decision", "nir_workflow"}


def record_model_response(process, model, messages, *, run_id=None, session_id=None, elapsed_seconds=None, project_id=None, workflow_start_call_id=None, estimated_tokens=None, intake_lease_id=None, request_profile=None):
    if process is None:
        return
    model_type = str(getattr(model, "_llm_type", type(model).__name__))
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        if session_id or project_id or workflow_start_call_id:
            response_id = "response_" + _digest([session_id or project_id or workflow_start_call_id, message.id, message.content, message.tool_calls])[:32]
            accounting = {
                "session_id": session_id,
                "project_id": project_id,
                "workflow_start_call_id": workflow_start_call_id,
                "message_id": message.id,
                "agent_run_id": run_id,
                "model_type": model_type,
                "token_usage": message.usage_metadata,
                "elapsed_seconds": elapsed_seconds,
                "issued_at": _now(),
                "admission_tokens_estimated": estimated_tokens,
                "intake_lease_id": intake_lease_id,
            }
            accounting.update(
                source="lead_agent_model_call",
                status="succeeded",
                model_name=str(getattr(model, "model_name", getattr(model, "model", model_type))),
                provider_class=type(model).__name__,
                request_id=(message.response_metadata or {}).get("request_id"),
                usage_kind="actual" if message.usage_metadata else "unavailable",
            )
            accounting["finish_reason"] = (message.response_metadata or {}).get("finish_reason")
            accounting["invalid_tool_calls"] = [{"id": call.get("id"), "name": call.get("name"), "args_sha256": _digest(call.get("args")), "args_chars": len(call.get("args") or "")} for call in message.invalid_tool_calls]
            if request_profile is not None:
                accounting["request_profile"] = request_profile
            with process._connection(write=True) as db:
                db.execute("INSERT OR IGNORE INTO model_issued_calls VALUES (?,'__model_response__',?,?)", (response_id, _digest(accounting), _json(accounting)))
        for call in message.tool_calls:
            if call["name"] not in DECISION_CALLS:
                continue
            receipt = {
                "source": "lead_agent_model_call",
                "model_class": type(model).__name__,
                "model_type": model_type,
                "model_name": str(getattr(model, "model_name", getattr(model, "model", model_type))),
                "message_id": message.id,
                "tool_call_id": call["id"],
                "agent_run_id": run_id,
                "response_sha256": _digest({"content": message.content, "tool_calls": message.tool_calls}),
                "issued_at": _now(),
                "token_usage": message.usage_metadata,
                "verification_kind": "fake_model" if "fake" in model_type.lower() else "provider_model",
            }
            with process._connection(write=True) as db:
                old = db.execute("SELECT args_hash FROM model_issued_calls WHERE id=?", (call["id"],)).fetchone()
                digest = _digest(call["args"])
                if old is not None and old[0] != digest:
                    raise ValueError("Conflicting model-issued call identity")
                db.execute("INSERT OR IGNORE INTO model_issued_calls VALUES (?,?,?,?)", (call["id"], call["name"], digest, _json(receipt)))


def issued_origin(process, call_id, tool, args):
    with process._connection() as db:
        row = db.execute("SELECT * FROM model_issued_calls WHERE id=?", (call_id,)).fetchone()
    if row is None or row["tool"] != tool or row["args_hash"] != _digest(args):
        raise ValueError("A matching server model-issued tool call is required; caller author fields are not provenance")
    return json.loads(row["payload"])
