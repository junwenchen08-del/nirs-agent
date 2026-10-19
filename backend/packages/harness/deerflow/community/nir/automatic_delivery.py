"""Deterministic completion of a persisted LLM terminal choice, with no provider.

This composes the existing freeze, single-use Test and publication services.
Durable state remains in SearchSessionStore/ProcessStore, not a second scheduler.
Only authoritative orphan-run recovery may release a crashed executor lease.
"""

import json
import uuid

from .decision_budget import final_checkpoint
from .process import _now


def _event(state, stage, **facts):
    state.update(status=stage)
    state["events"] = [*state.get("events", []), {"stage": stage, "at": _now(), "source": "server_terminal_executor", **facts}][-30:]


def complete_terminal_delivery(runtime, search, workflow, *, recovering_run_id=None):
    """Only the named persisted successful experiment can be frozen/delivered."""
    from . import search_tools

    sid = workflow["search_session_id"]
    completed = None
    lease = {"id": "terminal_" + uuid.uuid4().hex, "run_id": (runtime.context or {}).get("run_id")}
    with search.process._connection(write=True) as db:
        session = search._session(db, sid)
        terminal = session.get("automatic_delivery")
        if not (session.get("delivery_policy") or {}).get("automatic_delivery") or not terminal:
            raise ValueError("Automatic delivery requires a persisted validated terminal decision")
        row = db.execute("SELECT payload,status FROM initial_decisions WHERE id=? AND session_id=?", (terminal["terminal_decision_id"], sid)).fetchone()
        record = json.loads(row["payload"]) if row else {}
        retained = session.get("retained_experiment_id")
        if not row or row["status"] != "stopped" or record.get("action") not in {"STOP", "RETAIN"} or record["origin"]["source"] != "lead_agent_model_call" or record["proposal"]["retained_experiment_id"] != retained:
            raise ValueError("Terminal identity or model-issued provenance mismatch")
        if session.get("stop_decision_id") != record["decision_id"] or terminal["retained_experiment_id"] != retained or workflow.get("retained_experiment_id") != retained:
            raise ValueError("Retained identity mismatch; the executor never selects a replacement")
        if session.get("delivery_interrupted_by_user"):
            failure = "Terminal delivery was cancelled; explicit handling is required"
        elif terminal.get("status") == "delivered" and session["stage"] == "finalized" and session.get("export"):
            # Verify the same immutable files on replay, without resetting
            # approval or adding another lifecycle transition/publication.
            completed = (session["final"], terminal["workflow"])
            failure = None
        elif session["stage"] in {"testing", "final_test_failed"} or session.get("export_state", {}).get("status") == "building":
            _event(terminal, "audit_required", reason="uncertain_test_or_publication_lease")
            search._save(db, session)
            failure = "Uncertain Test/publication lease requires audit; no automatic replay"
        elif session["stage"] not in {"stopped", "frozen", "finalized"}:
            failure = "Terminal session is closed or cancelled"
        elif terminal.get("lease") and not (recovering_run_id and terminal["lease"].get("run_id") == recovering_run_id):
            failure = "Terminal executor is in-flight or uncertain; requires authoritative orphan audit"
        else:
            if terminal.get("lease"):
                _event(terminal, "recovered", orphan_run_id=recovering_run_id)
            terminal["lease"] = lease
            _event(terminal, "freezing")
            search._save(db, session)
            failure = None
    if failure:
        raise ValueError(failure)
    if completed:
        from pathlib import Path

        from .decision_delivery import export_delivery

        try:
            search.freeze(sid, retained)
            output = Path(search_tools._resolve(runtime, "/mnt/user-data/outputs", read_only=False))
            return export_delivery(runtime, search, completed[0], output), completed[1]
        except Exception as exc:
            with search.process._connection(write=True) as db:
                current = search._session(db, sid)
                _event(current["automatic_delivery"], "audit_required", reason="completed_artifact_integrity_or_resolution_failure", error_type=type(exc).__name__)
                search._save(db, current)
            raise

    def transition(stage):
        with search.process._connection(write=True) as db:
            current = search._session(db, sid)
            if current["automatic_delivery"].get("lease") != lease:
                raise ValueError("Conflicting deterministic terminal executor")
            _event(current["automatic_delivery"], stage)
            search._save(db, current)

    try:
        checkpoint = search_tools.budget_for_runtime(runtime).checkpoint
        final_checkpoint(search, sid, checkpoint)
        search.freeze(sid, retained)
        transition("frozen")
        final_checkpoint(search, sid, checkpoint)
        transition("final_test_pending")
        final = search.finalize(sid, checkpoint=checkpoint)
        transition("final_test_completed")
        final_checkpoint(search, sid, checkpoint)
        transition("publishing")
        from .decision_provider import runtime_provider_cost

        cost = runtime_provider_cost(runtime)
        if cost is not None:
            with search.process._connection(write=True) as db:
                current = search._session(db, sid)
                current["provider_cost_snapshot"] = cost
                search._save(db, current)
        payload, updated = search_tools.publish_final_delivery(runtime, search, workflow, final)
        with search.process._connection(write=True) as db:
            current = search._session(db, sid)
            state = current["automatic_delivery"]
            if state.get("lease") != lease:
                raise ValueError("Conflicting publication completion")
            _event(state, "delivered")
            state.update(lease=None, workflow=updated)
            search._save(db, current)
        return payload, updated
    except BaseException as exc:
        with search.process._connection(write=True) as db:
            current = search._session(db, sid)
            state = current["automatic_delivery"]
            if state.get("lease") == lease:
                status = "failed"
                if current["stage"] in {"testing", "final_test_failed"} or current.get("export_state", {}).get("status") == "building":
                    status = "audit_required"
                if current["stage"] == "cancelled" or current.get("delivery_interrupted_by_user"):
                    status = "cancelled"
                _event(state, status, error_type=type(exc).__name__)
                state.update(lease=None, error_type=type(exc).__name__)
                search._save(db, current)
        raise


def recover_orphan_terminal(runtime, process, orphan_run_id):
    """Called only by the existing Gateway RunManager orphan reconciliation."""
    from .dataset_session import SearchSessionStore

    search = SearchSessionStore(process)
    with process._connection() as db:
        sessions = [json.loads(row[0]) for row in db.execute("SELECT payload FROM search_sessions")]
    for session in sessions:
        state = session.get("automatic_delivery") or {}
        workflow = (session.get("task_snapshot") or {}).get("_workflow_snapshot")
        if not workflow or not state or state.get("status") == "delivered" or (state.get("lease") or {}).get("run_id") not in {None, orphan_run_id}:
            continue
        if state.get("trigger_run_id") and state["trigger_run_id"] != orphan_run_id:
            continue
        workflow = {**workflow, "search_session_id": session["id"], "retained_experiment_id": session["retained_experiment_id"], "stop_decision_id": session["stop_decision_id"], "stop_source": session["stop_source"]}
        payload, updated = complete_terminal_delivery(runtime, search, workflow, recovering_run_id=orphan_run_id)
        from .web_entry import persist_unified_workflow

        persist_unified_workflow(runtime, updated)
