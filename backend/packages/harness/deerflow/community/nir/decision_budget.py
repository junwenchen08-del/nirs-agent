"""Versioned resource policy; ledgers live in the existing search session.

Wall checks are cooperative. New unified policies count tokens as statistics;
legacy policies retain estimated admission. Single-call output stays bounded.
This is not a currency limit or a promise about provider billing/tokenization.
"""

import json
import os
import time
import uuid

from .process import _now

SERVER_DEFAULTS = {
    "max_experiments": 40,
    "max_fits": 1000,
    "max_decisions": 100,
    "max_wall_seconds": 7200,
    "max_llm_calls": 150,
    "max_llm_tokens": 3_000_000,
    "max_invalid_decision_repairs": 2,
    "max_output_tokens": 4096,
    "final_llm_calls": 6,
    "final_llm_tokens": 500_000,
    "final_wall_seconds": 300,
}
PROFILES = {
    "development": {**SERVER_DEFAULTS, "max_experiments": 5, "max_fits": 50, "max_decisions": 15, "max_wall_seconds": 600, "max_llm_calls": 30, "max_llm_tokens": 1_200_000},
    "autonomous": SERVER_DEFAULTS,
    "unified": {**SERVER_DEFAULTS, "max_llm_tokens": None, "final_llm_tokens": None, "max_invalid_decision_repairs": 5},
}

TOKEN_FIELDS = frozenset({"max_llm_tokens", "final_llm_tokens"})


def token_cap_exceeded(policy, field, total):
    """None means disabled, rather than a fabricated very large allowance."""
    cap = policy[field]
    return cap is not None and total > cap


class DecisionBudgetError(ValueError):
    def __init__(self, message, **details):
        super().__init__("budget: " + message)
        self.details = details


def resolve_policy(profile="development", requested=None):
    if profile not in PROFILES:
        raise ValueError("Unknown autonomous budget profile")
    requested = requested or {}
    if not isinstance(requested, dict) or set(requested) - set(SERVER_DEFAULTS):
        raise ValueError("Unknown autonomous budget fields")
    ceilings = {}
    for key, default in SERVER_DEFAULTS.items():
        if profile == "unified" and key in TOKEN_FIELDS:
            ceilings[key] = None
            continue
        if profile == "unified" and key == "max_invalid_decision_repairs":
            default = 5
        value = int(os.environ.get("NIR_AUTONOMOUS_" + key.upper(), default))
        if value < 1:
            raise ValueError("Invalid server autonomous ceiling: " + key)
        ceilings[key] = value
    defaults = dict(PROFILES[profile])
    if profile == "unified":
        defaults = {key: min(value, ceilings[key]) if value is not None else None for key, value in defaults.items()}
    values = defaults | requested
    for key, value in values.items():
        if profile == "unified" and key in TOKEN_FIELDS:
            if value is not None:
                raise ValueError(f"{key}: unified token usage is statistics only; cap must be null")
            continue
        if type(value) is not int or not 1 <= value <= ceilings[key]:
            raise ValueError(f"{key}: task budget must be 1..{ceilings[key]}; choose an explicit smaller request")
    minimum_final_calls = 2 if profile == "unified" else 4
    if values["final_llm_calls"] < minimum_final_calls:
        raise ValueError(f"final_llm_calls: reserve at least {minimum_final_calls} terminal tool-call responses")
    policy = {"schema_version": 1, "profile": profile, **values, "server_ceilings": ceilings, "wall_enforcement": "cooperative_checkpoints", "token_enforcement": "estimated_admission_and_actual_usage", "monetary_limit": None}
    if profile == "unified":
        policy.update(schema_version=2, token_enforcement="statistics_only", automatic_delivery=True)
    return policy


def initial_ledger():
    bucket = {"calls": 0, "charged_tokens": 0, "actual_input_tokens": 0, "actual_output_tokens": 0, "unknown_usage_calls": 0, "seconds": 0.0}
    return {"search": dict(bucket), "finalization": dict(bucket), "inflight": None}


INTAKE_ID = "phase4_delivery_intake"


def reserve_intake_call(process, policy, estimated_input_tokens, *, initial_receipts=None):
    """Bound task intake in the same receipt store, before a session exists."""
    from .process import _json

    with process._connection(write=True) as db:
        row = db.execute("SELECT payload FROM model_issued_calls WHERE id=?", (INTAKE_ID,)).fetchone()
        ledger = json.loads(row[0]) if row else {"started_at": time.time(), "budget": initial_ledger(), "stage": "intake"}
        if not row and policy.get("profile") == "unified":
            # The first semantic start was issued before a task existed. Count
            # its trusted receipt once, before admitting the next model call.
            for receipt in initial_receipts or []:
                first = ledger["budget"]["search"]
                usage = receipt.get("token_usage") or {}
                first["calls"] += 1
                first["actual_input_tokens"] += usage.get("input_tokens", 0)
                first["actual_output_tokens"] += usage.get("output_tokens", 0)
                first["charged_tokens"] += max(receipt.get("admission_tokens_estimated") or 0, usage.get("total_tokens", 0))
                first["unknown_usage_calls"] += int(not bool(usage))
                first["seconds"] += receipt.get("elapsed_seconds") or 0.0
                ledger["started_at"] -= receipt.get("elapsed_seconds") or 0.0
        if ledger["budget"]["inflight"] or ledger["stage"] != "intake":
            raise ValueError("intake: closed or uncertain provider request; no automatic replay")
        bucket = ledger["budget"]["search"]
        estimate = max(1, int(estimated_input_tokens)) + policy["max_output_tokens"]
        reasons = []
        if bucket["calls"] >= policy["max_llm_calls"]:
            reasons.append("llm_calls")
        if token_cap_exceeded(policy, "max_llm_tokens", bucket["charged_tokens"] + estimate):
            reasons.append("llm_tokens")
        if time.time() - ledger["started_at"] >= policy["max_wall_seconds"]:
            reasons.append("wall_seconds")
        if reasons:
            ledger.update(stage="no_deliverable_model", stop_source="budget", stop_reason="intake_allowance_exhausted")
            ledger["stop_reasons"] = reasons
            lease = None
        else:
            lease = {"id": "intake_" + uuid.uuid4().hex, "bucket": "search", "estimated_tokens": estimate, "issued_at": _now()}
            bucket["calls"] += 1
            bucket["charged_tokens"] += estimate
            ledger["budget"]["inflight"] = lease
        db.execute("INSERT OR REPLACE INTO model_issued_calls VALUES (?,'__delivery_intake__','',?)", (INTAKE_ID, _json(ledger)))
    if lease is None:
        raise DecisionBudgetError("intake allowance exhausted; no successful model exists")
    return lease


def finish_intake_call(process, lease, usage, seconds, *, failed=False):
    from .process import _json

    with process._connection(write=True) as db:
        ledger = json.loads(db.execute("SELECT payload FROM model_issued_calls WHERE id=?", (INTAKE_ID,)).fetchone()[0])
        if ledger["budget"]["inflight"] != lease:
            raise ValueError("intake: conflicting provider lease")
        bucket = ledger["budget"]["search"]
        bucket["seconds"] += seconds
        if usage:
            inp, out = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
            bucket["actual_input_tokens"] += inp
            bucket["actual_output_tokens"] += out
            bucket["charged_tokens"] += max(0, inp + out - lease["estimated_tokens"])
        else:
            bucket["unknown_usage_calls"] += 1
        ledger["budget"]["inflight"] = None
        bucket["last_call"] = {"lease_id": lease["id"], "status": "failed" if failed else "succeeded", "elapsed_seconds": seconds, "usage_kind": "actual" if usage else "unavailable"}
        if failed:
            ledger.update(stage="safety_closed", stop_source="tool_failure")
        db.execute("UPDATE model_issued_calls SET payload=? WHERE id=?", (_json(ledger), INTAKE_ID))


def end_search(session, reasons):
    if session["stage"] == "searching":
        session.update(stage="budget_exhausted", stop_source="budget", stop_reason=list(reasons), stopped_at=_now())


def wall_exhausted(session):
    return time.time() - session["budget_started_at"] >= session["delivery_policy"]["max_wall_seconds"]


def reserve_call(search, sid, estimated_input_tokens, *, defer_finalization=False):
    """Durable before-provider lease; uncertain calls are never refunded/reissued."""
    with search.process._connection(write=True) as db:
        session = search._session(db, sid)
        policy = session.get("delivery_policy")
        if not policy:
            return None
        ledger = session["llm_budget"]
        if ledger["inflight"]:
            raise ValueError("llm: an in-flight/uncertain model request must not be replayed")
        estimate = max(1, int(estimated_input_tokens)) + policy["max_output_tokens"]
        if session["stage"] == "searching":
            bucket = ledger["search"]
            reasons = []
            if bucket["calls"] >= policy["max_llm_calls"]:
                reasons.append("llm_calls")
            if token_cap_exceeded(policy, "max_llm_tokens", bucket["charged_tokens"] + estimate):
                reasons.append("llm_tokens")
            if wall_exhausted(session):
                reasons.append("wall_seconds")
            if reasons:
                end_search(session, reasons)
                # A plan accepted before a resource terminal is preserved but
                # never executed. Close its unused reservation without fitting.
                from .process import _json

                pending = list(db.execute("SELECT id,payload FROM initial_decisions WHERE session_id=? AND status='accepted'", (sid,)))
                for row in pending:
                    record = json.loads(row["payload"])
                    feedback = {
                        "decision_id": row["id"],
                        "experiment_id": record["experiment_id"],
                        "status": "failed",
                        "model_executed": False,
                        "actual_configuration": record["accepted_spec"],
                        "fits": 0,
                        "elapsed_seconds": 0.0,
                        "failure_reason": "Search resource limit reached before execution",
                        "stop_source": "budget",
                    }
                    db.execute("UPDATE initial_decisions SET status='failed',execution=? WHERE id=?", (_json(feedback), row["id"]))
                if pending:
                    session.update(active_decision_id=None, reserved_fits=0, decision_context_id=None)
                if defer_finalization:
                    search._save(db, session)
                    return None  # Rebuild the final request before charging its reserve.
        if session["stage"] in {"cancelled", "safety_closed", "no_deliverable_model", "finalization_unavailable", "final_test_failed", "testing"}:
            raise ValueError("llm: session is closed or requires operator recovery")
        name = "search" if session["stage"] == "searching" else "finalization"
        if name == "finalization" and "final_budget_started_at" not in session:
            session["final_budget_started_at"] = time.time()
        bucket = ledger[name]
        max_calls = policy["max_llm_calls"] if name == "search" else policy["final_llm_calls"]
        token_field = "max_llm_tokens" if name == "search" else "final_llm_tokens"
        final_expired = name == "finalization" and time.time() - session["final_budget_started_at"] >= policy["final_wall_seconds"]
        if bucket["calls"] >= max_calls or token_cap_exceeded(policy, token_field, bucket["charged_tokens"] + estimate) or final_expired:
            reasons = [
                *(["llm_calls"] if bucket["calls"] >= max_calls else []),
                *(["llm_tokens"] if token_cap_exceeded(policy, token_field, bucket["charged_tokens"] + estimate) else []),
                *(["finalization_wall_seconds"] if final_expired else []),
            ]
            session.update(finalization_error="Reserved provider allowance unavailable: " + ",".join(reasons), finalization_reason=reasons)
            if not session.get("final"):
                session["stage"] = "finalization_unavailable"
            search._save(db, session)
            failure = True
        else:
            lease = {"id": "llm_" + uuid.uuid4().hex, "bucket": name, "estimated_tokens": estimate, "issued_at": _now()}
            bucket["calls"] += 1
            bucket["charged_tokens"] += estimate
            ledger["inflight"] = lease
            search._save(db, session)
            failure = False
    if failure:
        raise DecisionBudgetError("reserved finalization LLM allowance exhausted")
    return lease


def finish_call(search, sid, lease, usage, seconds, *, failed=False, failure_reason=None):
    if not lease:
        return
    with search.process._connection(write=True) as db:
        session = search._session(db, sid)
        ledger = session["llm_budget"]
        if ledger["inflight"] != lease:
            raise ValueError("llm: conflicting request lease")
        bucket = ledger[lease["bucket"]]
        bucket["seconds"] += seconds
        if usage:
            inp, out = usage.get("input_tokens", 0), usage.get("output_tokens", 0)
            bucket["actual_input_tokens"] += inp
            bucket["actual_output_tokens"] += out
            bucket["charged_tokens"] += max(0, inp + out - lease["estimated_tokens"])
        else:
            bucket["unknown_usage_calls"] += 1
        ledger["inflight"] = None
        bucket["last_call"] = {"lease_id": lease["id"], "status": "failed" if failed else "succeeded", "elapsed_seconds": seconds, "usage_kind": "actual" if usage else "unavailable", "failure": failure_reason if failed else None}
        if failed:
            session["provider_failure"] = failure_reason or {"exception_type": "unknown", "http_status": None}
            session.update(terminal_failure_source="tool_failure", terminal_failure_reason="provider_call_failed")
            if not session.get("final"):
                session.update(stage="safety_closed", stopped_at=_now())
                if session.get("stop_source") != "budget":
                    session.update(stop_source="tool_failure", stop_reason=["provider_call_failed"])
        search._save(db, session)


def record_invalid(search, sid, call_id=None):
    with search.process._connection(write=True) as db:
        session = search._session(db, sid)
        if not session.get("delivery_policy"):
            return True
        # Rejected calls after a terminal choice/Test must not replace a
        # durable frozen/final result or falsely relabel consumed Test sealed.
        if session["stage"] in {"stopped", "frozen", "testing", "finalized", "final_test_failed"}:
            return True
        if call_id and call_id in session.get("rejected_calls", []):
            return session["stage"] != "safety_closed"
        session["rejected_calls"] = [*session.get("rejected_calls", []), call_id][-10:]
        session["invalid_decisions"] = session.get("invalid_decisions", 0) + 1
        allowed = session["invalid_decisions"] <= session["delivery_policy"]["max_invalid_decision_repairs"]
        if not allowed:
            session.update(stage="safety_closed", terminal_failure_source="decision_invalid", terminal_failure_reason="invalid_decision_repair_limit", stopped_at=_now())
            if session.get("stop_source") != "budget":
                session.update(stop_source="decision_invalid", stop_reason=["invalid_decision_repair_limit"])
        search._save(db, session)
        return allowed


def close_incomplete_workflow(search, sid, *, invalid_json, reason="workflow_continuation_limit"):
    """Persist bounded continuation failure without selecting or testing a model."""
    with search.process._connection(write=True) as db:
        session = search._session(db, sid)
        if session.get("freeze") or session.get("final") or session["stage"] == "stopped":
            return session
        source = "decision_invalid" if invalid_json else "tool_failure"
        session.update(stage="safety_closed", terminal_failure_source=source, terminal_failure_reason=reason, stopped_at=_now())
        if session.get("stop_source") != "budget":
            session.update(stop_source=source, stop_reason=[reason])
        search._save(db, session)
        return session


def final_checkpoint(search, sid, checkpoint):
    """Check cancellation and the reserved cooperative terminal deadline."""
    try:
        checkpoint("autonomous_finalization")
    except BaseException as exc:
        if getattr(exc, "code", None) == "resource_cancelled":
            with search.process._connection(write=True) as db:
                session = search._session(db, sid)
                if session.get("delivery_policy") and session["stage"] != "finalized":
                    session.update(stage="cancelled", stop_source="user", stop_reason=["runtime_cancelled"], stopped_at=_now())
                    search._save(db, session)
                elif session.get("delivery_policy"):
                    session.update(delivery_interrupted_by_user=True, delivery_cancelled_at=_now())
                    search._save(db, session)
        raise
    expired = False
    with search.process._connection(write=True) as db:
        session = search._session(db, sid)
        if session.get("delivery_policy") and session["stage"] != "finalized":
            session.setdefault("final_budget_started_at", time.time())
            expired = time.time() - session["final_budget_started_at"] >= session["delivery_policy"]["final_wall_seconds"]
            if expired:
                session.update(stage="finalization_unavailable", finalization_error="Reserved cooperative finalization deadline exceeded")
            search._save(db, session)
    if expired:
        raise DecisionBudgetError("reserved finalization deadline exceeded")
