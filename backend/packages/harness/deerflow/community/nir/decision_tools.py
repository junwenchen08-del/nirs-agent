"""Lead Agent context -> immutable proposal -> governed execution and feedback."""

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from pydantic import ValidationError

from deerflow.tools.types import Runtime
from deerflow.utils.messages import get_original_user_content_text

from ._resources import budget_for_runtime
from .decision_budget import DecisionBudgetError, record_invalid
from .decision_context import InitialDecisionService
from .search_tools import _result, _service, _workflow


def _initial(runtime):
    workflow = _workflow(runtime, {"execution", "decision_review", "budget_exhausted", "awaiting_final_confirmation", "frozen", "review"})
    if workflow.get("decision_strategy") != "llm_autonomous" or not workflow.get("search_session_id"):
        raise ValueError("Prepare a sealed llm_autonomous workflow first")
    return workflow, InitialDecisionService(_service(runtime))


def _error(runtime, exc, *, closed_workflow=None, rejected_proposal=None):
    if isinstance(exc, ValidationError):
        errors = [{"field": ".".join(map(str, item["loc"])), "message": item["msg"], "type": item["type"]} for item in exc.errors(include_input=False, include_url=False)]
    else:
        message = str(exc)[:1000] if isinstance(exc, ValueError) else "Decision operation failed; no replacement plan was executed"
        prefix, separator, _ = message.partition(":")
        field = prefix if separator and len(prefix) < 100 and " " not in prefix else "decision"
        errors = [{"field": field, "message": message, "type": type(exc).__name__}]
    payload = {"status": "error", "code": "nir_decision_rejected", "errors": errors, "can_correct_proposal": True}
    if ((runtime.state or {}).get("nir_workflow") or {}).get("autonomous_delivery"):
        workflow = runtime.state["nir_workflow"]
        if workflow.get("search_session_id"):
            service = _service(runtime)
            with service.process._connection() as db:
                session = service._session(db, workflow["search_session_id"])
                row = db.execute("SELECT payload FROM decision_contexts WHERE id=? AND session_id=?", (session.get("decision_context_id"), session["id"])).fetchone()
            if row:
                import json

                context = json.loads(row[0])
                payload["repair_contract"] = {
                    "reuse_context_id": context["context_id"],
                    "phase": context["phase"],
                    "required_submission_fields": context["proposal_schema"]["required"],
                    "allowed_actions": context["proposal_schema"]["properties"]["action"].get("enum"),
                    "allowed_evidence_ids": context["evidence_ids"],
                    "remaining_repairs": max(0, session["delivery_policy"]["max_invalid_decision_repairs"] - session.get("invalid_decisions", 0) + 1),
                    "nesting": context.get("submission_guide", {}).get("nesting"),
                }
        payload["correction_rules"] = [
            "Decision action must equal the selected alternative action. Only that alternative is executed; the others are comparisons, not extra training arms.",
            "Only keys under capabilities.models[model].parameters are writable. operational_parameters (including random_state/n_jobs) are fixed by the adapter; omit them.",
            "TUNE keeps model/preprocessing; CHANGE_PREPROCESS keeps model; SWITCH_MODEL keeps preprocessing; EXPLORE may change both.",
            "expected_fits = Cartesian candidates of the selected alternative * CV folds + 1. An oversized plan is rejected; explicitly revise it, never assume automatic shrinking.",
        ]
    if closed_workflow:
        payload.update(code="nir_invalid_decision_limit", can_correct_proposal=False)
    if isinstance(exc, DecisionBudgetError):
        payload["details"] = exc.details
    if rejected_proposal is not None:
        # Reuse existing process actions for rejected submissions. These are
        # explicitly not accepted decisions or trained experiments. An actual
        # model-issued receipt is required even for this explanatory evidence.
        import json

        from .decision_origin import issued_origin
        from .process import _now, _safe

        workflow = (runtime.state or {}).get("nir_workflow") or {}
        if workflow.get("search_session_id") and workflow.get("project_id"):
            search = _service(runtime)
            try:
                origin = issued_origin(search.process, runtime.tool_call_id, "nir_submit_decision", {"proposal": rejected_proposal})
            except ValueError:
                origin = None
            if origin:
                proposed = rejected_proposal
                if isinstance(proposed, str):
                    try:
                        proposed = json.loads(proposed) if len(proposed) <= 40000 else {}
                    except ValueError:
                        proposed = {}
                proposed = proposed if isinstance(proposed, dict) else {}
                summary = {key: proposed.get(key) for key in ("action", "phase", "rationale", "expected_outcome", "selected_alternative_id")}
                search.process.record_action(
                    workflow["project_id"],
                    runtime.tool_call_id,
                    "reflection",
                    "rejected",
                    {
                        "stage": workflow["stage"],
                        "session_id": workflow["search_session_id"],
                        "decision_rejection": {
                            "call_id": runtime.tool_call_id,
                            "created_at": _now(),
                            "proposal_summary": _safe(summary),
                            "errors": errors,
                            "code": payload["code"],
                            "model_executed": False,
                            "origin": {key: origin.get(key) for key in ("source", "model_name", "verification_kind")},
                        },
                    },
                )
    return _result(runtime, payload, closed_workflow)


@tool("nir_decision_context")
def nir_decision_context_tool(runtime: Runtime, knowledge_query: str | None = None):
    """Read sourced Cal-only task/knowledge/capabilities/budget and decision schema.

    Multi-round mode includes persisted current/recent/best/retained feedback,
    never Test. Optional knowledge_query retrieves a small additional method card.
    Knowledge relevance is not an execution order or selection rule.
    """
    try:
        workflow, service = _initial(runtime)
        human = [message for message in runtime.state.get("messages", []) if isinstance(message, HumanMessage) and not message.additional_kwargs.get("hide_from_ui")]
        request = get_original_user_content_text(human[0].content, human[0].additional_kwargs) if human else "User task requirements are recorded in the workflow"
        context = service.context(workflow["search_session_id"], workflow, request, knowledge_query=knowledge_query)
        updated = {**workflow, "decision_context_id": context["context_id"], "revision": workflow["revision"] + 1, "next_action": "submit_decision" if workflow.get("autonomous_rounds") else "submit_initial_decision"}
        if context["phase"] == "final_selection":
            updated.update(stage="budget_exhausted", next_action="submit_final_selection", stop_source="budget")
        return _result(runtime, context, updated)
    except Exception as exc:
        return _error(runtime, exc)


@tool("nir_submit_decision")
def nir_submit_decision_tool(runtime: Runtime, proposal: dict | str):
    """Submit a DecisionProposal object after reading the current context.

    Compare alternatives and select your own full preprocessing/model/parameter
    plan. The server validates it and saves the immutable pre-execution record.
    Errors require an explicit corrected proposal; no default plan is substituted.
    The selected alternative's action must match the top-level action. Only its
    pipeline is trained. Writable model keys are under capabilities.parameters;
    operational_parameters such as random_state/n_jobs are immutable and must
    be omitted from fixed_params/search_space. Compute fits for that single
    selected Cartesian grid, not the list of alternative pipelines.
    """
    try:
        workflow, service = _initial(runtime)
        record = service.submit(workflow["search_session_id"], proposal, runtime.tool_call_id)
        persisted = service.get(record["decision_id"])
        if record.get("action") in {"STOP", "RETAIN"} and (workflow.get("delivery_policy") or {}).get("automatic_delivery"):
            from .automatic_delivery import complete_terminal_delivery

            terminal_workflow = {
                **workflow,
                "retained_experiment_id": record["proposal"]["retained_experiment_id"],
                "stop_decision_id": record["decision_id"],
                "stop_source": persisted["execution"]["stop_source"],
                "active_decision_id": None,
            }
            try:
                payload, updated = complete_terminal_delivery(runtime, service.search, terminal_workflow)
                return _result(
                    runtime,
                    {**record, "automatic_delivery": {"status": "delivered", "model": payload["model"], "report_html": payload["report_html"], "bundle": payload["bundle"], "lifecycle": payload["lifecycle"]}},
                    updated,
                    artifacts=[payload[key] for key in ("model", "report_html", "bundle")],
                )
            except Exception as delivery_error:
                # Acceptance is durable. A publication/Test failure is not an
                # invalid LLM proposal and must not consume its repair counter.
                updated = {**terminal_workflow, "stage": "blocked", "revision": workflow["revision"] + 1, "next_action": "audit_terminal_delivery"}
                failure = {"status": "failed", "error_type": type(delivery_error).__name__, "next_step": "Inspect durable terminal state; uncertain Test/publication leases require audit. A completed Test may only resume publication."}
                return _result(runtime, {**record, "automatic_delivery": failure}, updated)
        if persisted["status"] in {"succeeded", "failed"}:
            return _result(runtime, {**record, "status": persisted["status"], "execution": persisted["execution"], "replayed": True})
        if persisted["status"] == "stopped" and workflow.get("autonomous_delivery"):
            with service.search.process._connection() as db:
                session = service.search._session(db, workflow["search_session_id"])
            if session["stage"] != "stopped":
                return _result(runtime, {**record, "status": "stopped", "execution": persisted["execution"], "replayed": True})
        updated = {
            **workflow,
            "initial_decision_id": workflow.get("initial_decision_id") or record["decision_id"],
            "active_decision_id": record["decision_id"],
            "stage": "execution",
            "revision": workflow["revision"] + 1,
            "next_action": "execute_decision" if workflow.get("autonomous_rounds") else "execute_initial_decision",
        }
        if record.get("action") in {"STOP", "RETAIN"}:
            updated.update(
                stage="awaiting_final_confirmation",
                next_action="freeze_selected_model" if workflow.get("autonomous_delivery") else "await_final_confirmation",
                stop_source=persisted["execution"]["stop_source"],
                retained_experiment_id=record["proposal"]["retained_experiment_id"],
                stop_decision_id=record["decision_id"],
                active_decision_id=None,
            )
        return _result(runtime, record, updated)
    except Exception as exc:
        workflow = (runtime.state or {}).get("nir_workflow") or {}
        if workflow.get("autonomous_delivery") and workflow.get("search_session_id"):
            allowed = record_invalid(_service(runtime), workflow["search_session_id"], runtime.tool_call_id)
            if not allowed:
                search = _service(runtime)
                with search.process._connection() as db:
                    session = search._session(db, workflow["search_session_id"])
                return _error(
                    runtime, exc, rejected_proposal=proposal, closed_workflow={**workflow, "stage": "blocked", "stop_source": session["stop_source"], "next_action": "report_safety_termination", "revision": workflow["revision"] + 1}
                )
        return _error(runtime, exc, rejected_proposal=proposal)


@tool("nir_execute_decision")
def nir_execute_decision_tool(runtime: Runtime, decision_id: str):
    """Execute only an accepted decision_id using the sealed search service.

    No method, preprocessing, parameter overrides or data paths are accepted.
    Multi-round mode returns linked Cal CV/Val feedback and requires a fresh
    context for the next LLM action. Single-round mode waits after one execution.
    """
    try:
        workflow = _workflow(runtime, {"execution", "initial_result"})
        if workflow.get("decision_strategy") != "llm_autonomous" or (workflow.get("active_decision_id") if workflow.get("autonomous_rounds") else workflow.get("initial_decision_id")) != decision_id:
            raise ValueError("decision_id must be the currently accepted decision for this workflow")
        service = InitialDecisionService(_service(runtime))
        feedback = service.execute(decision_id, runtime.tool_call_id, checkpoint=budget_for_runtime(runtime).checkpoint)
        metrics = feedback.get("metrics", {})
        quality = metrics.get("quality", {})
        evidence = {
            "decision_id": decision_id,
            "experiment_id": feedback.get("experiment_id"),
            "tool_name": "nir_execute_decision",
            "method": feedback["actual_configuration"]["method"],
            "protocol": "sealed_single_target_holdout",
            "validation_scope": "independent_holdout_not_external",
            "metrics_summary": {
                "train": metrics.get("train"),
                "train_cv": metrics.get("train_cv"),
                "val": metrics.get("val"),
                "RMSECV": metrics.get("RMSECV"),
                "passed": quality.get("passed"),
                "grade": quality.get("grade"),
                "evaluation_scope": "validation",
                "test_status": "sealed",
            },
        }
        updated = {
            **workflow,
            "stage": "initial_result",
            "revision": workflow["revision"] + 1,
            "next_action": "await_next_phase",
            "initial_execution": {key: feedback.get(key) for key in ("decision_id", "experiment_id", "status", "fits", "elapsed_seconds")},
            "attempt_evidence": evidence,
        }
        if workflow.get("autonomous_rounds"):
            updated["latest_execution"] = updated["initial_execution"]
            updated["initial_execution"] = workflow.get("initial_execution") or updated["initial_execution"]
            with service.search.process._connection() as db:
                session = service.search._session(db, workflow["search_session_id"])
            closed = session["stage"] == "budget_exhausted"
            updated.update(
                stage="budget_exhausted" if closed else "decision_review",
                next_action="report_budget_exhausted" if closed else "read_decision_context",
                active_decision_id=None,
                decision_context_id=None,
                stop_source=session.get("stop_source"),
                retained_experiment_id=session.get("retained_experiment_id"),
            )
            if not any(item.get("decision_id") == decision_id for item in workflow.get("attempts", [])):
                updated["attempts"] = [*workflow.get("attempts", []), evidence][-12:]
            updated["attempt"] = session["experiments"]
            if workflow.get("autonomous_delivery") and closed:
                updated["next_action"] = "read_final_selection_context"
            if workflow.get("autonomous_delivery") and session["stage"] not in {"searching", "budget_exhausted"}:
                updated.update(stage="blocked", next_action="report_search_closed", stop_source=session.get("stop_source"))
        return _result(runtime, feedback, updated)
    except Exception as exc:
        return _error(runtime, exc)


@tool("nir_decision_history")
def nir_decision_history_tool(runtime: Runtime, offset: int = 0, limit: int = 4, experiment_id: str | None = None):
    """Page trusted task-local decision/experiment details (limit 1..4); never Test.

    Current context has compact history. Use this read-only tool for older full
    pipeline configurations, prior hypotheses and execution feedback.
    """
    try:
        workflow = _workflow(runtime, {"execution", "decision_review", "initial_result", "awaiting_final_confirmation", "budget_exhausted"})
        if workflow.get("decision_strategy") != "llm_autonomous":
            raise ValueError("history: requires llm_autonomous")
        return _result(runtime, InitialDecisionService(_service(runtime)).history(workflow["search_session_id"], offset=offset, limit=limit, experiment_id=experiment_id))
    except Exception as exc:
        return _error(runtime, exc)
