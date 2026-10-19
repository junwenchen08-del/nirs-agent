"""Feedback actions use persisted experiments and the same sealed executor."""

import copy
import json

import pytest
from test_nir_initial_decisions import issue, proposal, submit
from test_nir_test_sealing import partitions

from deerflow.community.nir.dataset_session import SearchSessionStore
from deerflow.community.nir.decision_context import InitialDecisionService
from deerflow.community.nir.process import ProcessStore


def setup_multi(tmp_path, **limits):
    search = SearchSessionStore(ProcessStore(tmp_path / "private"))
    session = search.prepare("multi", partitions(), cv_folds=3, decision_strategy="llm_autonomous", autonomous_rounds=True, **limits)
    service = InitialDecisionService(search)
    workflow = {"decision_strategy": "llm_autonomous", "search_protocol": "sealed_holdout", "autonomous_rounds": True}
    context = service.context(session["session_id"], workflow, "Decide using feedback")
    return service, context, workflow


def execute(service, record, call="execute"):
    issue(service, "nir_execute_decision", {"decision_id": record["decision_id"]}, call)
    return service.execute(record["decision_id"], call)


def next_plan(context, action="TUNE", family="ridge", steps=None, params=None):
    plan = proposal(context, family=family, steps=steps, params=params or {"alpha": 3.0})
    plan.update(
        phase="feedback",
        action=action,
        base_experiment_id=context["history_state"]["current"]["experiment_id"],
        uncertainties=["Validation variation is uncertain"],
        not_selected_reasons={"b": "Not selected by this LLM response"},
        hypotheses=["This declared change may reduce Val RMSE"],
        expected_fits=4,
        expected_improvement={"metric": "val_rmse", "comparison": "previous", "minimum_change": 0.0},
        difference="This changes the specified pipeline dimension",
    )
    plan["alternatives"][0]["action"] = action
    return plan


@pytest.mark.parametrize(
    "action,family,steps,params",
    [
        ("TUNE", "ridge", [], {"alpha": 3.0}),
        ("CHANGE_PREPROCESS", "ridge", [{"method": "snv", "params": {}}], {"alpha": 2.0}),
        ("SWITCH_MODEL", "svr", [], {"C": 2.0}),
        ("EXPLORE", "et", [{"method": "snv", "params": {}}], {"n_estimators": 7, "max_depth": 3}),
    ],
)
def test_feedback_actions_and_immutable_linkage(tmp_path, action, family, steps, params):
    service, context, workflow = setup_multi(tmp_path)
    first = submit(service, context, proposal(context))
    before = copy.deepcopy(first)
    one = execute(service, first)
    context = service.context(context["session_id"], workflow, "Decide using feedback")
    assert context["phase"] == "feedback"
    assert context["history_state"]["current"]["metrics"]["val"] == one["metrics"]["val"]
    assert context["budget"]["fits_remaining"] == 196
    second = submit(service, context, next_plan(context, action, family, steps, params), "submit-2")
    two = execute(service, second, "execute-2")
    assert two["status"] == "succeeded", two
    assert two["feedback_id"] and two["previous_experiment_id"] == one["experiment_id"]
    assert two["configuration"] == second["accepted_spec"]
    assert two["expected_improvement"]["status"] in {"achieved", "not_achieved"}
    assert service.get(first["decision_id"])["proposal"] == before["proposal"]
    assert service.history(context["session_id"])["total"] == 2


def test_llm_stop_names_retained_success_and_never_loads_test(tmp_path, monkeypatch):
    service, context, workflow = setup_multi(tmp_path)
    first = submit(service, context, proposal(context))
    feedback = execute(service, first)
    context = service.context(context["session_id"], workflow, "Decide using feedback")
    plan = next_plan(context, "STOP")
    plan.update(retained_experiment_id=feedback["experiment_id"], expected_fits=0)
    plan["alternatives"][0].update(pipeline=None, optimization=None)
    import numpy as np

    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("STOP reads data"))
    stopped = submit(service, context, plan, "stop")
    assert service.get(stopped["decision_id"])["status"] == "stopped"
    with service.search.process._connection() as db:
        session = service.search._session(db, context["session_id"])
    assert session["stage"] == "stopped_pending_confirmation" and session["stop_source"] == "llm"
    assert service.search._view(session)["test_status"] == "sealed"


def test_duplicates_illegal_actions_and_parameters_do_not_change_history(tmp_path):
    service, context, workflow = setup_multi(tmp_path)
    invalid_initial = proposal(context)
    invalid_initial["alternatives"][0]["action"] = "TUNE"
    with pytest.raises(ValueError, match="action"):
        submit(service, context, invalid_initial, "invalid-initial-action")
    first = submit(service, context, proposal(context))
    execute(service, first)
    context = service.context(context["session_id"], workflow, "Decide using feedback")
    with pytest.raises(ValueError, match="duplicate"):
        submit(service, context, next_plan(context, params={"alpha": 2.0}), "dup")
    with pytest.raises(ValueError, match="TUNE"):
        submit(service, context, next_plan(context, family="svr", params={"C": 1.0}), "wrong-action")
    with pytest.raises(ValueError, match="alpha"):
        submit(service, context, next_plan(context, params={"alpha": -1.0}), "bad")
    plan = next_plan(context, params={"alpha": 2.0})
    plan["replication_reason"] = "Repeat to verify deterministic provider execution"
    repeat = submit(service, context, plan, "repeat")
    assert execute(service, repeat, "execute-repeat")["status"] == "succeeded"


def test_hard_budget_is_distinct_from_llm_stop_and_does_not_select_best(tmp_path):
    service, context, workflow = setup_multi(tmp_path, max_experiments=2, max_fits=8)
    first = submit(service, context, proposal(context))
    execute(service, first)
    context = service.context(context["session_id"], workflow, "Decide using feedback")
    second = submit(service, context, next_plan(context), "submit-2")
    execute(service, second, "execute-2")
    with service.search.process._connection() as db:
        session = service.search._session(db, context["session_id"])
    assert session["fits"] == 8 and session["experiments"] == 2
    assert session["stage"] == "budget_exhausted" and session["stop_source"] == "hard_budget"
    assert not session.get("retained_experiment_id")
    assert service.search._view(session)["test_status"] == "sealed"


def test_action_budget_is_cumulative_and_invalid_repairs_do_not_spend_it(tmp_path):
    service, context, workflow = setup_multi(tmp_path, max_decisions=2)
    execute(service, submit(service, context, proposal(context)))
    context = service.context(context["session_id"], workflow, "Decide using feedback")
    assert context["budget"]["actions_remaining"] == 1
    with pytest.raises(ValueError, match="alpha"):
        submit(service, context, next_plan(context, params={"alpha": -1.0}), "invalid")
    second = submit(service, context, next_plan(context), "valid")
    execute(service, second, "execute-valid")
    with service.search.process._connection() as db:
        session = service.search._session(db, context["session_id"])
    assert session["stop_reason"] == ["accepted_actions"] and session["fits"] == 8


def test_grid_duplicate_and_equivalent_fixed_plan_cannot_evade_duplicate_guard(tmp_path):
    service, context, workflow = setup_multi(tmp_path)
    execute(service, submit(service, context, proposal(context)))
    context = service.context(context["session_id"], workflow, "Decide using feedback")
    for index, values in enumerate(([2.0], [2], [2.0, 2.0], [2, 2.0])):
        plan = next_plan(context)
        plan["alternatives"][0]["pipeline"]["fixed_params"] = {}
        plan["alternatives"][0]["optimization"] = {"strategy": "grid", "search_space": {"alpha": values}, "max_trials": len(values), "max_fits": len(values) * 3 + 1}
        plan["expected_fits"] = len(values) * 3 + 1
        with pytest.raises(ValueError, match="duplicate"):
            submit(service, context, plan, "duplicate-grid-" + str(index))


def test_history_context_is_bounded_and_sourced(tmp_path):
    service, context, workflow = setup_multi(tmp_path)
    for number in range(5):
        plan = proposal(context) if not number else next_plan(context, params={"alpha": float(number + 2)})
        record = submit(service, context, plan, f"s-{number}")
        execute(service, record, f"e-{number}")
        context = service.context(context["session_id"], workflow, "Decide using feedback")
    assert len(context["history"]) <= 4
    assert context["history_state"]["source"] == "private_persisted_decisions_and_experiments"
    assert len(json.dumps(context)) < 45000
    page = service.history(context["session_id"], offset=1, limit=2)
    assert page["total"] == 5 and len(page["records"]) == 2
    assert "holdout_hash" not in json.dumps(page) and '"test":' not in json.dumps(page)


def test_failed_llm_choice_can_switch_but_user_constraint_cannot(tmp_path, monkeypatch):
    service, context, workflow = setup_multi(tmp_path)
    first = submit(service, context, proposal(context))
    original_search = service.search.search

    def fail(*a, **k):
        raise ValueError("synthetic provider failure")

    monkeypatch.setattr(service.search, "search", fail)
    failed = execute(service, first)
    assert failed["status"] == "failed" and failed["fits"] == 0
    monkeypatch.setattr(service.search, "search", original_search)
    context = service.context(context["session_id"], workflow, "Decide using feedback")
    assert context["history_state"]["failed_count"] == 1
    second = submit(service, context, next_plan(context, "SWITCH_MODEL", "svr", params={"C": 1.0}), "switch")
    assert execute(service, second, "switch-exec")["status"] == "succeeded"
    context = service.context(context["session_id"], {**workflow, "model_constraint": "svr", "pipeline_constraint": "[]"}, "Keep the user's required SVR and Raw")
    with pytest.raises(ValueError, match="model_constraint"):
        submit(service, context, next_plan(context, "SWITCH_MODEL", "ridge"), "forbidden-model")
    with pytest.raises(ValueError, match="pipeline_constraint"):
        submit(service, context, next_plan(context, "CHANGE_PREPROCESS", "svr", [{"method": "snv", "params": {}}], {"C": 2.0}), "forbidden-pipe")


def test_grid_stays_in_authorized_region_across_rounds(tmp_path):
    service, context, workflow = setup_multi(tmp_path)
    execute(service, submit(service, context, proposal(context)))
    context = service.context(context["session_id"], workflow, "Decide using feedback")
    plan = next_plan(context)
    plan["alternatives"][0]["pipeline"]["fixed_params"] = {}
    plan["alternatives"][0]["optimization"] = {"strategy": "grid", "search_space": {"alpha": [0.7, 4.2]}, "max_trials": 2, "max_fits": 7}
    plan["expected_fits"] = 7
    accepted = submit(service, context, plan, "grid")
    feedback = execute(service, accepted, "grid-exec")
    assert feedback["actual_parameters"]["alpha"] in [0.7, 4.2]
    assert feedback["fits"] == 7
    assert {trial["parameters"]["alpha"] for trial in feedback["trials"]} == {0.7, 4.2}
    assert service.context(context["session_id"], workflow, "Decide using feedback")["budget"]["fits_used"] == 11


def test_stale_context_and_inflight_decision_cannot_be_replaced(tmp_path):
    service, context, workflow = setup_multi(tmp_path)
    first = submit(service, context, proposal(context))
    with pytest.raises(ValueError, match="must execute"):
        service.context(context["session_id"], workflow, "Decide using feedback")
    with pytest.raises(ValueError, match="awaiting execution"):
        submit(service, context, proposal(context), "parallel")
    execute(service, first)
    with pytest.raises(ValueError, match="stale"):
        submit(service, context, proposal(context), "old-context")
    assert submit(service, context, proposal(context)) == first  # Exact model-call replay.


def test_on_demand_knowledge_can_restore_cached_context_without_stale_pointer(tmp_path):
    service, original, workflow = setup_multi(tmp_path)
    other = service.context(original["session_id"], workflow, "Decide using feedback", knowledge_query="noise smoothing")
    assert other["context_id"] != original["context_id"]
    restored = service.context(original["session_id"], workflow, "Decide using feedback")
    assert restored["context_id"] == original["context_id"]
    assert submit(service, restored, proposal(restored))["context_id"] == restored["context_id"]


def test_v4_migration_preserves_payloads_and_artifact_bytes(tmp_path):
    import sqlite3

    process = ProcessStore(tmp_path / "private")
    with process._connection() as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 5
    original = '{ "decision_id" : "old", "rationale" : "original bytes" }'
    with sqlite3.connect(process.db) as db:
        db.execute("DROP TABLE initial_decisions")
        db.execute("CREATE TABLE initial_decisions (id TEXT PRIMARY KEY, session_id TEXT NOT NULL UNIQUE, payload TEXT NOT NULL, status TEXT NOT NULL, execution TEXT)")
        db.execute("INSERT INTO initial_decisions VALUES ('old','session',?,'succeeded','{}')", (original,))
        db.execute("PRAGMA user_version=4")
    artifact = process.root / "historical.joblib"
    artifact.write_bytes(b"historical opaque artifact")
    with process._connection(write=True) as db:
        assert db.execute("SELECT payload FROM initial_decisions WHERE id='old'").fetchone()[0] == original
        db.execute("INSERT INTO initial_decisions VALUES ('new','session','{}','accepted',NULL)")
        assert db.execute("PRAGMA user_version").fetchone()[0] == 5
    assert artifact.read_bytes() == b"historical opaque artifact"


@pytest.mark.parametrize("quality", [True, False])
def test_quality_never_selects_an_action_or_closes_the_loop(tmp_path, monkeypatch, quality):
    service, context, workflow = setup_multi(tmp_path)
    import nir_core.utils.metrics

    monkeypatch.setattr(nir_core.utils.metrics, "evaluate_quality", lambda *a, **k: {"passed": quality, "grade": "excellent" if quality else "poor"})
    first = submit(service, context, proposal(context))
    execute(service, first)
    context = service.context(context["session_id"], workflow, "Decide using feedback")
    assert context["history_state"]["current"]["metrics"]["quality"]["passed"] == quality
    chosen = submit(service, context, next_plan(context), "continued-tune")
    assert chosen["action"] == "TUNE" and chosen["accepted_spec"]["method"] == "ridge"


def test_every_round_remains_independent_of_locked_test_targets(tmp_path, monkeypatch):
    import numpy as np

    original_load = np.load

    def guarded(path, *a, **k):
        assert "holdout" not in str(path)
        return original_load(path, *a, **k)

    # Preparation is trusted ingestion; all following context/execution calls
    # are spied. Change only Test y, and drive decisions from actual Val values.
    outcomes = []
    for variant in (1, 100):
        data = partitions()
        data["y_test"] = data["y_test"] * variant
        search = SearchSessionStore(ProcessStore(tmp_path / str(variant)))
        session = search.prepare("multi", data, cv_folds=3, decision_strategy="llm_autonomous", autonomous_rounds=True)
        service = InitialDecisionService(search)
        workflow = {"search_protocol": "sealed_holdout", "decision_strategy": "llm_autonomous", "autonomous_rounds": True}
        with monkeypatch.context() as patcher:
            patcher.setattr(np, "load", guarded)
            context = service.context(session["session_id"], workflow, "Use feedback")
            first = execute(service, submit(service, context, proposal(context)))
            context = service.context(session["session_id"], workflow, "Use feedback")
            alpha = 3.0 + first["metrics"]["val"]["RMSE"]
            second = execute(service, submit(service, context, next_plan(context, params={"alpha": alpha}), "s2"), "e2")
            outcomes.append((alpha, first["metrics"]["val"], second["metrics"]["val"], second["actual_parameters"]))
    assert outcomes[0] == outcomes[1]


@pytest.mark.parametrize("stage", ["execution", "decision_review", "awaiting_final_confirmation", "budget_exhausted"])
@pytest.mark.parametrize("tool", ["nir_search_model", "nir_freeze_search", "nir_finalize_search", "nir_train_model", "chemotools_fit_estimator", "read_file", "bash", "task", "nir_workflow"])
def test_new_loop_cannot_bypass_record_or_test_sealing(stage, tool):
    from langgraph.types import Command
    from test_nir_workflow_middleware import _request

    from deerflow.agents.middlewares.nir_workflow_middleware import NIRWorkflowMiddleware
    from deerflow.community.nir.workflow import start_workflow

    workflow = start_workflow(
        task_type="calibration",
        data_path="/mnt/user-data/uploads/data.npz",
        domain="default",
        analyte="protein",
        unit="%",
        validation_goal="internal_holdout",
        search_protocol="sealed_holdout",
        decision_strategy="llm_autonomous",
        autonomous_rounds=True,
    )
    workflow["stage"] = stage
    called = []
    request = _request(tool, {"nir_workflow": workflow})
    request.tool_call["args"] = {"action": "start"} if tool == "nir_workflow" else {}
    result = NIRWorkflowMiddleware().wrap_tool_call(request, lambda request: called.append(request))
    assert not called
    message = result.update["messages"][-1] if isinstance(result, Command) else result
    assert json.loads(message.content)["code"] == "nir_sealed_search_tool_denied"
