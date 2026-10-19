"""Normal STOP and resource-terminal retention are distinct LLM decisions."""

import copy

import pytest
from test_nir_initial_decisions import proposal, submit
from test_nir_multiround_decisions import execute, next_plan
from test_nir_test_sealing import partitions

from deerflow.community.nir.dataset_session import SearchSessionStore
from deerflow.community.nir.decision_budget import resolve_policy
from deerflow.community.nir.decision_context import InitialDecisionService
from deerflow.community.nir.process import ProcessStore


def setup_delivery(tmp_path, **budgets):
    search = SearchSessionStore(ProcessStore(tmp_path / "private"))
    workflow = {"decision_strategy": "llm_autonomous", "search_protocol": "sealed_holdout", "autonomous_rounds": True, "autonomous_delivery": True}
    sid = search.prepare("delivery", partitions(), cv_folds=3, decision_strategy="llm_autonomous", autonomous_rounds=True, delivery_policy=resolve_policy("development", budgets), task_snapshot=workflow)["session_id"]
    service = InitialDecisionService(search)
    return service, service.context(sid, workflow, "Model autonomously and deliver"), workflow


def stop_plan(context, retained, action="STOP"):
    fixture_context = {**context, "knowledge": {"results": [{"evidence_id": "history:session"}]}} if context["phase"] == "final_selection" else context
    plan = next_plan(fixture_context, action)
    plan.update(
        phase=context["phase"],
        retained_experiment_id=retained,
        expected_fits=0,
        stop_evidence={
            "retained_advantages": ["This fitted model is simple"],
            "retained_limitations": ["Small synthetic sample"],
            "search_coverage": "Only persisted pipelines were trained",
            "unverified_directions": [{"direction": "More complex nonlinear models", "reason": "Not worth the declared cost for this task"}],
        },
    )
    plan["evidence_refs"].append("experiment:" + retained)
    plan["alternatives"][0].update(pipeline=None, optimization=None)
    return plan


def test_early_stop_can_retain_earlier_success_without_rewriting(tmp_path):
    service, ctx, wf = setup_delivery(tmp_path)
    first = submit(service, ctx, proposal(ctx))
    before = copy.deepcopy(first)
    feedback = execute(service, first)
    ctx = service.context(ctx["session_id"], wf, "Continue or stop")
    stopped = submit(service, ctx, stop_plan(ctx, feedback["experiment_id"]), "stop")
    with service.search.process._connection() as db:
        session = service.search._session(db, ctx["session_id"])
    assert session["stage"] == "stopped" and session["stop_source"] == "llm"
    assert session["fits"] == 4 and session["max_fits"] == 50
    assert session["retained_selection_source"] == "llm"
    assert stopped["stop_budget_snapshot"]["fits_remaining"] == 46
    assert service.get(first["decision_id"])["proposal"] == before["proposal"]
    assert submit(service, ctx, stop_plan(ctx, feedback["experiment_id"]), "stop") == stopped


def test_budget_terminal_choice_reads_only_history_and_costs_zero_fits(tmp_path, monkeypatch):
    service, ctx, wf = setup_delivery(tmp_path, max_experiments=1, max_fits=4)
    first = submit(service, ctx, proposal(ctx))
    result = execute(service, first)
    import numpy as np

    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("Terminal context must use persisted evidence"))
    ctx = service.context(ctx["session_id"], wf, "Retain a successful model")
    assert ctx["phase"] == "final_selection" and ctx["budget"]["stop_source"] == "budget"
    with pytest.raises(ValueError, match="final_selection"):
        submit(service, ctx, stop_plan(ctx, result["experiment_id"], "TUNE"), "illegal-tune")
    kept = submit(service, ctx, stop_plan(ctx, result["experiment_id"], "RETAIN"), "retain")
    with service.search.process._connection() as db:
        session = service.search._session(db, ctx["session_id"])
    assert session["stage"] == "stopped" and session["stop_source"] == "budget"
    assert session["fits"] == 4 and session["retained_experiment_id"] == result["experiment_id"]
    assert kept["action"] == "RETAIN"


@pytest.mark.parametrize("action,family,steps", [("TUNE", "ridge", []), ("CHANGE_PREPROCESS", "ridge", [{"method": "snv", "params": {}}]), ("SWITCH_MODEL", "svr", [])])
def test_three_nonimprovements_do_not_determine_next_action(tmp_path, action, family, steps):
    service, ctx, wf = setup_delivery(tmp_path, max_experiments=10)
    first = None
    for index in range(4):
        plan = proposal(ctx, params={"alpha": 1.0}) if index == 0 else next_plan(ctx, params={"alpha": 1.0})
        if index:
            plan["replication_reason"] = "Verify the identical deterministic experiment"
        result = execute(service, submit(service, ctx, plan, f"s-{index}"), f"e-{index}")
        first = first or result["experiment_id"]
        ctx = service.context(ctx["session_id"], wf, "Decide independently of a plateau")
    assert ctx["budget"]["experiments_remaining"] == 6
    plan = next_plan(ctx, action, family, steps, {"alpha": 2.0} if family == "ridge" else {"C": 2.0})
    assert execute(service, submit(service, ctx, plan, "next"), "next-exec")["status"] == "succeeded"
    ctx = service.context(ctx["session_id"], wf, "Stop and retain the older experiment for simplicity")
    assert submit(service, ctx, stop_plan(ctx, first), "stop")["proposal"]["retained_experiment_id"] == first


def test_more_than_ten_experiments_share_large_budget(tmp_path):
    service, ctx, wf = setup_delivery(tmp_path, max_experiments=12, max_fits=300, max_decisions=30)
    for index in range(11):
        plan = proposal(ctx, params={"alpha": float(index + 1)}) if not index else next_plan(ctx, params={"alpha": float(index + 1)})
        assert execute(service, submit(service, ctx, plan, f"s-{index}"), f"e-{index}")["status"] == "succeeded"
        ctx = service.context(ctx["session_id"], wf, "Choose a next action")
    assert ctx["budget"]["fits_used"] == 44 and ctx["budget"]["fits_remaining"] == 256


def test_all_failed_budget_terminal_has_no_retained_model(tmp_path, monkeypatch):
    service, ctx, wf = setup_delivery(tmp_path, max_decisions=1)
    monkeypatch.setattr(service.search, "search", lambda *a, **k: (_ for _ in ()).throw(ValueError("synthetic failure")))
    assert execute(service, submit(service, ctx, proposal(ctx)))["status"] == "failed"
    with service.search.process._connection() as db:
        session = service.search._session(db, ctx["session_id"])
    assert session["stage"] == "no_deliverable_model" and not session.get("freeze")
    assert session["stop_source"] == "budget"


@pytest.mark.parametrize("stop_now", [True, False])
def test_good_quality_allows_both_stop_and_continuation(tmp_path, monkeypatch, stop_now):
    import nir_core.utils.metrics

    monkeypatch.setattr(nir_core.utils.metrics, "evaluate_quality", lambda *a, **k: {"passed": True, "grade": "excellent"})
    service, ctx, wf = setup_delivery(tmp_path)
    result = execute(service, submit(service, ctx, proposal(ctx)))
    ctx = service.context(ctx["session_id"], wf, "Decide from good Val evidence")
    assert ctx["history_state"]["current"]["metrics"]["quality"]["passed"] is True
    plan = stop_plan(ctx, result["experiment_id"]) if stop_now else next_plan(ctx)
    accepted = submit(service, ctx, plan, "follow-up")
    assert accepted["action"] == ("STOP" if stop_now else "TUNE")
