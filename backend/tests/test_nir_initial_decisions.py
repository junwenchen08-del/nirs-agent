"""Initial decisions are immutable, issued by model calls, and bound before fit."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from langchain_core.messages import AIMessage
from test_nir_test_sealing import partitions

from deerflow.community.nir.dataset_session import SearchSessionStore
from deerflow.community.nir.decision_context import InitialDecisionService
from deerflow.community.nir.decision_origin import record_model_response
from deerflow.community.nir.process import ProcessStore


def proposal(context, *, family="ridge", steps=None, params=None):
    return {
        "context_id": context["context_id"],
        "expected_revision": context["revision"],
        "context_sha256": context["context_sha256"],
        "phase": "initial",
        "action": "EXPLORE",
        "observations": [{"claim": "Cal sample and feature counts are observed", "evidence_refs": ["profile:calibration"]}],
        "evidence_refs": [context["knowledge"]["results"][0]["evidence_id"]],
        "alternatives": [
            {
                "id": "a",
                "pipeline": {"preprocessing": steps or [], "model_method": family, "fixed_params": params or {"alpha": 2.0}},
                "optimization": {"strategy": "fixed", "search_space": {}, "max_trials": 1, "max_fits": 4},
                "comparison": "Prefer a compact initial hypothesis; alternatives not executed",
            },
            {
                "id": "b",
                "pipeline": {"preprocessing": [], "model_method": "pls", "fixed_params": {"n_components": 2}},
                "optimization": {"strategy": "fixed", "search_space": {}, "max_trials": 1, "max_fits": 4},
                "comparison": "Alternative latent-variable hypothesis is not selected or trained",
            },
        ],
        "selected_alternative_id": "a",
        "rationale": "Start with this declared hypothesis based on Cal and method limitations",
        "expected_outcome": "Obtain independent Val error without claiming prior performance",
    }


def setup(tmp_path):
    process = ProcessStore(tmp_path / "private")
    search = SearchSessionStore(process)
    session = search.prepare("initial", partitions(), cv_folds=3, decision_strategy="llm_autonomous")
    service = InitialDecisionService(search)
    workflow = {"search_protocol": "sealed_holdout", "decision_strategy": "llm_autonomous", "analyte": "protein", "unit": "%", "domain": "default"}
    context = service.context(session["session_id"], workflow, "Choose an initial single-target regression pipeline")
    return service, context


def issue(service, name, args, call_id):
    message = AIMessage(content="", id="model-message-" + call_id, tool_calls=[{"id": call_id, "name": name, "args": args}])
    record_model_response(service.search.process, SimpleNamespace(_llm_type="fake-test", model_name="fake-test"), [message], run_id="test-run")


def submit(service, context, plan, call_id="submit"):
    raw = json.dumps(plan)
    issue(service, "nir_submit_decision", {"proposal": raw}, call_id)
    return service.submit(context["session_id"], raw, call_id)


def test_context_is_real_cal_only_sourced_and_without_experiment_history(tmp_path, monkeypatch):
    original = np.load

    def spy(path, *a, **k):
        assert "holdout" not in str(path)
        return original(path, *a, **k)

    monkeypatch.setattr(np, "load", spy)
    service, context = setup(tmp_path)
    assert context["history"] == []
    assert context["dataset_profile"]["scope"] == "calibration"
    assert context["knowledge"]["results"] and context["knowledge"]["ranking_is_execution_order"] is False
    assert set(context["capabilities"]["models"]) == {"pls", "ridge", "svr", "et"}
    assert "holdout_hash" not in json.dumps(context) and "y_test" not in json.dumps(context)
    assert context["evaluation"]["cv_manifest"]["n_splits"] == 3


def test_forged_author_without_server_model_call_is_rejected(tmp_path):
    service, context = setup(tmp_path)
    with pytest.raises(ValueError, match="model-issued"):
        service.submit(context["session_id"], json.dumps(proposal(context)), "forged")


def test_invalid_parameters_can_be_corrected_without_a_saved_decision(tmp_path):
    service, context = setup(tmp_path)
    with pytest.raises(ValueError, match="alpha"):
        submit(service, context, proposal(context, params={"alpha": -1}), "bad")
    accepted = submit(service, context, proposal(context), "corrected")
    assert accepted["decision_id"]


def test_accept_before_execute_actual_configuration_cost_and_immutable_reason(tmp_path, monkeypatch):
    service, context = setup(tmp_path)
    plan = proposal(context)
    accepted = submit(service, context, plan)
    original = service.search.search

    def spy(*a, **k):
        with service.search.process._connection() as db:
            assert db.execute("SELECT status FROM initial_decisions WHERE id=?", (accepted["decision_id"],)).fetchone()[0] == "executing"
        return original(*a, **k)

    monkeypatch.setattr(service.search, "search", spy)
    issue(service, "nir_execute_decision", {"decision_id": accepted["decision_id"]}, "execute")
    result = service.execute(accepted["decision_id"], "execute")
    assert result["actual_parameters"]["alpha"] == 2.0
    assert result["fits"] == 4 and result["elapsed_seconds"] >= 0
    assert result["decision_id"] == accepted["decision_id"]
    assert result["experiment_id"] == accepted["experiment_id"]
    assert result["metrics"]["val"]["scope"] == "validation"
    saved = service.get(accepted["decision_id"])
    assert saved["proposal"]["rationale"] == plan["rationale"]
    assert saved["origin"]["source"] == "lead_agent_model_call"
    assert saved["execution"]["experiment_id"] == result["experiment_id"]
    issue(service, "nir_execute_decision", {"decision_id": accepted["decision_id"]}, "replay")
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("Replay refit"))
    assert service.execute(accepted["decision_id"], "replay") == result


def test_direct_search_cannot_bypass_accepted_decision(tmp_path):
    service, context = setup(tmp_path)
    with pytest.raises(ValueError, match="decision"):
        service.search.search(context["session_id"], "bypass", method="ridge", steps=[])


def test_context_hash_revision_and_evidence_are_validated(tmp_path):
    service, context = setup(tmp_path)
    plan = proposal(context)
    plan["expected_revision"] += 1
    with pytest.raises(ValueError, match="context"):
        submit(service, context, plan, "stale")
    plan = proposal(context)
    plan["evidence_refs"] = ["fabricated:test-score"]
    with pytest.raises(ValueError, match="evidence"):
        submit(service, context, plan, "fabricated")


def test_autonomous_scope_does_not_inject_cross_task_memory(monkeypatch):
    from langchain_core.messages import HumanMessage

    from deerflow.agents.lead_agent import prompt
    from deerflow.agents.middlewares.dynamic_context_middleware import DynamicContextMiddleware
    from deerflow.agents.middlewares.memory_middleware import MemoryMiddleware

    monkeypatch.setattr(prompt, "_get_memory_context", lambda *a, **k: pytest.fail("Cross-task memory read"))
    state = {"nir_workflow": {"decision_strategy": "llm_autonomous"}, "messages": [HumanMessage(content="Initial modeling")]}
    update = DynamicContextMiddleware()._inject(state)
    assert all("<memory>" not in str(message.content) for message in update["messages"])
    assert MemoryMiddleware().after_agent(state, None) is None


def test_locked_test_mutation_cannot_change_initial_context_or_execution(tmp_path, monkeypatch):
    baseline, a = setup(tmp_path / "baseline")
    changed, b = setup(tmp_path / "changed")
    data = partitions()
    np.savez(changed.search._dir(b["session_id"]) / "holdout.npz", X_test=data["X_test"], y_test=data["y_test"] + 100000)
    original = np.load

    def spy(path, *args, **kwargs):
        assert "holdout" not in str(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", spy)
    repeated = changed.context(b["session_id"], {"search_protocol": "sealed_holdout", "decision_strategy": "llm_autonomous", "analyte": "protein", "unit": "%", "domain": "default"}, "Choose an initial single-target regression pipeline")
    assert repeated == b
    for key in ("dataset_profile", "knowledge", "capabilities", "budget", "evaluation"):
        assert a[key] == b[key]
    results = []
    for service, context in ((baseline, a), (changed, b)):
        accepted = submit(service, context, proposal(context))
        issue(service, "nir_execute_decision", {"decision_id": accepted["decision_id"]}, "execute")
        results.append(service.execute(accepted["decision_id"], "execute"))
    for key in ("metrics", "parameters", "trials", "configuration", "cv_manifest", "fits"):
        assert results[0][key] == results[1][key]


def test_issued_arguments_and_accepted_plan_cannot_be_overridden(tmp_path):
    service, context = setup(tmp_path)
    raw = json.dumps(proposal(context))
    issue(service, "nir_submit_decision", {"proposal": raw}, "submit")
    with pytest.raises(ValueError, match="model-issued"):
        service.submit(context["session_id"], raw + " ", "submit")
    accepted = service.submit(context["session_id"], raw, "submit")
    assert service.get(accepted["decision_id"])["status"] == "accepted"
    with pytest.raises(ValueError, match="decision"):
        service.search.search(context["session_id"], "override", method="svr", steps=[], decision_id=accepted["decision_id"])
    with pytest.raises(ValueError, match="Unknown accepted"):
        issue(service, "nir_execute_decision", {"decision_id": "unaccepted"}, "execute")
        service.execute("unaccepted", "execute")


def test_one_initial_decision_and_concurrent_execution_use_one_lease(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    service, context = setup(tmp_path)
    accepted = submit(service, context, proposal(context))
    with pytest.raises(ValueError, match="one initial"):
        submit(service, context, proposal(context, params={"alpha": 3.0}), "second")
    for call_id in ("execute1", "execute2"):
        issue(service, "nir_execute_decision", {"decision_id": accepted["decision_id"]}, call_id)

    def run(call_id):
        try:
            return service.execute(accepted["decision_id"], call_id)
        except ValueError as exc:
            assert "lease" in str(exc)
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, ("execute1", "execute2")))
    assert any(result and result["status"] == "succeeded" for result in results)
    with service.search.process._connection() as db:
        assert db.execute("SELECT count(*) FROM search_experiments").fetchone()[0] == 1
        assert service.search._session(db, context["session_id"])["fits"] == 4


def test_cv_manifest_tampering_fails_before_fit(tmp_path, monkeypatch):
    service, context = setup(tmp_path)
    accepted = submit(service, context, proposal(context))
    with service.search.process._connection(write=True) as db:
        session = service.search._session(db, context["session_id"])
        session["cv_manifest"]["folds"][0]["train"] = session["cv_manifest"]["folds"][0]["train"][1:]
        service.search._save(db, session)
    from sklearn.linear_model import Ridge

    monkeypatch.setattr(Ridge, "fit", lambda *a, **k: pytest.fail("Fit used altered CV"))
    issue(service, "nir_execute_decision", {"decision_id": accepted["decision_id"]}, "execute")
    result = service.execute(accepted["decision_id"], "execute")
    assert result["status"] == "failed" and result["fits"] == 0
    assert "CV" in result["failure_reason"]
    assert result["experiment_id"] == accepted["experiment_id"]
    assert result["cv_manifest_id"] == accepted["cv_manifest_id"]


@pytest.mark.parametrize(
    "tool_name,args",
    [("nir_search_model", {}), ("nir_train_model", {}), ("nir_train_auto_split_model", {}), ("nir_freeze_search", {}), ("nir_finalize_search", {}), ("nir_workflow", {"action": "start"}), ("task", {}), ("bash", {}), ("read_file", {})],
)
def test_autonomous_mode_denies_old_or_unrecorded_execution(tool_name, args):
    from test_nir_workflow_middleware import _request

    from deerflow.agents.middlewares.nir_workflow_middleware import NIRWorkflowMiddleware
    from deerflow.community.nir.workflow import start_workflow

    workflow = start_workflow(task_type="calibration", search_protocol="sealed_holdout", decision_strategy="llm_autonomous", validation_goal="internal_holdout")
    result = NIRWorkflowMiddleware().wrap_tool_call(_request(tool_name, {"nir_workflow": workflow}, args=args), lambda _: pytest.fail("Bypassed policy"))
    message = result.update["messages"][-1] if hasattr(result, "update") else result
    assert json.loads(message.content)["code"] == "nir_sealed_search_tool_denied"


def test_autonomous_strategy_requires_sealed_protocol():
    from deerflow.community.nir.workflow import start_workflow

    with pytest.raises(ValueError, match="sealed_holdout"):
        start_workflow(task_type="calibration", decision_strategy="llm_autonomous")


def test_cal_val_results_cannot_be_presented_as_test_performance():
    from deerflow.community.nir.response_grounding import validate_nir_response

    workflow = {
        "stage": "initial_result",
        "search_protocol": "sealed_holdout",
        "decision_strategy": "llm_autonomous",
        "attempt_evidence": {"validation_scope": "independent_holdout_not_external", "metrics_summary": {"val": {"R2": 0.9}, "evaluation_scope": "validation", "test_status": "sealed"}},
    }
    result = validate_nir_response("内部验证集 R2 = 0.9，最终测试集 R2 = 0.9", workflow)
    assert "validation_scope_overclaim:sealed_test" in result.violations
    assert "validation_scope_overclaim:sealed_test" not in validate_nir_response("内部验证集 R2 = 0.9，最终测试集尚未评价。", workflow).violations


def test_user_hard_constraints_are_canonical_and_enforced(tmp_path):
    from deerflow.community.nir.workflow import start_workflow

    service, context = setup(tmp_path)
    workflow = start_workflow(task_type="calibration", search_protocol="sealed_holdout", decision_strategy="llm_autonomous", validation_goal="internal_holdout", model_constraint="svr", pipeline_constraint='["snv"]')
    context = service.context(context["session_id"], workflow, "Use SNV and SVR")
    assert json.loads(context["task"]["pipeline_constraint"]) == [{"method": "snv", "params": {}}]
    with pytest.raises(ValueError, match="model_constraint"):
        submit(service, context, proposal(context), "wrong-model")
    with pytest.raises(ValueError, match="pipeline_constraint"):
        submit(service, context, proposal(context, family="svr", params={"C": 2.0}), "wrong-pipeline")
    assert submit(service, context, proposal(context, family="svr", steps=[{"method": "snv", "params": {}}], params={"C": 2.0}), "corrected")["decision_id"]


def test_preprocessing_provider_cannot_change_after_acceptance(tmp_path, monkeypatch):
    monkeypatch.setenv("NIR_PREPROCESSING_PROVIDER_POLICY", "catalog_default")
    service, context = setup(tmp_path)
    accepted = submit(service, context, proposal(context, steps=[{"method": "sg_smooth", "params": {"window": 5, "order": 2}}]))
    monkeypatch.setenv("NIR_PREPROCESSING_PROVIDER_POLICY", "native")
    issue(service, "nir_execute_decision", {"decision_id": accepted["decision_id"]}, "execute")
    result = service.execute(accepted["decision_id"], "execute")
    assert result["status"] == "failed" and result["fits"] == 0
    assert "runtime" in result["failure_reason"]
