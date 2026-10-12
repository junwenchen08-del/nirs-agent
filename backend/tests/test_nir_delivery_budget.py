"""Phase-4 budgets are opt-in, immutable and shared across provider requests."""

import pytest
from test_nir_test_sealing import partitions

from deerflow.community.nir.dataset_session import SearchSessionStore
from deerflow.community.nir.process import ProcessStore


def test_profiles_and_server_ceiling(monkeypatch):
    from deerflow.community.nir.decision_budget import resolve_policy

    assert resolve_policy("development")["max_fits"] == 50
    assert resolve_policy("autonomous")["max_experiments"] == 40
    monkeypatch.setenv("NIR_AUTONOMOUS_MAX_FITS", "300")
    with pytest.raises(ValueError, match="max_fits"):
        resolve_policy("autonomous")
    assert resolve_policy("autonomous", {"max_fits": 250})["max_fits"] == 250
    with pytest.raises(ValueError, match="Unknown"):
        resolve_policy("development", {"unlimited": True})


def test_intake_calls_are_durable_and_transferred_without_double_charging(tmp_path):
    from deerflow.community.nir.decision_budget import finish_intake_call, reserve_intake_call, resolve_policy

    process = ProcessStore(tmp_path / "private")
    policy = resolve_policy("development", {"max_llm_calls": 2})
    lease = reserve_intake_call(process, policy, 10)
    finish_intake_call(process, lease, {"input_tokens": 12, "output_tokens": 3}, 0.5)
    lease = reserve_intake_call(ProcessStore(tmp_path / "private"), policy, 20)
    finish_intake_call(process, lease, None, 0.2)
    search = SearchSessionStore(process)
    sid = search.prepare("intake", partitions(), decision_strategy="llm_autonomous", autonomous_rounds=True, delivery_policy=policy)["session_id"]
    with process._connection() as db:
        session = search._session(db, sid)
    assert session["llm_budget"]["search"]["calls"] == 2
    assert session["llm_budget"]["search"]["actual_input_tokens"] == 12
    assert session["llm_budget"]["search"]["unknown_usage_calls"] == 1
    assert session["intake_model_calls"] == 2
    with pytest.raises(ValueError, match="intake"):
        reserve_intake_call(process, policy, 10)


def test_uncertain_intake_cannot_be_replayed_or_prepared(tmp_path):
    from deerflow.community.nir.decision_budget import reserve_intake_call, resolve_policy

    process = ProcessStore(tmp_path / "private")
    policy = resolve_policy()
    reserve_intake_call(process, policy, 10)
    with pytest.raises(ValueError, match="uncertain"):
        reserve_intake_call(ProcessStore(tmp_path / "private"), policy, 10)
    with pytest.raises(ValueError, match="uncertain"):
        SearchSessionStore(process).prepare("uncertain", partitions(), decision_strategy="llm_autonomous", autonomous_rounds=True, delivery_policy=policy)


def test_prepare_binds_large_policy_without_changing_old_limits(tmp_path):
    from deerflow.community.nir.decision_budget import resolve_policy

    store = SearchSessionStore(ProcessStore(tmp_path / "private"))
    policy = resolve_policy("autonomous", {"max_experiments": 15, "max_fits": 500})
    result = store.prepare("new", partitions(), cv_folds=3, decision_strategy="llm_autonomous", autonomous_rounds=True, delivery_policy=policy)
    with store.process._connection() as db:
        session = store._session(db, result["session_id"])
    assert session["max_experiments"] == 15 and session["max_fits"] == 500
    assert store.prepare("new", partitions(), cv_folds=3, decision_strategy="llm_autonomous", autonomous_rounds=True, delivery_policy=policy) == result
    with pytest.raises(ValueError, match="locked"):
        store.prepare("new", partitions(), cv_folds=3, decision_strategy="llm_autonomous", autonomous_rounds=True, delivery_policy=resolve_policy("development"))
    with pytest.raises(ValueError, match="bounds"):
        store.prepare("old", partitions(), max_experiments=15)


def test_provider_call_reservation_cannot_reset_on_restart(tmp_path):
    from deerflow.community.nir.decision_budget import finish_call, reserve_call, resolve_policy

    store = SearchSessionStore(ProcessStore(tmp_path / "private"))
    policy = resolve_policy("development", {"max_llm_calls": 1})
    sid = store.prepare("calls", partitions(), decision_strategy="llm_autonomous", autonomous_rounds=True, delivery_policy=policy)["session_id"]
    lease = reserve_call(store, sid, 10)
    finish_call(store, sid, lease, {"input_tokens": 12, "output_tokens": 3}, 0.5)
    restored = SearchSessionStore(ProcessStore(tmp_path / "private"))
    final = reserve_call(restored, sid, 10)
    with restored.process._connection() as db:
        session = restored._session(db, sid)
    assert session["stage"] == "budget_exhausted" and session["stop_source"] == "budget"
    assert session["llm_budget"]["search"]["calls"] == 1
    assert session["llm_budget"]["finalization"]["calls"] == 1
    finish_call(restored, sid, final, None, 0.2)
    with restored.process._connection() as db:
        session = restored._session(db, sid)
    assert session["llm_budget"]["search"]["actual_input_tokens"] == 12
    assert session["llm_budget"]["finalization"]["unknown_usage_calls"] == 1


def test_parallel_or_uncertain_model_call_fails_closed(tmp_path):
    from deerflow.community.nir.decision_budget import reserve_call, resolve_policy

    store = SearchSessionStore(ProcessStore(tmp_path / "private"))
    sid = store.prepare("uncertain", partitions(), decision_strategy="llm_autonomous", autonomous_rounds=True, delivery_policy=resolve_policy("development"))["session_id"]
    reserve_call(store, sid, 10)
    with pytest.raises(ValueError, match="in-flight"):
        reserve_call(SearchSessionStore(ProcessStore(tmp_path / "private")), sid, 10)


@pytest.mark.parametrize("failure", ["invalid_decision", "provider"])
def test_terminal_failure_preserves_original_budget_stop_source(tmp_path, failure):
    from deerflow.community.nir.decision_budget import finish_call, record_invalid, reserve_call, resolve_policy

    store = SearchSessionStore(ProcessStore(tmp_path / "private"))
    sid = store.prepare("terminal-failure", partitions(), decision_strategy="llm_autonomous", autonomous_rounds=True, delivery_policy=resolve_policy("development", {"max_llm_calls": 1}))["session_id"]
    finish_call(store, sid, reserve_call(store, sid, 10), None, 0.1)
    terminal = reserve_call(store, sid, 10)
    finish_call(store, sid, terminal, None, 0.1, failed=failure == "provider")
    if failure == "invalid_decision":
        for number in range(3):
            record_invalid(store, sid, str(number))
    with store.process._connection() as db:
        session = store._session(db, sid)
    assert session["stage"] == "safety_closed"
    assert session["stop_source"] == "budget" and session["stop_reason"] == ["llm_calls"]
    assert session["terminal_failure_source"] in {"decision_invalid", "tool_failure"}
    assert session["fits"] == 0 and session["freeze"] is None


def test_workflow_new_policy_does_not_expand_legacy():
    from deerflow.community.nir.workflow import start_workflow

    values = dict(task_type="calibration", data_path="/mnt/user-data/uploads/a.npz", validation_goal="internal_holdout", search_protocol="sealed_holdout", decision_strategy="llm_autonomous", autonomous_rounds=True)
    new = start_workflow(**values, autonomous_delivery=True, budget_profile="autonomous")
    assert new["delivery_policy"]["max_experiments"] == 40
    assert start_workflow(**values)["max_attempts"] == 3
    with pytest.raises(ValueError):
        start_workflow(**values, max_attempts=11)


def test_budget_error_reports_exact_grid_and_never_shrinks(tmp_path):
    from test_nir_initial_decisions import proposal, submit

    from deerflow.community.nir.decision_budget import DecisionBudgetError, resolve_policy
    from deerflow.community.nir.decision_context import InitialDecisionService

    store = SearchSessionStore(ProcessStore(tmp_path / "private"))
    sid = store.prepare("grid", partitions(), cv_folds=3, decision_strategy="llm_autonomous", autonomous_rounds=True, delivery_policy=resolve_policy("development"))["session_id"]
    service = InitialDecisionService(store)
    context = service.context(sid, {"decision_strategy": "llm_autonomous", "search_protocol": "sealed_holdout"}, "Choose your own grid")
    plan = proposal(context)
    plan["alternatives"][0]["pipeline"]["fixed_params"] = {}
    plan["alternatives"][0]["optimization"] = {"strategy": "grid", "search_space": {"alpha": list(range(1, 21))}, "max_trials": 20, "max_fits": 61}
    with pytest.raises(DecisionBudgetError) as error:
        submit(service, context, plan)
    assert error.value.details == {"fits_remaining": 50, "cv_folds": 3, "candidates": 20, "required_fits": 61, "max_executable_candidates": 16, "experiments_remaining": 5}
    with store.process._connection() as db:
        assert db.execute("SELECT count(*) FROM initial_decisions").fetchone()[0] == 0
