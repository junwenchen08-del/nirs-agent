"""New policies disable cumulative token caps explicitly; old policies stay finite."""

from types import SimpleNamespace

import pytest
from test_nir_test_sealing import partitions

from deerflow.community.nir.dataset_session import SearchSessionStore
from deerflow.community.nir.decision_budget import finish_call, reserve_call, resolve_policy
from deerflow.community.nir.process import ProcessStore
from deerflow.community.nir.web_entry import bind_web_request


def test_unified_policy_is_explicitly_uncapped_and_server_bounded(monkeypatch):
    monkeypatch.setenv("NIR_AUTONOMOUS_MAX_FITS", "160")
    monkeypatch.setenv("NIR_AUTONOMOUS_MAX_LLM_CALLS", "36")
    policy = resolve_policy("unified")
    assert policy["max_llm_tokens"] is None and policy["final_llm_tokens"] is None
    assert policy["max_fits"] == 160 and policy["max_llm_calls"] == 36
    assert policy["automatic_delivery"] is True
    assert policy["token_enforcement"] == "statistics_only"
    assert resolve_policy("development")["max_llm_tokens"] == 1_200_000


def test_huge_token_usage_does_not_end_new_search(tmp_path):
    search = SearchSessionStore(ProcessStore(tmp_path / "private"))
    policy = resolve_policy("unified", {"max_llm_calls": 2})
    sid = search.prepare("uncapped", partitions(), decision_strategy="llm_autonomous", autonomous_rounds=True, delivery_policy=policy)["session_id"]
    lease = reserve_call(search, sid, 10**12)
    finish_call(search, sid, lease, {"input_tokens": 10**12, "output_tokens": 10**9}, 0.1)
    second = reserve_call(search, sid, 10**12)
    finish_call(search, sid, second, None, 0.1)
    with search.process._connection() as db:
        session = search._session(db, sid)
    assert session["stage"] == "searching" and session.get("stop_source") is None
    assert session["llm_budget"]["search"]["actual_input_tokens"] == 10**12
    # Calls remain a real safety limit, independent of token amounts.
    terminal = reserve_call(search, sid, 10**12)
    with search.process._connection() as db:
        session = search._session(db, sid)
    assert session["stop_source"] == "budget" and session["stop_reason"] == ["llm_calls"]
    finish_call(search, sid, terminal, {"input_tokens": 10**12, "output_tokens": 3}, 0.1)


def test_new_web_chat_gets_pending_intent_without_starting_workflow():
    config = {"context": {"nir_unified_entry": {"forged": True}}}
    body = SimpleNamespace(context={}, input={"messages": [{"role": "user", "content": "什么是 SNV？"}]}, assistant_id="lead_agent")
    bind_web_request(config, body)
    assert config["context"]["nir_unified_entry"] == {"schema_version": 1, "intent": "pending"}
    assert "nir_web_request" not in config["context"] and "nir_workflow" not in config


@pytest.mark.parametrize("field", ["max_llm_tokens", "final_llm_tokens"])
def test_old_policy_still_requires_explicit_positive_token_cap(field):
    with pytest.raises(ValueError, match=field):
        resolve_policy("development", {field: None})


def test_unified_policy_cannot_enable_a_token_cap_by_request():
    with pytest.raises(ValueError, match="statistics"):
        resolve_policy("unified", {"max_llm_tokens": 100})


def test_default_gateway_ignores_old_or_forged_mode_budgets():
    config = {"context": {"nir_web_request": {"forged": True}}}
    body = SimpleNamespace(assistant_id="lead_agent", input={"messages": [{"role": "user", "content": "请建模"}]}, context={"nir_autonomous": {"enabled": True, "budget": {"max_fits": 10**9}}})
    bind_web_request(config, body)
    assert config["context"] == {"nir_unified_entry": {"schema_version": 1, "intent": "pending"}}


def test_provider_cost_is_actual_scoped_and_never_invented():
    from deerflow.community.nir.decision_provider import runtime_provider_cost
    from deerflow.runtime.journal import RunJournal

    journal = RunJournal("owned-run", "owned-thread", SimpleNamespace())
    journal._total_input_tokens = 120
    journal._total_output_tokens = 5
    journal._total_tokens = 125
    journal._llm_call_count = 2
    journal._lead_agent_tokens = 100
    journal._middleware_tokens = 25
    runtime = SimpleNamespace(config={"configurable": {"thread_id": "owned-thread"}, "callbacks": [journal]}, context={"run_id": "owned-run"})
    cost = runtime_provider_cost(runtime)
    assert cost["total_tokens"] == 125 and cost["middleware_tokens"] == 25
    assert cost["usage_kind"] == "actual_provider_usage" and "until_deterministic_publication" in cost["scope"]
    assert "last_ai_message" not in cost
    runtime.context["run_id"] = "other-run"
    assert runtime_provider_cost(runtime) is None


def test_custom_agent_keeps_existing_entry():
    config = {"context": {"agent_name": "research"}}
    body = SimpleNamespace(assistant_id="lead_agent", input={"messages": [{"role": "user", "content": "research"}]}, context={})
    bind_web_request(config, body)
    assert config["context"] == {"agent_name": "research"}


def test_pending_questions_keep_summary_but_started_intake_skips_auxiliary_provider():
    from langchain_core.messages import AIMessage, HumanMessage

    from deerflow.agents.middlewares.summarization_middleware import DeerFlowSummarizationMiddleware

    runtime = SimpleNamespace(context={"nir_unified_entry": {"schema_version": 1, "intent": "pending"}})
    state = {"messages": [HumanMessage(content="什么是 SNV")]}
    assert not DeerFlowSummarizationMiddleware._governed_nir_context(state, runtime)
    state["messages"].append(AIMessage(content="", tool_calls=[{"id": "actual-start", "name": "nir_workflow", "args": {"action": "start", "task_type": "calibration"}}]))
    assert DeerFlowSummarizationMiddleware._governed_nir_context(state, runtime)
