"""The web selection is a validated execution mode, independent of prose."""

import json
from types import SimpleNamespace

import pytest

from deerflow.community.nir.web_entry import bind_web_request, web_request


def body(value, **kwargs):
    return SimpleNamespace(context={"nir_autonomous": value}, config=kwargs.get("config"), input=kwargs.get("input"))


def test_natural_web_request_binds_all_four_mode_flags():
    config = {}
    bind_web_request(config, body({"enabled": True, "profile": "development"}))
    selected = web_request(config["context"])
    assert selected["search_protocol"] == "sealed_holdout"
    assert selected["decision_strategy"] == "llm_autonomous"
    assert selected["autonomous_rounds"] and selected["autonomous_delivery"]
    assert selected["delivery_policy"]["final_llm_calls"] >= 4


@pytest.mark.parametrize(
    "value",
    [
        {"enabled": True, "profile": "other"},
        {"enabled": True, "budget": {"max_fits": 0}},
        {"enabled": True, "budget": {"max_fits": 1001}},
        {"enabled": True, "budget": {"final_llm_calls": 100}},
        {"enabled": True, "author": "llm"},
    ],
)
def test_web_entry_rejects_invalid_or_unexposed_budget(value):
    with pytest.raises(ValueError):
        bind_web_request({}, body(value))


def test_web_request_cannot_inject_workflow_state():
    with pytest.raises(ValueError, match="nir_workflow"):
        bind_web_request({}, body({"enabled": True}, input={"nir_workflow": {"stage": "frozen"}}))


def test_standard_chat_cannot_forge_server_owned_workflow():
    with pytest.raises(ValueError, match="nir_workflow"):
        bind_web_request({}, SimpleNamespace(context={}, input={"nir_workflow": {"revision": 999, "approval_status": "approved", "decision_strategy": "legacy"}}))


def test_config_cannot_forge_internal_web_binding():
    config = {"context": {"nir_web_request": {"enabled": True}}}
    bind_web_request(config, SimpleNamespace(context={}, input=None))
    assert web_request(config["context"]) is None


def test_custom_budget_is_resolved_by_existing_policy():
    config = {}
    bind_web_request(config, body({"enabled": True, "profile": "custom", "budget": {"max_fits": 8, "max_experiments": 2}}))
    request = web_request(config["context"])
    assert request["delivery_policy"]["max_fits"] == 8
    assert json.loads(request["task_budget"])["max_experiments"] == 2


def test_custom_web_budget_uses_server_bounded_autonomous_llm_allowance(monkeypatch):
    monkeypatch.setenv("NIR_AUTONOMOUS_MAX_LLM_CALLS", "36")
    monkeypatch.setenv("NIR_AUTONOMOUS_MAX_LLM_TOKENS", "2700000")
    config = {}
    bind_web_request(config, body({"enabled": True, "profile": "custom", "budget": {"max_experiments": 8, "max_fits": 160, "max_decisions": 24, "max_wall_seconds": 1800}}))
    request = web_request(config["context"])
    assert request["ui_profile"] == "custom"
    assert request["delivery_policy"]["max_llm_calls"] == 36
    assert request["delivery_policy"]["max_llm_tokens"] == 2700000
    assert request["delivery_policy"]["final_llm_tokens"] == 500000
    assert json.loads(request["task_budget"])["max_fits"] == 160
    config = {}
    bind_web_request(config, body({"enabled": True, "profile": "development"}))
    assert web_request(config["context"])["delivery_policy"]["max_llm_tokens"] == 1200000


def test_legacy_has_no_autonomous_binding():
    config = {}
    bind_web_request(config, SimpleNamespace(context={"mode": "flash"}, input=None))
    assert web_request(config.get("context")) is None


def test_web_mode_blocks_legacy_before_start_without_prompt_flags():
    from deerflow.agents.middlewares.nir_workflow_middleware import _authorize

    config = {}
    bind_web_request(config, body({"enabled": True}))
    request = SimpleNamespace(runtime=SimpleNamespace(context=config["context"], state={"messages": []}), tool_call={"name": "nir_train_model", "id": "bad", "args": {}}, state={"messages": []})
    assert json.loads(_authorize(request).content)["code"] == "nir_delivery_mode_required"


def test_web_mode_cannot_upgrade_or_bypass_existing_phase_one_session():
    from deerflow.agents.middlewares.nir_workflow_middleware import _authorize

    config = {}
    bind_web_request(config, body({"enabled": True}))
    request = SimpleNamespace(
        runtime=SimpleNamespace(context=config["context"], state=None),
        tool_call={"name": "nir_search_model", "id": "bad", "args": {"method": "ridge"}},
        state={"nir_workflow": {"search_protocol": "sealed_holdout", "decision_strategy": "legacy", "stage": "execution"}},
    )
    assert json.loads(_authorize(request).content)["code"] == "nir_web_mode_conflict"


def test_workflow_tool_enforces_server_bound_policy(monkeypatch):
    from deerflow.community.nir.workflow import nir_workflow_tool

    config = {}
    bind_web_request(config, body({"enabled": True, "profile": "custom", "budget": {"max_fits": 8}}))
    runtime = SimpleNamespace(state={}, context=config["context"], tool_call_id="start", config={})
    command = nir_workflow_tool.func(runtime=runtime, action="start", task_type="calibration", data_path="/mnt/user-data/uploads/x.csv", analyte="protein", unit="%", domain="default", validation_goal="internal_holdout")
    workflow = command.update["nir_workflow"]
    assert workflow["decision_strategy"] == "llm_autonomous" and workflow["search_protocol"] == "sealed_holdout"
    assert workflow["autonomous_rounds"] and workflow["autonomous_delivery"]
    assert workflow["delivery_policy"]["max_fits"] == 8


def test_first_web_turn_does_not_load_cross_task_memory(monkeypatch):
    from langchain_core.messages import HumanMessage

    from deerflow.agents.middlewares.dynamic_context_middleware import DynamicContextMiddleware

    config = {}
    bind_web_request(config, body({"enabled": True}))
    middleware = DynamicContextMiddleware()
    monkeypatch.setattr(middleware, "_build_full_reminder", lambda: pytest.fail("web autonomous intake must not read memory"))
    result = middleware.before_agent({"messages": [HumanMessage(content="请自主建模", id="h1")]}, SimpleNamespace(context=config["context"]))
    assert not any("__memory" in str(message.id) for message in result["messages"])


def test_first_web_turn_removes_preexisting_memory_before_provider():
    from langchain_core.messages import HumanMessage

    from deerflow.agents.middlewares.nir_workflow_middleware import NIRWorkflowMiddleware

    config = {}
    bind_web_request(config, body({"enabled": True}))
    messages = [HumanMessage(content="<memory>prior task</memory>", additional_kwargs={"dynamic_context_reminder": True}), HumanMessage(content="预测蛋白")]
    request = SimpleNamespace(state={}, messages=messages, runtime=SimpleNamespace(context=config["context"]))
    request.override = lambda **kwargs: SimpleNamespace(**kwargs)
    assert NIRWorkflowMiddleware._without_autonomous_memory(request).messages == messages[1:]
