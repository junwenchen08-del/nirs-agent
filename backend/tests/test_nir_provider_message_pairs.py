"""Provider projection must preserve call/result pairing after malformed JSON."""

import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deerflow.agents.middlewares.dangling_tool_call_middleware import DanglingToolCallMiddleware
from deerflow.agents.middlewares.nir_workflow_middleware import NIRWorkflowMiddleware
from deerflow.community.nir.response_grounding import render_grounded_nir_response
from deerflow.models.patched_deepseek import PatchedChatDeepSeek


def assert_provider_pairs(messages):
    """Validate the actual DeepSeek adapter payload without making an API call."""
    payload = PatchedChatDeepSeek(model="deepseek-chat", api_key="unused-local-test")._get_request_payload(messages)
    pending = set()
    for message in payload["messages"]:
        if message["role"] == "tool":
            assert message["tool_call_id"] in pending, f"Orphan provider tool result: {message['tool_call_id']}"
            pending.remove(message["tool_call_id"])
        else:
            assert not pending, f"Tool calls interrupted before their results: {pending}"
            if message["role"] == "assistant":
                pending = {call["id"] for call in message.get("tool_calls", [])}
    assert not pending
    return payload


def project(messages, *, autonomous=True):
    recovered = DanglingToolCallMiddleware()._build_patched_messages(messages) or messages
    request = SimpleNamespace(messages=recovered, state={"nir_workflow": {"autonomous_rounds": autonomous}})
    request.override = lambda **updates: SimpleNamespace(**{**vars(request), **updates})
    return NIRWorkflowMiddleware._compact_decision_messages(request).messages


def context_pair():
    return [
        AIMessage(content="", tool_calls=[{"id": "context", "name": "nir_decision_context", "args": {}}]),
        ToolMessage(content=json.dumps({"context_id": "ctx", "phase": "feedback"}), name="nir_decision_context", tool_call_id="context"),
    ]


@pytest.mark.parametrize("raw_payload", [False, True])
@pytest.mark.parametrize("has_context", [False, True])
def test_invalid_call_and_recovery_result_are_projected_together(raw_payload, has_context):
    invalid = AIMessage(
        content="Resubmit a decision.",
        invalid_tool_calls=[{"id": "broken", "name": "nir_submit_decision", "args": '{"proposal":', "error": "incomplete JSON"}],
        additional_kwargs={"tool_calls": [{"id": "broken", "type": "function", "function": {"name": "nir_submit_decision", "arguments": '{"proposal":'}}]} if raw_payload else {},
    )
    reminder = HumanMessage(content="Correct the JSON and resubmit.", name="nir_workflow_completion_reminder", additional_kwargs={"hide_from_ui": True, "nir_invalid_tool_call": True})
    messages = [HumanMessage(content="Build the uploaded dataset."), *(context_pair() if has_context else []), invalid, reminder]
    original = [message.model_dump() for message in messages]
    projected = project(messages)
    payload = assert_provider_pairs(projected)
    assert not any(isinstance(message, ToolMessage) and message.tool_call_id == "broken" for message in projected)
    assert not any(isinstance(message, AIMessage) and message.invalid_tool_calls for message in projected)
    assert "Previous tool response contained invalid JSON" in json.dumps(payload)
    assert projected[-1] is reminder
    assert [message.model_dump() for message in messages] == original


def test_mixed_valid_and_invalid_calls_keep_valid_results_and_recovery_pairs():
    message = AIMessage(
        content="",
        tool_calls=[{"id": "valid", "name": "nir_decision_history", "args": {}}],
        invalid_tool_calls=[{"id": "broken", "name": "nir_submit_decision", "args": "{", "error": None}],
    )
    result = ToolMessage(content='{"history":[]}', name="nir_decision_history", tool_call_id="valid")
    messages = [*context_pair(), message, result]
    projected = project(messages)
    assert_provider_pairs(projected)
    assert result in projected and message in projected
    assert message.invalid_tool_calls and message.tool_calls


def test_non_autonomous_invalid_call_keeps_existing_recovery_contract():
    message = AIMessage(content="", invalid_tool_calls=[{"id": "bad", "name": "write_file", "args": "{", "error": None}])
    projected = project([message], autonomous=False)
    assert_provider_pairs(projected)
    assert projected[0] is message
    assert projected[1].status == "error"


def test_same_transport_id_in_another_valid_batch_retains_its_result():
    valid = AIMessage(content="", tool_calls=[{"id": "shared", "name": "external_read", "args": {}}])
    result = ToolMessage(content="Recorded fact", name="external_read", tool_call_id="shared")
    invalid = AIMessage(content="", invalid_tool_calls=[{"id": "shared", "name": "nir_submit_decision", "args": "{", "error": None}])
    projected = project([valid, result, *context_pair(), invalid])
    assert_provider_pairs(projected)
    assert result in projected


@pytest.mark.parametrize(
    "source,reason,expected",
    [
        ("tool_failure", "llm: session is closed or requires operator recovery", "模型服务调用失败"),
        ("decision_invalid", "invalid_decision_repair_limit", "决策内容未能正确解析"),
        ("safety", "unknown admission failure", "执行异常"),
        ("budget", "max_fits", "运行限制"),
        ("token_budget", "total limit", "运行限制"),
        ("loop_detection", "same context", "操作重复"),
    ],
)
def test_stopped_response_distinguishes_provider_errors_from_limits(source, reason, expected):
    workflow = {"stage": "blocked", "search_protocol": "sealed_holdout", "decision_strategy": "llm_autonomous", "runtime_termination": {"source": source, "reason": reason}}
    rendered = render_grounded_nir_response(workflow)
    assert expected in rendered
    assert "已有结果已保留" in rendered
    if source in {"tool_failure", "decision_invalid", "safety"}:
        assert "运行限制" not in rendered
    assert reason not in rendered
