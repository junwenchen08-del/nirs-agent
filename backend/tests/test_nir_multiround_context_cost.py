"""Input compression deletes redundant complete pairs, never persisted facts."""

import json
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from deerflow.agents.middlewares.nir_workflow_middleware import NIRWorkflowMiddleware


def test_pruning_keeps_latest_pair_other_tools_and_original_messages():
    messages = [HumanMessage(content="User's original hard requirements")]
    for number, name in enumerate(["nir_decision_context", "nir_submit_decision", "nir_execute_decision", "external_other_tool", "nir_decision_context"]):
        call_id = str(number)
        messages.append(AIMessage(content="", tool_calls=[{"id": call_id, "name": name, "args": {}}]))
        messages.append(ToolMessage(content=json.dumps({"context_id": call_id}) if name == "nir_decision_context" else '{"facts": "unchanged"}', tool_call_id=call_id, name=name))
    messages.append(AIMessage(content="", tool_calls=[{"id": "pending", "name": "nir_submit_decision", "args": {}}]))
    before = [message.model_dump() for message in messages]
    request = SimpleNamespace(messages=messages, state={"nir_workflow": {"autonomous_rounds": True}})
    request.override = lambda **updates: SimpleNamespace(**{**vars(request), **updates})
    compressed = NIRWorkflowMiddleware._compact_decision_messages(request)
    retained_tool_ids = [message.tool_call_id for message in compressed.messages if isinstance(message, ToolMessage)]
    assert retained_tool_ids == ["3", "4"]
    assert len(compressed.messages) < len(messages)
    assert [message.model_dump() for message in messages] == before
    assert compressed.messages[-1].tool_calls[0]["id"] == "pending"
    assert compressed.messages[0].content == messages[0].content


def test_single_round_messages_are_unchanged():
    request = SimpleNamespace(state={"nir_workflow": {"autonomous_rounds": False}})
    assert NIRWorkflowMiddleware._compact_decision_messages(request) is request


def test_delivery_compression_preserves_preparation_and_supplies_current_state():
    messages = [
        HumanMessage(content="Task"),
        AIMessage(content="", tool_calls=[{"id": "prepare", "name": "nir_prepare_search", "args": {}}]),
        ToolMessage(content='{"session_id":"sealed"}', tool_call_id="prepare", name="nir_prepare_search"),
        AIMessage(content="", tool_calls=[{"id": "context", "name": "nir_decision_context", "args": {}}]),
        ToolMessage(content='{"context_id":"ctx"}', tool_call_id="context", name="nir_decision_context"),
    ]
    request = SimpleNamespace(
        messages=messages,
        system_message=SystemMessage(content="Original agent prompt"),
        state={
            "nir_workflow": {
                "autonomous_delivery": True,
                "autonomous_rounds": True,
                "search_protocol": "sealed_holdout",
                "decision_strategy": "llm_autonomous",
                "stage": "execution",
                "search_session_id": "sealed",
                "decision_context_id": "ctx",
            }
        },
    )
    request.override = lambda **updates: SimpleNamespace(**{**vars(request), **updates})
    compact = NIRWorkflowMiddleware._compact_decision_messages(request)
    assert any(isinstance(message, ToolMessage) and message.tool_call_id == "prepare" for message in compact.messages)
    current = NIRWorkflowMiddleware._with_delivery_state(compact)
    assert "submit_decision" in current.system_message.content and '"decision_context_id": "ctx"' in current.system_message.content
    assert request.system_message.content == "Original agent prompt"
    request.state["nir_workflow"]["stage"] = "awaiting_final_confirmation"
    closed = NIRWorkflowMiddleware._compact_decision_messages(request)
    assert not any(isinstance(message, ToolMessage) and message.name == "nir_decision_context" for message in closed.messages)
    assert any(isinstance(message, ToolMessage) and message.tool_call_id == "prepare" for message in closed.messages)
