"""Atomic decision contracts must survive the production output-budget wrapper."""

import json
from types import SimpleNamespace

import pytest
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import ToolMessage
from langgraph.types import Command
from test_nir_delivery_decisions import setup_delivery

from deerflow.agents.middlewares.nir_workflow_middleware import NIRWorkflowMiddleware
from deerflow.agents.middlewares.tool_output_budget_middleware import ToolOutputBudgetMiddleware
from deerflow.community.nir import decision_tools
from deerflow.config.tool_output_config import ToolOutputConfig


@pytest.mark.parametrize("name", ["nir_decision_context", "nir_submit_decision", "nir_execute_decision", "nir_decision_history", "nir_prepare_search", "nir_freeze_search", "nir_finalize_search"])
def test_sealed_protocol_json_is_atomic_at_tool_and_history_boundaries(tmp_path, name):
    middleware = ToolOutputBudgetMiddleware(config=ToolOutputConfig(exempt_tools=["read_file"]))
    payload = {"context_id": "ctx-actual", "revision": 7, "context_sha256": "actual-digest", "bounded_evidence": "x" * 35000}
    message = ToolMessage(content=json.dumps(payload), name=name, tool_call_id="actual-call")
    result = Command(update={"nir_workflow": {"decision_strategy": "llm_autonomous", "search_protocol": "sealed_holdout"}, "messages": [message]})
    request = SimpleNamespace(tool_call={"name": name}, runtime=SimpleNamespace(state={}, config={}))
    # Atomic protocol messages cannot become a read_file instruction, even when
    # an existing operator config only exempts generic read tools.
    assert middleware.wrap_tool_call(request, lambda _: result) is result
    model_request = ModelRequest(model=None, tools=[], messages=[message], state={})
    seen = []
    middleware.wrap_model_call(model_request, lambda forwarded: seen.append(forwarded) or [])
    assert seen[0] is model_request
    assert json.loads(seen[0].messages[0].content) == payload
    assert not (tmp_path / ".tool-results").exists()


@pytest.mark.anyio
async def test_async_actual_context_keeps_provider_schema_and_binding(tmp_path):
    service, context, workflow = setup_delivery(tmp_path)
    workflow.update(stage="execution", search_session_id=context["session_id"], decision_context_id=context["context_id"])
    message = ToolMessage(content=json.dumps(context), name="nir_decision_context", tool_call_id="context-call")
    assert len(message.content) > 12000  # This is the production spill threshold.
    result = Command(update={"nir_workflow": workflow, "messages": [message]})
    middleware = ToolOutputBudgetMiddleware()
    request = SimpleNamespace(tool_call={"name": message.name}, runtime=SimpleNamespace(state={}, config={}))

    async def tool_handler(_):
        return result

    assert await middleware.awrap_tool_call(request, tool_handler) is result
    model_request = ModelRequest(model=None, tools=[decision_tools.nir_submit_decision_tool], messages=[message], state={"nir_workflow": workflow})
    seen = []

    async def model_handler(forwarded):
        seen.append(NIRWorkflowMiddleware._with_delivery_contract(forwarded))
        return []

    await middleware.awrap_model_call(model_request, model_handler)
    schema = seen[0].tools[0].args["proposal"]
    assert schema["type"] == "object"
    assert schema["properties"]["context_id"]["const"] == context["context_id"]
    assert schema["properties"]["context_sha256"]["const"] == context["context_sha256"]
    assert schema["properties"]["expected_revision"]["const"] == context["revision"]
