"""Governed NIR contexts must not trigger unbudgeted auxiliary LLM summaries."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from deerflow.agents.middlewares.summarization_middleware import DeerFlowSummarizationMiddleware


def fixture(mode):
    model = MagicMock()
    model.with_config.return_value = model
    model.invoke.return_value = SimpleNamespace(text="ordinary conversation summary")
    middleware = DeerFlowSummarizationMiddleware(model=model, trigger=("messages", 2), keep=("messages", 1), token_counter=len)
    state = {"messages": [HumanMessage(content="hard user constraints"), AIMessage(content="older response"), HumanMessage(content="current request")]}
    context = {}
    if mode == "web_intake":
        context["nir_web_request"] = {"decision_strategy": "llm_autonomous", "search_protocol": "sealed_holdout"}
    elif mode == "active":
        state["nir_workflow"] = {"decision_strategy": "llm_autonomous", "search_protocol": "sealed_holdout"}
    else:
        state["nir_workflow"] = {"decision_strategy": "legacy"}
    return middleware, model, state, SimpleNamespace(context=context)


@pytest.mark.parametrize("mode", ["web_intake", "active"])
def test_governed_nir_preserves_evidence_without_summary_provider(mode):
    middleware, model, state, runtime = fixture(mode)
    before = [message.model_dump() for message in state["messages"]]
    assert middleware.before_model(state, runtime) is None
    model.invoke.assert_not_called()
    assert [message.model_dump() for message in state["messages"]] == before


@pytest.mark.parametrize("mode", ["web_intake", "active"])
@pytest.mark.asyncio
async def test_async_governed_nir_preserves_evidence_without_summary_provider(mode):
    middleware, model, state, runtime = fixture(mode)
    before = [message.model_dump() for message in state["messages"]]
    assert await middleware.abefore_model(state, runtime) is None
    model.ainvoke.assert_not_called()
    assert [message.model_dump() for message in state["messages"]] == before


def test_legacy_summary_remains_available():
    middleware, model, state, runtime = fixture("legacy")
    result = middleware.before_model(state, runtime)
    model.invoke.assert_called_once()
    assert result["summary_text"] == "ordinary conversation summary"


@pytest.mark.asyncio
async def test_async_legacy_summary_remains_available():
    from unittest.mock import AsyncMock

    middleware, model, state, runtime = fixture("legacy")
    model.ainvoke = AsyncMock(return_value=SimpleNamespace(text="ordinary conversation summary"))
    result = await middleware.abefore_model(state, runtime)
    model.ainvoke.assert_awaited_once()
    assert result["summary_text"] == "ordinary conversation summary"
