"""Runtime limits must end NIR execution without selecting or testing a model."""

import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from test_nir_delivery_agent import make_delivery_lead
from test_nir_unified_agent import UnifiedFakeLLM

from deerflow.agents.middlewares.loop_detection_middleware import LoopDetectionMiddleware
from deerflow.agents.middlewares.nir_workflow_middleware import NIRWorkflowMiddleware
from deerflow.community.nir import search_tools
from deerflow.community.nir.process import runtime_store
from deerflow.community.nir.user_wait import waiting_for_user
from deerflow.community.nir.web_entry import bind_web_request, stored_workflow
from deerflow.config import paths
from deerflow.runtime.user_context import get_effective_user_id


def test_normal_result_refreshes_are_distinct_but_same_result_still_hits_limit():
    mw = LoopDetectionMiddleware(warn_threshold=2, hard_limit=3)
    runtime = SimpleNamespace(context={"thread_id": "progress", "run_id": "run"})
    calls = [{"id": "same", "name": "nir_decision_context", "args": {}}]
    workflow = {"autonomous_delivery": True, "search_protocol": "sealed_holdout", "search_session_id": "search"}
    for index in range(8):
        workflow["attempt_evidence"] = {"experiment_id": f"experiment-{index}", "decision_id": f"decision-{index}"}
        assert mw.after_model({"nir_workflow": workflow, "messages": [AIMessage(content="", tool_calls=calls)]}, runtime) is None
        assert not mw._pending_warnings
    for index in range(2):
        # New transport IDs and workflow revisions alone are not progress.
        workflow["revision"] = 100 + index
        result = mw.after_model({"nir_workflow": workflow, "messages": [AIMessage(content="", tool_calls=[{**calls[0], "id": str(index)}])]}, runtime)
    assert result["messages"][0].additional_kwargs["agent_termination"]["source"] == "loop_detection"
    assert result["messages"][0].tool_calls == []


@pytest.mark.parametrize("source", ["loop_detection", "token_budget"])
def test_typed_stop_is_terminal_and_plain_chinese(source):
    workflow = {"stage": "decision_review", "revision": 1, "task_type": "calibration", "autonomous_delivery": True, "search_protocol": "sealed_holdout", "decision_strategy": "llm_autonomous"}
    message = AIMessage(content="Internal text [FORCED STOP]", additional_kwargs={"agent_termination": {"source": source, "reason": "internal safety limit"}})
    update = NIRWorkflowMiddleware().after_model({"nir_workflow": workflow, "messages": [message]}, SimpleNamespace(context={}))
    assert update["jump_to"] == "end" and update["nir_workflow"]["stage"] == "blocked"
    assert not update["nir_workflow"].get("response_guard_events")
    assert "已停止" in update["messages"][0].content
    assert "FORCED STOP" not in update["messages"][0].content and "Internal" not in update["messages"][0].content


class RepeatingContextLLM(UnifiedFakeLLM):
    _looping = False

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        results = [m for m in messages if isinstance(m, ToolMessage)]
        if self._looping or (results and results[-1].name == "nir_decision_context" and json.loads(results[-1].content).get("phase") == "feedback"):
            self._looping = True
            self._calls += 1
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Reading refreshed context to submit my terminal decision.", tool_calls=[{"id": f"repeat-{self._calls}", "name": "nir_decision_context", "args": {}}]))])
        return super()._generate(messages, stop, run_manager, **kwargs)


@pytest.mark.parametrize("asynchronous", [False, True])
def test_actual_lead_loop_stop_is_durable_no_retry_no_test(tmp_path, monkeypatch, asynchronous):
    monkeypatch.setattr(paths, "_paths", paths.Paths(tmp_path / "home"))
    thread = f"runtime-stop-{asynchronous}"
    owner = get_effective_user_id()
    paths.get_paths().ensure_thread_dirs(thread, user_id=owner)
    source = paths.get_paths().sandbox_uploads_dir(thread, user_id=owner) / "source.npz"
    output = paths.get_paths().sandbox_outputs_dir(thread, user_id=owner)
    X = np.random.default_rng(81).normal(size=(60, 8))
    np.savez(source, X=X, y=X[:, 0] * 2 + X[:, 1])
    monkeypatch.setattr(search_tools, "_resolve", lambda *a, **k: str(source) if k.get("read_only") else str(output))
    opened = []
    original = np.load

    def spy(path, *a, **k):
        opened.append(str(path))
        return original(path, *a, **k)

    monkeypatch.setattr(np, "load", spy)
    config = {"configurable": {"thread_id": thread}, "recursion_limit": 70}
    body = SimpleNamespace(assistant_id="lead_agent", context={}, input={"messages": [HumanMessage(content="请根据上传数据建模", id="start")]})
    bind_web_request(config, body)
    model = RepeatingContextLLM()
    graph = make_delivery_lead(monkeypatch, model, middlewares=[NIRWorkflowMiddleware(), LoopDetectionMiddleware(warn_threshold=2, hard_limit=3)])

    def invoke(value):
        return asyncio.run(graph.ainvoke(value, config, context=config["context"])) if asynchronous else graph.invoke(value, config, context=config["context"])

    first = invoke(body.input)
    workflow = first["nir_workflow"]
    assert workflow["stage"] == "blocked" and not waiting_for_user(workflow)
    assert workflow["runtime_termination"]["source"] == "loop_detection"
    assert "已停止" in first["messages"][-1].content
    assert "FORCED STOP" not in first["messages"][-1].content
    assert not workflow.get("response_guard_events")
    process = runtime_store(SimpleNamespace(config=config, context=config["context"]))
    assert stored_workflow(process)["runtime_termination"] == workflow["runtime_termination"]
    with process._connection() as db:
        session = search_tools.SearchSessionStore(process)._session(db, workflow["search_session_id"])
    assert session["stage"] == "safety_closed" and session["stop_source"] == "tool_failure"
    assert session["terminal_failure_reason"] == "loop_detection_limit"
    assert session["experiments"] == 1 and session["fits"] == 4
    assert session["freeze"] is None and session["final"] is None and not session.get("stop_decision_id")
    calls = model._calls
    again = invoke(first)
    assert model._calls == calls and again["nir_workflow"]["stage"] == "blocked"
    assert not any("holdout.npz" in path for path in opened)
