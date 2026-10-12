"""Regression for real malformed provider proposals and misleading closure."""

import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from test_nir_delivery_agent import make_delivery_lead
from test_nir_unified_agent import UnifiedFakeLLM

from deerflow.community.nir import search_tools
from deerflow.community.nir.process import runtime_store
from deerflow.community.nir.response_grounding import render_grounded_nir_response


def test_sealed_evidence_deduplicates_by_experiment_identity():
    first = {"experiment_id": "exp-first", "metrics_summary": {"val": {"RMSE": 0.42}}}
    last = {"experiment_id": "exp-last", "metrics_summary": {"val": {"RMSE": 0.44}, "RMSECV": 0.7}}
    workflow = {"search_protocol": "sealed_holdout", "decision_strategy": "llm_autonomous", "stage": "blocked", "attempts": [first, last], "attempt_evidence": last}
    rendered = render_grounded_nir_response(workflow)
    assert rendered.count("exp-last") == 1
    assert "尝试 3" not in rendered
    assert "Val" in rendered and "Test" in rendered


class MalformedDecisionLLM(UnifiedFakeLLM):
    never_correct: bool = False
    _malformed_count = 0

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        results = [m for m in messages if isinstance(m, ToolMessage)]
        if results and results[-1].name in {"nir_workflow", "nir_prepare_search"}:
            assert json.loads(results[-1].content).get("status") != "error", results[-1].content
        if results and results[-1].name == "nir_decision_context":
            assert "phase" in json.loads(results[-1].content), results[-1].content
        if results and results[-1].name == "nir_decision_context" and json.loads(results[-1].content)["phase"] == "feedback" and (self.never_correct or self._malformed_count == 0):
            self._calls += 1
            self._malformed_count += 1
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content="I will resubmit the proposal.",
                            id=f"malformed-{self._malformed_count}",
                            invalid_tool_calls=[{"name": "nir_submit_decision", "id": f"invalid-{self._malformed_count}", "args": '{"proposal": {"action": "TUNE",', "error": None}],
                            response_metadata={"finish_reason": "length"},
                            usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120},
                        )
                    )
                ]
            )
        repair = next((m for m in reversed(messages) if isinstance(m, HumanMessage) and m.additional_kwargs.get("nir_invalid_tool_call")), None)
        if self._malformed_count:
            assert repair is not None
            assert "JSON" in repair.content and "line" in repair.content
            assert repair.additional_kwargs["hide_from_ui"] is True
        return super()._generate(messages, stop, run_manager, **kwargs)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("never_correct", [False, True])
def test_actual_lead_malformed_json_repairs_or_closes_durably(tmp_path, monkeypatch, asynchronous, never_correct):
    from deerflow.config import paths
    from deerflow.runtime.user_context import get_effective_user_id

    thread = f"malformed-{asynchronous}-{never_correct}"
    monkeypatch.setattr(paths, "_paths", paths.Paths(tmp_path / "home"))
    user_id = get_effective_user_id()
    paths.get_paths().ensure_thread_dirs(thread, user_id=user_id)
    source = paths.get_paths().sandbox_uploads_dir(thread, user_id=user_id) / "source.npz"
    output = paths.get_paths().sandbox_outputs_dir(thread, user_id=user_id)
    X = np.random.default_rng(80).normal(size=(50, 8))
    np.savez(source, X=X, y=X[:, 0] + X[:, 1])
    monkeypatch.setattr(search_tools, "_resolve", lambda runtime, path, **k: str(source) if k.get("read_only") else str(output))
    original = np.load
    opened = []

    def spy(path, *a, **k):
        opened.append(str(path))
        return original(path, *a, **k)

    monkeypatch.setattr(np, "load", spy)
    model = MalformedDecisionLLM(never_correct=never_correct)
    graph = make_delivery_lead(monkeypatch, model)
    from deerflow.community.nir.web_entry import bind_web_request, stored_workflow

    body = SimpleNamespace(assistant_id="lead_agent", context={}, input={"messages": [HumanMessage(content="请自主建模并交付")]})
    config = {"configurable": {"thread_id": thread}, "recursion_limit": 55}
    bind_web_request(config, body)
    args = (body.input, config)
    result = asyncio.run(graph.ainvoke(*args, context=config["context"])) if asynchronous else graph.invoke(*args, context=config["context"])
    wf = result["nir_workflow"]
    store = runtime_store(SimpleNamespace(config=config, context=config["context"]))
    with store._connection() as db:
        session = search_tools.SearchSessionStore(store)._session(db, wf["search_session_id"])
        decisions = db.execute("SELECT payload FROM initial_decisions").fetchall()
        receipts = [json.loads(r[0]) for r in db.execute("SELECT payload FROM model_issued_calls WHERE tool='__model_response__'")]
    assert any(r.get("invalid_tool_calls") for r in receipts)
    assert not any(json.loads(r[0])["origin"]["tool_call_id"].startswith("invalid-") for r in decisions)
    if never_correct:
        assert wf["stage"] == "blocked" and wf["stop_source"] == "decision_invalid"
        assert session["stage"] == "safety_closed"
        assert session["stop_source"] == "decision_invalid" and session["terminal_failure_reason"] == "workflow_continuation_limit"
        assert session["fits"] == 4 and session["freeze"] is None and session["final"] is None
        assert "JSON" in result["messages"][-1].content and "Test" in result["messages"][-1].content
        assert not any("holdout.npz" in p for p in opened)
        assert stored_workflow(store)["stage"] == "blocked"
    else:
        assert wf["stage"] == "review" and session["stop_source"] == "llm"
        assert session["fits"] == 8
        assert len([p for p in opened if "holdout.npz" in p]) == 1
        assert session["automatic_delivery"]["status"] == "delivered"


def test_legacy_invalid_tool_call_does_not_enable_autonomous_repair():
    from deerflow.agents.middlewares.nir_workflow_middleware import NIRWorkflowMiddleware

    message = AIMessage(content="", invalid_tool_calls=[{"name": "nir_submit_decision", "id": "bad", "args": "{", "error": None}])
    assert NIRWorkflowMiddleware()._guard_final_response({"messages": [message]}) is None


def test_provider_projection_drops_raw_invalid_json_but_preserves_checkpoint():
    from deerflow.agents.middlewares.nir_workflow_middleware import NIRWorkflowMiddleware

    invalid = AIMessage(
        content="Promise",
        invalid_tool_calls=[{"name": "nir_submit_decision", "id": "bad", "args": "{" + "x" * 20000, "error": None}],
        additional_kwargs={"tool_calls": [{"id": "bad", "function": {"arguments": "{" + "x" * 20000}}]},
    )
    context = ToolMessage(content=json.dumps({"context_id": "ctx-real"}), name="nir_decision_context", tool_call_id="context-read")
    request = SimpleNamespace(state={"nir_workflow": {"autonomous_rounds": True}}, messages=[context, invalid])
    request.override = lambda **updates: SimpleNamespace(**{**vars(request), **updates})
    updated = NIRWorkflowMiddleware._compact_decision_messages(request)
    assert invalid.invalid_tool_calls[0]["args"] == "{" + "x" * 20000
    assert not updated.messages[-1].invalid_tool_calls and "tool_calls" not in updated.messages[-1].additional_kwargs
    assert len(updated.messages[-1].content) < 200
    assert updated.messages[0] is context


@pytest.mark.parametrize("already_frozen", [False, True])
def test_continuation_closure_preserves_budget_source_and_frozen_artifacts(tmp_path, already_frozen):
    from test_nir_delivery_decisions import setup_delivery

    from deerflow.community.nir.decision_budget import close_incomplete_workflow

    service, ctx, _ = setup_delivery(tmp_path)
    with service.search.process._connection(write=True) as db:
        session = service.search._session(db, ctx["session_id"])
        session.update(stage="budget_exhausted", stop_source="budget", stop_reason=["max_fits"])
        if already_frozen:
            session.update(stage="frozen", freeze={"experiment_id": "already-retained"})
        service.search._save(db, session)
    result = close_incomplete_workflow(service.search, ctx["session_id"], invalid_json=True)
    assert result["stop_source"] == "budget" and result["stop_reason"] == ["max_fits"]
    assert result["stage"] == ("frozen" if already_frozen else "safety_closed")
    assert result.get("freeze") == ({"experiment_id": "already-retained"} if already_frozen else None)
