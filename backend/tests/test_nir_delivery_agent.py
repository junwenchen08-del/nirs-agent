"""Natural task -> actual Lead tools -> immutable choices -> final delivery."""

import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from test_nir_autonomous_initial_agent import InitialFakeLLM
from test_nir_delivery_decisions import stop_plan
from test_nir_initial_decisions import proposal
from test_nir_multiround_decisions import next_plan

from deerflow.community.nir import decision_tools, search_tools
from deerflow.community.nir.process import runtime_store
from deerflow.community.nir.workflow import nir_workflow_tool


class DeliveryFakeLLM(InitialFakeLLM):
    finish_with: str = "llm"
    invalid: bool = False
    _round = 0

    @property
    def _llm_type(self):
        return "fake-delivery-decision"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self._calls += 1
        results = [message for message in messages if isinstance(message, ToolMessage)]
        last = results[-1] if results else None
        value = json.loads(last.content) if last else None
        if last is None:
            name, args = (
                "nir_workflow",
                {
                    "action": "start",
                    "task_type": "calibration",
                    "data_path": "/mnt/user-data/uploads/source.npz",
                    "analyte": "protein",
                    "unit": "%",
                    "domain": "default",
                    "validation_goal": "internal_holdout",
                    "search_protocol": "sealed_holdout",
                    "decision_strategy": "llm_autonomous",
                    "autonomous_rounds": True,
                    "autonomous_delivery": True,
                    "task_budget": json.dumps({"max_experiments": 2 if self.finish_with == "budget" else 5, "max_fits": 8 if self.finish_with == "budget" else 50}),
                },
            )
        elif last.name == "nir_workflow":
            name, args = "nir_prepare_search", {"cv_folds": 3}
        elif last.name in {"nir_prepare_search", "nir_execute_decision"}:
            name, args = "nir_decision_context", {}
        elif last.name == "nir_decision_context" or value.get("code") == "nir_decision_rejected":
            ctx = next(json.loads(message.content) for message in reversed(results) if message.name == "nir_decision_context")
            if ctx["phase"] == "initial":
                plan = proposal(ctx)
            elif ctx["phase"] == "final_selection":
                plan = stop_plan(ctx, ctx["history_state"]["successful_experiments"][0]["experiment_id"], "RETAIN")
            elif self._round == 1:
                plan = next_plan(ctx, params={"alpha": 0.2})
            else:
                plan = stop_plan(ctx, ctx["history"][0]["experiment_id"])
            if self.invalid:
                plan = proposal(ctx, params={"alpha": -1.0})
            name, args = "nir_submit_decision", {"proposal": json.dumps(plan)}
        elif last.name == "nir_submit_decision":
            if value["action"] in {"STOP", "RETAIN"}:
                name, args = "nir_freeze_search", {"experiment_id": value["proposal"]["retained_experiment_id"]}
            else:
                self._round += 1
                name, args = "nir_execute_decision", {"decision_id": value["decision_id"]}
        elif last.name == "nir_freeze_search":
            assert value.get("status") != "error", value
            name, args = "nir_finalize_search", {}
        elif last.name == "nir_finalize_search":
            assert value.get("status") != "error", value
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="内部独立留出评价和报告交付完成，等待用户审批。"))])
        else:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="无效决策达到安全限制，搜索关闭，Test 仍封存。"))])
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content="", tool_calls=[{"id": f"delivery-call-{self._calls}", "name": name, "args": args}], usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120}))]
        )


def make_delivery_lead(monkeypatch, model, *, middlewares=None, extra_tools=None):
    import deerflow.tools
    from deerflow.agents.lead_agent import agent as lead
    from deerflow.agents.middlewares.nir_workflow_middleware import NIRWorkflowMiddleware
    from deerflow.config.app_config import AppConfig
    from deerflow.config.model_config import ModelConfig
    from deerflow.config.sandbox_config import SandboxConfig

    tools = [
        nir_workflow_tool,
        search_tools.nir_prepare_search_tool,
        decision_tools.nir_decision_context_tool,
        decision_tools.nir_submit_decision_tool,
        decision_tools.nir_execute_decision_tool,
        decision_tools.nir_decision_history_tool,
        search_tools.nir_freeze_search_tool,
        search_tools.nir_finalize_search_tool,
        *(extra_tools or []),
    ]
    app = AppConfig(models=[ModelConfig(name="fake", use="langchain_openai:ChatOpenAI", model="fake", supports_thinking=False)], sandbox=SandboxConfig(use="deerflow.sandbox.local:LocalSandboxProvider"))
    monkeypatch.setattr(lead, "create_chat_model", lambda **k: model)
    monkeypatch.setattr(deerflow.tools, "get_available_tools", lambda **k: tools)
    monkeypatch.setattr(lead, "_load_enabled_skills_for_tool_policy", lambda *a, **k: [])
    monkeypatch.setattr(lead, "build_tracing_callbacks", lambda: [])
    monkeypatch.setattr(lead, "build_middlewares", lambda *a, **k: middlewares if middlewares is not None else [NIRWorkflowMiddleware()])
    return lead._make_lead_agent({"configurable": {"thinking_enabled": False}}, app_config=app)


def test_explicit_delivery_request_cannot_silently_start_legacy():
    from deerflow.agents.middlewares.nir_workflow_middleware import _authorize

    request = SimpleNamespace(
        runtime=None, tool_call={"id": "bad-start", "name": "nir_workflow", "args": {"action": "start", "task_type": "calibration"}}, state={"messages": [HumanMessage(content="请使用 autonomous_delivery=true 完成建模交付")]}
    )
    denied = _authorize(request)
    assert json.loads(denied.content)["code"] == "nir_delivery_mode_required"
    request.tool_call = {"id": "bad-tool", "name": "nir_train_model", "args": {}}
    assert json.loads(_authorize(request).content)["code"] == "nir_delivery_mode_required"


def test_failed_delivery_intake_is_bounded_before_prepare(tmp_path, monkeypatch):
    class InvalidStartLLM(DeliveryFakeLLM):
        def _generate(self, messages, **kwargs):
            self._calls += 1
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="", tool_calls=[{"id": f"bad-{self._calls}", "name": "nir_workflow", "args": {"action": "start", "task_type": "calibration"}}]))])

    monkeypatch.setenv("NIR_AUTONOMOUS_MAX_LLM_CALLS", "2")
    model = InvalidStartLLM()
    result = make_delivery_lead(monkeypatch, model).invoke({"messages": [HumanMessage(content="请使用 autonomous_delivery=true 完成建模交付")]}, {"configurable": {"thread_id": "invalid-intake"}, "recursion_limit": 20})
    assert model._calls == 2
    assert result["messages"][-1].additional_kwargs["nir_program_termination"]["stop_source"] == "budget"
    store = runtime_store(SimpleNamespace(config={"configurable": {"thread_id": "invalid-intake"}}, context={}))
    with store._connection() as db:
        assert not db.execute("SELECT id FROM search_sessions").fetchall()


@pytest.mark.parametrize("finish_with,asynchronous", [("llm", False), ("budget", False), ("llm", True), ("budget", True)])
def test_fake_full_lead_delivery_and_closed_test(tmp_path, monkeypatch, finish_with, asynchronous):
    from deerflow.config.paths import get_paths

    thread = "delivery-" + finish_with + str(asynchronous)
    get_paths().ensure_thread_dirs(thread)
    source = get_paths().sandbox_uploads_dir(thread) / "source.npz"
    output = get_paths().sandbox_outputs_dir(thread)
    rng = np.random.default_rng(402)
    X = rng.normal(size=(60, 8))
    np.savez(source, X=X, y=3 * X[:, 0] - X[:, 1] + rng.normal(scale=0.02, size=60), wv=np.arange(8, dtype=float))
    monkeypatch.setattr(search_tools, "_resolve", lambda runtime, path, **k: str(source) if k.get("read_only") else str(output))
    original = np.load
    opened = []

    def spy(path, *a, **k):
        opened.append(str(path))
        return original(path, *a, **k)

    monkeypatch.setattr(np, "load", spy)
    graph = make_delivery_lead(monkeypatch, DeliveryFakeLLM(finish_with=finish_with))
    args = ({"messages": [HumanMessage(content="请根据光谱数据自主建模，结合验证反馈选择下一步，在合适时停止并交付模型报告。")]}, {"configurable": {"thread_id": thread}, "recursion_limit": 45})
    result = asyncio.run(graph.ainvoke(*args)) if asynchronous else graph.invoke(*args)
    wf = result["nir_workflow"]
    assert wf["stage"] == "review" and wf["approval_status"] == "pending", wf
    store = runtime_store(SimpleNamespace(config={"configurable": {"thread_id": thread}}, context={}))
    with store._connection() as db:
        session = search_tools.SearchSessionStore(store)._session(db, wf["search_session_id"])
        decisions = db.execute("SELECT payload FROM initial_decisions ORDER BY rowid").fetchall()
    assert session["stop_source"] == finish_with and session["fits"] == 8
    assert session["freeze"]["experiment_id"] == json.loads(decisions[0][0])["experiment_id"]
    assert len([path for path in opened if "holdout.npz" in path]) == 1
    assert session["export"]["lifecycle"]["registered"] is False


def test_invalid_llm_repairs_are_bounded_in_real_tool_loop(tmp_path, monkeypatch):
    from deerflow.config.paths import get_paths

    thread = "delivery-invalid"
    get_paths().ensure_thread_dirs(thread)
    source = get_paths().sandbox_uploads_dir(thread) / "source.npz"
    X = np.random.default_rng(89).normal(size=(50, 8))
    np.savez(source, X=X, y=X[:, 0])
    monkeypatch.setattr(search_tools, "_resolve", lambda *a, **k: str(source))
    result = make_delivery_lead(monkeypatch, DeliveryFakeLLM(invalid=True)).invoke({"messages": [HumanMessage(content="自主回归建模")]}, {"configurable": {"thread_id": thread}, "recursion_limit": 30})
    assert result["nir_workflow"]["stage"] == "blocked"
    assert result["nir_workflow"]["stop_source"] == "decision_invalid"
