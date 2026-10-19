"""A Fake LLM drives the Lead Agent's real submit/execute tool chain."""

import asyncio
import json

import numpy as np
import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr
from test_nir_initial_decisions import proposal

from deerflow.community.nir import decision_tools, search_tools
from deerflow.community.nir.process import runtime_store


class InitialFakeLLM(BaseChatModel):
    family: str = "ridge"
    combination: bool = False
    correct_invalid: bool = False
    _calls: int = PrivateAttr(default=0)

    @property
    def _llm_type(self):
        return "fake-initial-decision"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self._calls += 1
        tool_messages = [m for m in messages if isinstance(m, ToolMessage)]
        last = json.loads(tool_messages[-1].content) if tool_messages else None
        if not tool_messages:
            name, args = "nir_prepare_search", {"cv_folds": 3}
        elif tool_messages[-1].name == "nir_prepare_search":
            name, args = "nir_decision_context", {}
        elif tool_messages[-1].name == "nir_decision_context" or last.get("status") == "error":
            context = next(json.loads(m.content) for m in tool_messages if m.name == "nir_decision_context")
            params = {"pls": {"n_components": 2}, "ridge": {"alpha": 1.7}, "svr": {"C": 2.0, "gamma": "scale", "epsilon": 0.05}, "et": {"n_estimators": 7, "max_depth": 3}}[self.family]
            steps = [{"method": "snv", "params": {}}, {"method": "mean_center", "params": {}}] if self.combination else []
            if self.correct_invalid and tool_messages[-1].name == "nir_decision_context":
                params = {"alpha": -1.0}
            plan = proposal(context, family=self.family, steps=steps, params=params)
            name, args = "nir_submit_decision", {"proposal": json.dumps(plan)}
        elif tool_messages[-1].name == "nir_submit_decision":
            assert last.get("status") != "error", last
            name, args = "nir_execute_decision", {"decision_id": last["decision_id"]}
        else:
            assert last["status"] == "succeeded", last
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="首轮执行已完成，返回验证证据并等待后续指令。"))])
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="", tool_calls=[{"id": "fake-call-" + str(self._calls), "name": name, "args": args}]))])


def make_test_lead(monkeypatch, model):
    import deerflow.tools
    from deerflow.agents.lead_agent import agent as lead
    from deerflow.agents.middlewares.nir_workflow_middleware import NIRWorkflowMiddleware
    from deerflow.config.app_config import AppConfig
    from deerflow.config.model_config import ModelConfig
    from deerflow.config.sandbox_config import SandboxConfig

    tools = [search_tools.nir_prepare_search_tool, decision_tools.nir_decision_context_tool, decision_tools.nir_submit_decision_tool, decision_tools.nir_execute_decision_tool]
    app = AppConfig(models=[ModelConfig(name="fake", use="langchain_openai:ChatOpenAI", model="fake", supports_thinking=False)], sandbox=SandboxConfig(use="deerflow.sandbox.local:LocalSandboxProvider"))
    monkeypatch.setattr(lead, "create_chat_model", lambda **k: model)
    monkeypatch.setattr(deerflow.tools, "get_available_tools", lambda **k: tools)
    monkeypatch.setattr(lead, "_load_enabled_skills_for_tool_policy", lambda *a, **k: [])
    monkeypatch.setattr(lead, "build_tracing_callbacks", lambda: [])
    # Retain the production NIR middleware; omit unrelated title/sandbox/MCP services.
    monkeypatch.setattr(lead, "build_middlewares", lambda *a, **k: [NIRWorkflowMiddleware()])
    return lead._make_lead_agent({"configurable": {"thinking_enabled": False}}, app_config=app)


@pytest.mark.parametrize(
    "family,combination,correct_invalid,asynchronous", [("pls", False, False, False), ("ridge", False, False, False), ("svr", True, False, False), ("et", False, False, False), ("ridge", False, True, False), ("svr", True, False, True)]
)
def test_fake_lead_selects_any_first_family_and_executes_only_that_plan(tmp_path, monkeypatch, family, combination, correct_invalid, asynchronous):
    from deerflow.community.nir.workflow import start_workflow
    from deerflow.config.paths import get_paths

    thread = f"initial-{family}-{correct_invalid}-{asynchronous}"
    get_paths().ensure_thread_dirs(thread)
    source = get_paths().sandbox_uploads_dir(thread) / "initial.npz"
    rng = np.random.default_rng(906)
    X = rng.normal(size=(60, 16))
    np.savez(source, X=X, y=2 * X[:, 0] - X[:, 1], wv=np.arange(16, dtype=float))
    workflow = start_workflow(
        task_type="calibration", data_path="/mnt/user-data/uploads/initial.npz", domain="default", analyte="protein", unit="%", validation_goal="internal_holdout", search_protocol="sealed_holdout", decision_strategy="llm_autonomous"
    )
    # Resolve virtual source without starting a sandbox process.
    monkeypatch.setattr(search_tools, "_resolve", lambda *a, **k: str(source))
    original = np.load

    def spy(path, *a, **k):
        assert "holdout" not in str(path), "Agent search accessed Test"
        return original(path, *a, **k)

    monkeypatch.setattr(np, "load", spy)
    if family != "pls":
        from sklearn.cross_decomposition import PLSRegression

        monkeypatch.setattr(PLSRegression, "fit", lambda *a, **k: pytest.fail("Non-PLS decision trained hidden PLS"))
    graph = make_test_lead(monkeypatch, InitialFakeLLM(family=family, combination=combination, correct_invalid=correct_invalid))
    # Durable binding restoration adds one before_agent graph node. Preserve the
    # same bounded model/tool script while allowing that framework transition.
    arguments = ({"messages": [HumanMessage(content="根据 Cal 特征自主选择首轮近红外回归方案，执行一次并返回 Val 结果。")], "nir_workflow": workflow}, {"configurable": {"thread_id": thread}, "recursion_limit": 20})
    result = asyncio.run(graph.ainvoke(*arguments)) if asynchronous else graph.invoke(*arguments)
    final = result["nir_workflow"]
    assert final["stage"] == "initial_result"
    store = runtime_store(type("Runtime", (), {"config": {"configurable": {"thread_id": thread}}, "context": {}})())
    with store._connection() as db:
        decision = db.execute("SELECT payload,execution FROM initial_decisions").fetchone()
        record, feedback = json.loads(decision[0]), json.loads(decision[1])
        assert db.execute("SELECT count(*) FROM search_experiments").fetchone()[0] == 1
    assert record["origin"]["verification_kind"] == "fake_model"
    assert record["accepted_spec"]["method"] == family
    assert len(record["accepted_spec"]["steps"]) == (2 if combination else 0)
    assert feedback["configuration"] == record["accepted_spec"]
    assert feedback["fits"] == 4 and feedback["metrics"]["test_status"] == "sealed"
