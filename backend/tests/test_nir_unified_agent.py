"""Natural chat uses the real Lead tool loop; terminal delivery makes no LLM calls."""

import asyncio
import json
from types import SimpleNamespace

import jsonschema
import numpy as np
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import PrivateAttr
from test_nir_delivery_agent import DeliveryFakeLLM, make_delivery_lead

from deerflow.community.nir import search_tools
from deerflow.community.nir.autonomous_view import AutonomousView
from deerflow.community.nir.process import runtime_store
from deerflow.community.nir.web_entry import bind_web_request, stored_workflow
from deerflow.config import paths
from deerflow.runtime.user_context import get_effective_user_id

HIDDEN = {"search_protocol", "decision_strategy", "autonomous_rounds", "autonomous_delivery", "max_attempts", "max_fits", "max_decisions", "task_budget", "budget_profile"}


class UnifiedFakeLLM(DeliveryFakeLLM):
    _schemas: dict = PrivateAttr(default_factory=dict)
    _tools_called: list = PrivateAttr(default_factory=list)
    question: bool = False

    def bind_tools(self, tools, **kwargs):
        self._schemas = {item.name: convert_to_openai_tool(item)["function"]["parameters"] for item in tools}
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if self.question:
            self._calls += 1
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="SNV 是逐条光谱的标准正态变量变换。"))])
        result = super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        response = result.generations[0].message
        for call in response.tool_calls:
            assert call["name"] not in {"nir_freeze_search", "nir_finalize_search"}, "No provider call after terminal acceptance is needed"
            self._tools_called.append(call["name"])
            if call["name"] == "nir_workflow":
                for key in HIDDEN:
                    call["args"].pop(key, None)
                assert not HIDDEN.intersection(self._schemas[call["name"]]["properties"])
            if call["name"] == "nir_submit_decision":
                plan = json.loads(call["args"]["proposal"])
                for alternative in plan["alternatives"]:
                    alternative.setdefault("action", "EXPLORE")
                    if plan["phase"] == "final_selection":
                        alternative.update(action="RETAIN", pipeline=None, optimization=None)
                call["args"]["proposal"] = plan
            jsonschema.validate(call["args"], self._schemas[call["name"]])
        # Deliberately large fake usage proves arithmetic only, not provider
        # context-window behavior or any real billing result.
        response.usage_metadata = {"input_tokens": 4_000_000, "output_tokens": 20, "total_tokens": 4_000_020}
        return result


@pytest.mark.parametrize("finish_with", ["llm", "budget"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_natural_task_two_executions_and_automatic_delivery(tmp_path, monkeypatch, finish_with, asynchronous):
    monkeypatch.setattr(paths, "_paths", paths.Paths(tmp_path / "home"))
    monkeypatch.setenv("NIR_AUTONOMOUS_MAX_EXPERIMENTS", "2" if finish_with == "budget" else "3")
    monkeypatch.setenv("NIR_AUTONOMOUS_MAX_FITS", "8" if finish_with == "budget" else "12")
    thread = f"unified-{finish_with}-{asynchronous}"
    user_id = get_effective_user_id()
    paths.get_paths().ensure_thread_dirs(thread, user_id=user_id)
    source = paths.get_paths().sandbox_uploads_dir(thread, user_id=user_id) / "source.npz"
    output = paths.get_paths().sandbox_outputs_dir(thread, user_id=user_id)
    X = np.random.default_rng(130).normal(size=(60, 8))
    np.savez(source, X=X, y=X[:, 0] * 3 - X[:, 1], wv=np.arange(8))
    monkeypatch.setattr(search_tools, "_resolve", lambda *a, **k: str(source if k.get("read_only") else output))
    original = np.load
    tests = []

    def load(path, *a, **k):
        if "holdout.npz" in str(path):
            tests.append(str(path))
        return original(path, *a, **k)

    monkeypatch.setattr(np, "load", load)
    task = HumanMessage(content="请用上传光谱预测蛋白质，结合验证反馈自主建模并交付。")
    config = {"configurable": {"thread_id": thread}, "recursion_limit": 100}
    body = SimpleNamespace(assistant_id="lead_agent", context={}, input={"messages": [task]})
    bind_web_request(config, body)
    body.input["artifacts"] = ["/mnt/user-data/outputs/previous-report.md"]
    model = UnifiedFakeLLM(finish_with=finish_with)
    graph = make_delivery_lead(monkeypatch, model)
    result = asyncio.run(graph.ainvoke(body.input, config, context=config["context"])) if asynchronous else graph.invoke(body.input, config, context=config["context"])
    workflow = result["nir_workflow"]
    assert workflow["stage"] == "review" and workflow["approval_status"] == "pending"
    assert result["messages"][-1].additional_kwargs["nir_deterministic_delivery"] is True
    assert result["messages"][-1].additional_kwargs.get("nir_delivery_session_id") == workflow["search_session_id"]
    assert {path.rsplit("/", 1)[-1] for path in result.get("artifacts", [])} >= {"model.joblib", "report.html", "delivery.zip"}
    assert "/mnt/user-data/outputs/previous-report.md" in result["artifacts"]
    process = runtime_store(SimpleNamespace(config=config, context=config["context"]))
    with process._connection() as db:
        session = search_tools.SearchSessionStore(process)._session(db, workflow["search_session_id"])
        decisions = [json.loads(row[0]) for row in db.execute("SELECT payload FROM initial_decisions ORDER BY rowid")]
    assert [item["action"] for item in decisions] == ["EXPLORE", "TUNE", "STOP" if finish_with == "llm" else "RETAIN"]
    assert session["fits"] == 8 and session["experiments"] == 2 and len(tests) == 1
    assert session["stop_source"] == finish_with and session["automatic_delivery"]["status"] == "delivered"
    assert session["llm_budget"]["search"]["calls"] + session["llm_budget"]["finalization"]["calls"] == model._calls
    assert session["llm_budget"]["search"]["actual_input_tokens"] > 3_000_000
    assert session["delivery_policy"]["max_llm_tokens"] is None
    assert session["freeze"]["experiment_id"] == decisions[0]["experiment_id"]
    assert stored_workflow(process)["search_session_id"] == session["id"]
    assert not {"nir_freeze_search", "nir_finalize_search"}.intersection(model._tools_called)
    from deerflow.agents.middlewares.nir_workflow_middleware import NIRWorkflowMiddleware

    restored = NIRWorkflowMiddleware._restore_unified_binding({}, SimpleNamespace(config=config, context=config["context"]))
    assert restored["nir_workflow"]["search_session_id"] == session["id"]
    restored_legacy = NIRWorkflowMiddleware._restore_unified_binding({"nir_workflow": {"revision": 999, "search_protocol": "legacy"}}, SimpleNamespace(config=config, context=config["context"]))
    assert restored_legacy["nir_workflow"]["search_protocol"] == "sealed_holdout"
    before = process.db.read_bytes()
    view = AutonomousView(process)
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("Readonly refresh cannot read Test/arrays"))
    assert view.summary(session["id"])["downloads"]
    assert view.summary(session["id"])["automatic_delivery"]["status"] == "delivered"
    assert process.db.read_bytes() == before


@pytest.mark.parametrize("question", ["什么是 SNV？", "PLS 与 SVR 有什么区别？", "解释近红外预处理。"])
def test_questions_never_create_modeling_sessions(tmp_path, monkeypatch, question):
    monkeypatch.setattr(paths, "_paths", paths.Paths(tmp_path / "home"))
    config = {"configurable": {"thread_id": "ordinary"}}
    body = SimpleNamespace(assistant_id="lead_agent", context={}, input={"messages": [HumanMessage(content=question)]})
    bind_web_request(config, body)
    model = UnifiedFakeLLM(question=True)
    result = make_delivery_lead(monkeypatch, model).invoke(body.input, config, context=config["context"])
    assert not result.get("nir_workflow") and model._calls == 1
    assert not runtime_store(SimpleNamespace(config=config, context=config["context"])).db.exists()


def test_rejected_unified_start_is_bounded_and_initial_usage_is_counted(tmp_path, monkeypatch):
    class InvalidStart(DeliveryFakeLLM):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            self._calls += 1
            return ChatResult(
                generations=[
                    ChatGeneration(
                        message=AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "id": f"invalid-start-{self._calls}",
                                    "name": "nir_workflow",
                                    "args": {"action": "start", "task_type": "calibration", "data_path": "/mnt/user-data/uploads/missing.npz"},
                                }
                            ],
                            usage_metadata={"input_tokens": 99, "output_tokens": 1, "total_tokens": 100},
                        )
                    )
                ]
            )

    monkeypatch.setattr(paths, "_paths", paths.Paths(tmp_path / "home"))
    monkeypatch.setenv("NIR_AUTONOMOUS_MAX_LLM_CALLS", "2")
    config = {"configurable": {"thread_id": "bad-unified"}, "recursion_limit": 15}
    body = SimpleNamespace(assistant_id="lead_agent", context={}, input={"messages": [HumanMessage(content="请建模，文件尚未上传。")]})
    bind_web_request(config, body)
    model = InvalidStart()
    result = make_delivery_lead(monkeypatch, model).invoke(body.input, config, context=config["context"])
    assert model._calls == 2 and not result.get("nir_workflow")
    assert result["messages"][-1].additional_kwargs["nir_program_termination"]["stop_source"] == "budget"
    process = runtime_store(SimpleNamespace(config=config, context=config["context"]))
    with process._connection() as db:
        ledger = json.loads(db.execute("SELECT payload FROM model_issued_calls WHERE id='phase4_delivery_intake'").fetchone()[0])
        assert not db.execute("SELECT id FROM search_sessions").fetchall()
    assert ledger["budget"]["search"]["calls"] == 2 and ledger["budget"]["search"]["actual_input_tokens"] == 198
    assert ledger["stop_reasons"] == ["llm_calls"]
