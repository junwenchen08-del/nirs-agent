"""Structured web selection drives the actual Lead without policy in prose."""

import json
from types import SimpleNamespace

import jsonschema
import numpy as np
import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import PrivateAttr
from sklearn.linear_model import Ridge
from test_nir_delivery_agent import DeliveryFakeLLM, make_delivery_lead

from deerflow.community.nir import search_tools
from deerflow.community.nir.autonomous_view import AutonomousView
from deerflow.community.nir.process import runtime_store
from deerflow.community.nir.web_entry import bind_web_request, web_request
from deerflow.config import paths

MODE_FIELDS = ("search_protocol", "decision_strategy", "autonomous_rounds", "autonomous_delivery", "budget_profile", "task_budget")


class WebDeliveryFakeLLM(DeliveryFakeLLM):
    """Reuse decisions; intake adopts the Gateway-bound provider contract."""

    web_bound: dict
    _schemas: dict = PrivateAttr(default_factory=dict)
    _start_arguments: dict = PrivateAttr(default_factory=dict)
    _skill_views: list = PrivateAttr(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        self._schemas = {tool.name: convert_to_openai_tool(tool)["function"]["parameters"] for tool in tools}
        return self

    def _generate(self, messages, **kwargs):
        self._skill_views.extend(message.content for message in messages if message.additional_kwargs.get("slash_skill_activation"))
        result = super()._generate(messages, **kwargs)
        response = result.generations[0].message
        for call in response.tool_calls:
            assert call["name"] in self._schemas, "Lead exposed the wrong process-stage tool"
            schema, args = self._schemas[call["name"]], call["args"]
            if call["name"] == "nir_workflow":
                # Discard the parent's explicit non-web defaults. The current
                # intake schema requires the four flags; emit only its bound
                # constants instead of relying on text or hardcoded budgets.
                for field in (*MODE_FIELDS, "max_attempts", "max_fits", "max_decisions"):
                    args.pop(field, None)
                for field in MODE_FIELDS:
                    assert schema["properties"][field]["const"] == self.web_bound[field]
                    args[field] = self.web_bound[field]
                self._start_arguments = dict(args)
            elif call["name"] == "nir_submit_decision":
                proposal = json.loads(args["proposal"])
                for alternative in proposal["alternatives"]:
                    alternative.setdefault("action", "EXPLORE")
                    if proposal["phase"] == "final_selection":
                        alternative.update(action="RETAIN", pipeline=None, optimization=None)
                args["proposal"] = proposal
            jsonschema.validate(args, schema)
        return result


def _assert_no_plot_arrays(value):
    if isinstance(value, dict):
        assert not {"charts", "observed", "predicted", "residual", "y_test", "X_test"}.intersection(value)
        for item in value.values():
            _assert_no_plot_arrays(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_plot_arrays(item)


@pytest.mark.parametrize("finish_with", ["llm", "budget"])
@pytest.mark.parametrize("with_skill", [False, True])
def test_web_bound_natural_task_two_rounds_and_single_test_cache(tmp_path, monkeypatch, finish_with, with_skill):
    monkeypatch.setattr(paths, "_paths", paths.Paths(tmp_path / "home"))
    thread = "web-entry-" + finish_with
    task = "请根据上传近红外光谱预测蛋白质含量，比较建模反馈后选择下一步，并在结束时交付模型和报告。"
    user_budget = {"max_experiments": 2 if finish_with == "budget" else 3, "max_fits": 8 if finish_with == "budget" else 12, "max_decisions": 8, "max_wall_seconds": 180}
    # Adding the real summarization hook adds a graph node per iteration, even
    # when its provider call is correctly skipped. Task/model budgets stay fixed.
    config = {"configurable": {"thread_id": thread}, "recursion_limit": 70}
    body = SimpleNamespace(context={"nir_autonomous": {"enabled": True, "profile": "custom", "budget": user_budget}}, input={"messages": [HumanMessage(content=task)]})
    bind_web_request(config, body)
    bound = web_request(config["context"])
    assert bound["ui_profile"] == "custom"
    assert all(bound["delivery_policy"][field] == value for field, value in user_budget.items())
    assert not any(field in task for field in MODE_FIELDS)

    paths.get_paths().ensure_thread_dirs(thread)
    source = paths.get_paths().sandbox_uploads_dir(thread) / "source.npz"
    output = paths.get_paths().sandbox_outputs_dir(thread)
    rng = np.random.default_rng(517)
    spectra = rng.normal(size=(60, 8))
    np.savez(source, X=spectra, y=3 * spectra[:, 0] - spectra[:, 1] + rng.normal(scale=0.02, size=60), wv=np.arange(8, dtype=float))
    monkeypatch.setattr(search_tools, "_resolve", lambda runtime, path, **kwargs: str(source) if kwargs.get("read_only") else str(output))
    original_load, original_fit = np.load, Ridge.fit
    holdout_reads, fits = [], []

    def load_spy(path, *args, **kwargs):
        if "holdout.npz" in str(path):
            holdout_reads.append(str(path))
        return original_load(path, *args, **kwargs)

    def fit_spy(estimator, *args, **kwargs):
        fits.append(estimator.alpha)
        return original_fit(estimator, *args, **kwargs)

    monkeypatch.setattr(np, "load", load_spy)
    monkeypatch.setattr(Ridge, "fit", fit_spy)
    model = WebDeliveryFakeLLM(finish_with=finish_with, web_bound=bound)
    # Keep the production output wrapper in this test. The older phase-4 helper
    # intentionally isolated NIR middleware and missed context externalization.
    from unittest.mock import MagicMock

    from deerflow.agents.middlewares.nir_workflow_middleware import NIRWorkflowMiddleware
    from deerflow.agents.middlewares.summarization_middleware import DeerFlowSummarizationMiddleware
    from deerflow.agents.middlewares.tool_output_budget_middleware import ToolOutputBudgetMiddleware

    auxiliary_model = MagicMock()
    auxiliary_model.with_config.return_value = auxiliary_model
    auxiliary_model.invoke.return_value = SimpleNamespace(text="unbudgeted summary")
    summarizer = DeerFlowSummarizationMiddleware(model=auxiliary_model, trigger=("messages", 2), keep=("messages", 1), token_counter=len)
    middlewares = [summarizer, ToolOutputBudgetMiddleware(), NIRWorkflowMiddleware()]
    if with_skill:
        from pathlib import Path

        from deerflow.agents.middlewares.skill_activation_middleware import SkillActivationMiddleware, SkillAutoRoute
        from deerflow.config.app_config import AppConfig
        from deerflow.config.sandbox_config import SandboxConfig
        from deerflow.config.skills_config import SkillsConfig

        skill_config = AppConfig(sandbox=SandboxConfig(use="deerflow.sandbox.local:LocalSandboxProvider"), skills=SkillsConfig(path=str(Path(__file__).resolve().parents[2] / "skills")))
        middlewares.insert(
            0,
            SkillActivationMiddleware(
                app_config=skill_config, available_skills={"nir-coordinator"}, auto_routes=(SkillAutoRoute(skill_name="nir-coordinator", patterns=("近红外",), state_key="nir_workflow", terminal_statuses=("completed", "blocked")),)
            ),
        )
    graph = make_delivery_lead(monkeypatch, model, middlewares=middlewares)
    result = graph.invoke(body.input, config, context=config["context"])
    auxiliary_model.invoke.assert_not_called()
    auxiliary_model.ainvoke.assert_not_called()
    workflow = result["nir_workflow"]
    assert workflow["stage"] == "review" and workflow["approval_status"] == "pending"
    assert all(model._start_arguments[field] == bound[field] for field in MODE_FIELDS)
    assert workflow["search_protocol"] == "sealed_holdout" and workflow["decision_strategy"] == "llm_autonomous"
    assert workflow["autonomous_rounds"] and workflow["autonomous_delivery"]
    assert all(workflow["delivery_policy"][field] == value for field, value in user_budget.items())

    store = runtime_store(SimpleNamespace(config=config, context=config["context"]))
    with store._connection() as db:
        session = search_tools.SearchSessionStore(store)._session(db, workflow["search_session_id"])
        rows = db.execute("SELECT payload,status,execution FROM initial_decisions WHERE session_id=? ORDER BY rowid", (session["id"],)).fetchall()
    decisions = [json.loads(row["payload"]) for row in rows]
    assert [decision["action"] for decision in decisions] == ["EXPLORE", "TUNE", "RETAIN" if finish_with == "budget" else "STOP"]
    assert [row["status"] for row in rows] == ["succeeded", "succeeded", "stopped"]
    for row in rows[:2]:
        feedback = json.loads(row["execution"])
        assert feedback["feedback_id"] and feedback["metrics"]["val"]["scope"] == "validation"
        assert feedback["metrics"]["train_cv"]["scope"] == "calibration_cv" and feedback["fits"] == 4
    assert session["stage"] == "finalized" and session["stop_source"] == finish_with
    assert session["experiments"] == 2 and session["fits"] == len(fits) == 8
    assert len(holdout_reads) == 1
    assert session["freeze"]["experiment_id"] == decisions[0]["experiment_id"]
    assert session["export"]["lifecycle"]["registered"] is False
    assert session["final"]["charts"]["final_test"]["scope"] == "final_test"
    assert session["final"]["charts"]["final_test"]["n_returned"] > 0
    if with_skill:
        assert len(model._skill_views) >= 8
        assert all("autonomous_skill_projection" in view and "## 模式 Q" not in view for view in model._skill_views)
        assert session["llm_budget"]["finalization"]["charged_tokens"] <= 500000
    for message in result["messages"]:
        if isinstance(message, ToolMessage):
            _assert_no_plot_arrays(json.loads(message.content))

    before = store.db.read_bytes()
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: pytest.fail("Read-only web replay cannot load arrays or Test"))
    monkeypatch.setattr(Ridge, "fit", lambda *args, **kwargs: pytest.fail("Read-only web replay cannot fit models"))
    view = AutonomousView(store)
    summary = view.summary(session["id"])
    timeline = view.decisions(session["id"])
    chart = view.chart(session["id"], "final_test")
    assert summary == view.summary(session["id"])
    assert chart == view.chart(session["id"], "final_test")
    assert timeline["total"] == 3 and len(summary["trends"]) == 2
    assert view.decision(session["id"], decisions[1]["decision_id"])["feedback"]["fits"] == 4
    assert summary["budget"]["fits_used"] == 8 and len(fits) == 8 and len(holdout_reads) == 1
    assert chart == session["final"]["charts"]["final_test"]
    assert chart["n_total"] == summary["metrics"]["test"]["n"]
    assert {item["name"] for item in summary["downloads"]} >= {"model.joblib", "report.md", "delivery.zip"}
    assert store.db.read_bytes() == before
