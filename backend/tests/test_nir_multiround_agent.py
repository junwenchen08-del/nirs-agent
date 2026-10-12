"""The production Lead/NIR loop consumes real feedback before choosing actions."""

import asyncio
import json
from types import SimpleNamespace

import numpy as np
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from test_nir_autonomous_initial_agent import InitialFakeLLM, make_test_lead
from test_nir_initial_decisions import proposal
from test_nir_multiround_decisions import next_plan

from deerflow.community.nir import search_tools
from deerflow.community.nir.process import runtime_store


class FeedbackFakeLLM(InitialFakeLLM):
    followups: list[str] = ["TUNE", "CHANGE_PREPROCESS", "SWITCH_MODEL", "EXPLORE"]

    @property
    def _llm_type(self):
        return "fake-feedback-decision"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self._calls += 1
        tools = [message for message in messages if isinstance(message, ToolMessage)]
        last = json.loads(tools[-1].content) if tools else None
        if not tools:
            name, args = "nir_prepare_search", {"cv_folds": 3}
        elif tools[-1].name in {"nir_prepare_search", "nir_execute_decision"}:
            assert last.get("status") != "failed", last
            name, args = "nir_decision_context", {}
        elif tools[-1].name == "nir_decision_context":
            context = last
            if context["phase"] == "initial":
                plan = proposal(context)
            else:
                # History comes from the tool, not from an invented narrative or
                # a fixed quality threshold. The Fake selects an action to test.
                current = context["history_state"]["current"]
                assert current["metrics"]["val"]["scope"] == "validation"
                index = context["budget"]["experiments_used"] - 1
                action = self.followups[index] if index < len(self.followups) else "STOP"
                spec = current["accepted_spec"]
                family, steps = spec["method"], spec["steps"]
                params = spec["parameter_plan"]["fixed_params"].copy()
                if action == "TUNE":
                    params["alpha"] = 3.0 + current["metrics"]["val"]["RMSE"]
                elif action == "CHANGE_PREPROCESS":
                    steps = [{"method": "snv", "params": {}}]
                elif action == "SWITCH_MODEL":
                    family, params = "svr", {"C": 2.0, "epsilon": 0.05}
                elif action == "EXPLORE":
                    family, params, steps = "et", {"n_estimators": 7, "max_depth": 3}, []
                plan = next_plan(context, action, family, steps, params)
                plan["rationale"] = f"Observed previous Val RMSE={current['metrics']['val']['RMSE']}; test this explicit {action} hypothesis"
                if action == "STOP":
                    plan.update(retained_experiment_id=current["experiment_id"], expected_fits=0)
                    plan["alternatives"][0].update(pipeline=None, optimization=None)
            name, args = "nir_submit_decision", {"proposal": json.dumps(plan)}
        elif tools[-1].name == "nir_submit_decision":
            assert last.get("status") != "error", last
            if last["action"] == "STOP":
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content="已记录停止决策，保留所选实验，Test 封存，等待最终确认。"))])
            name, args = "nir_execute_decision", {"decision_id": last["decision_id"]}
        else:
            pytest.fail(str(last))
        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content="", id=f"feedback-message-{self._calls}", tool_calls=[{"id": f"feedback-call-{self._calls}", "name": name, "args": args}], usage_metadata={"input_tokens": 10, "output_tokens": 10, "total_tokens": 20}
                    )
                )
            ]
        )


@pytest.mark.parametrize("asynchronous", [False, True])
def test_fake_lead_all_feedback_actions_and_stop(tmp_path, monkeypatch, asynchronous):
    from deerflow.community.nir.workflow import start_workflow
    from deerflow.config.paths import get_paths

    thread = f"feedback-{asynchronous}"
    get_paths().ensure_thread_dirs(thread)
    source = get_paths().sandbox_uploads_dir(thread) / "synthetic.npz"
    rng = np.random.default_rng(901)
    X = rng.normal(size=(60, 16))
    np.savez(source, X=X, y=2 * X[:, 0] - X[:, 1], wv=np.arange(16, dtype=float))
    workflow = start_workflow(
        task_type="calibration",
        data_path="/mnt/user-data/uploads/synthetic.npz",
        domain="default",
        analyte="protein",
        unit="%",
        validation_goal="internal_holdout",
        search_protocol="sealed_holdout",
        decision_strategy="llm_autonomous",
        autonomous_rounds=True,
        max_attempts=6,
        max_fits=24,
    )
    monkeypatch.setattr(search_tools, "_resolve", lambda *a, **k: str(source))
    original_load = np.load

    def guarded(path, *a, **k):
        assert "holdout" not in str(path), "Multi-round search read Test"
        return original_load(path, *a, **k)

    monkeypatch.setattr(np, "load", guarded)
    from sklearn.cross_decomposition import PLSRegression

    monkeypatch.setattr(PLSRegression, "fit", lambda *a, **k: pytest.fail("Hidden PLS fit"))
    graph = make_test_lead(monkeypatch, FeedbackFakeLLM())
    arguments = ({"messages": [HumanMessage(content="自主选择并根据真实验证反馈继续决策")], "nir_workflow": workflow}, {"configurable": {"thread_id": thread}, "recursion_limit": 90})
    result = asyncio.run(graph.ainvoke(*arguments)) if asynchronous else graph.invoke(*arguments)
    assert result["nir_workflow"]["stage"] == "awaiting_final_confirmation"
    store = runtime_store(SimpleNamespace(config={"configurable": {"thread_id": thread}}, context={}))
    with store._connection() as db:
        rows = list(db.execute("SELECT payload,execution FROM initial_decisions ORDER BY rowid"))
        records = [json.loads(row[0]) for row in rows]
        feedback = [json.loads(row[1]) for row in rows]
        assert db.execute("SELECT count(*) FROM search_experiments").fetchone()[0] == 5
        assert db.execute("SELECT count(*) FROM model_issued_calls WHERE tool='__model_response__'").fetchone()[0] >= 17
    assert [record["action"] for record in records] == ["EXPLORE", "TUNE", "CHANGE_PREPROCESS", "SWITCH_MODEL", "EXPLORE", "STOP"]
    for index in range(1, 5):
        assert feedback[index]["previous_experiment_id"] == feedback[index - 1]["experiment_id"]
        assert str(feedback[index - 1]["metrics"]["val"]["RMSE"]) in records[index]["proposal"]["rationale"]
    assert sum(result["fits"] for result in feedback[:-1]) == 20
    assert all(record["origin"]["verification_kind"] == "fake_model" for record in records)
