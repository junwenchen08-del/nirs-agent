"""Bounded, opt-in real-provider natural-task -> sealed final delivery smoke."""

import argparse
import json
import logging
import os
import time
import traceback
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from deerflow.agents.lead_agent import agent as lead
from deerflow.agents.middlewares.dynamic_context_middleware import DynamicContextMiddleware
from deerflow.agents.middlewares.nir_workflow_middleware import NIRWorkflowMiddleware
from deerflow.community.nir import search_tools
from deerflow.community.nir.dataset_session import SearchSessionStore
from deerflow.community.nir.decision_context import InitialDecisionService
from deerflow.community.nir.process import runtime_store
from deerflow.config.app_config import AppConfig
from deerflow.config.paths import get_paths
from deerflow.models.factory import create_chat_model
from deerflow.reflection import resolve_variable


class VerificationMiddleware(NIRWorkflowMiddleware):
    """Additional smoke-wide request cap includes task intake before prepare."""

    def __init__(self):
        super().__init__()
        self.requests = 0

    def wrap_model_call(self, request, handler):
        def charged(req):
            if self.requests >= 24:
                raise ValueError("Smoke provider request safety cap reached")
            self.requests += 1
            return handler(req)

        return super().wrap_model_call(request, charged)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-real-provider", action="store_true")
    parser.add_argument("--check-local", action="store_true", help="Validate config/tool imports without any provider call")
    args = parser.parse_args()
    if not args.allow_real_provider and not args.check_local:
        parser.error("--allow-real-provider is required; this uses the configured service and incurs API costs")
    root = Path(__file__).resolve().parents[2]
    app = AppConfig.from_file(str(root / "config.yaml"))
    names = ["nir_workflow", "nir_prepare_search", "nir_decision_context", "nir_submit_decision", "nir_execute_decision", "nir_decision_history", "nir_freeze_search", "nir_finalize_search"]
    configured = {item.name: item.use for item in app.tools}
    tools = [resolve_variable(configured[name]) for name in names]
    assert [tool.name for tool in tools] == names
    if args.check_local:
        print(json.dumps({"config_tools_verified": names, "provider_calls": 0}))
        return 0
    isolated = root / ".tmp" / ("phase4-real-home-" + uuid.uuid4().hex[:10])
    os.environ.setdefault("DEER_FLOW_HOME", str(isolated))
    if not Path(os.environ["DEER_FLOW_HOME"]).resolve().is_relative_to(root / ".tmp"):
        parser.error("Use a fresh private DEER_FLOW_HOME under the repository .tmp")
    logging.disable(logging.CRITICAL)
    budget = {
        "max_experiments": 3,
        "max_fits": 18,
        "max_decisions": 6,
        "max_wall_seconds": 180,
        "max_llm_calls": 18,
        "max_llm_tokens": 800000,
        "max_output_tokens": 3600,
        "max_invalid_decision_repairs": 2,
        "final_llm_calls": 6,
        "final_llm_tokens": 500000,
        "final_wall_seconds": 180,
    }
    for key, value in budget.items():
        env = "NIR_AUTONOMOUS_" + key.upper()
        # Smoke may lower existing server limits, never enlarge them.
        os.environ[env] = str(min(int(os.environ.get(env, value)), value))
    thread = "phase4-real-" + uuid.uuid4().hex[:12]
    evidence = root / ".tmp" / (thread + ".json")
    summary = {"status": "unverified", "thread_id": thread, "started_at": time.time(), "limits": budget, "steps": []}
    responses, opened = [], []
    middleware, search = None, None
    started = time.monotonic()
    try:
        selected = app.models[0]
        app = app.model_copy(deep=True)
        app.memory.enabled = app.memory.injection_enabled = False
        app.summarization.enabled = app.title.enabled = app.tool_search.enabled = False
        app.models[0] = app.models[0].model_copy(update={"max_tokens": 3600, "timeout": 45, "max_retries": 0, "streaming": False})
        model = create_chat_model(name=selected.name, thinking_enabled=False, app_config=app, attach_tracing=False)
        summary.update(model_name=selected.name, actual_model_class=type(model).__name__, provider=selected.use)
        get_paths().ensure_thread_dirs(thread)
        source = get_paths().sandbox_uploads_dir(thread) / "synthetic.npz"
        outputs = get_paths().sandbox_outputs_dir(thread)
        rng = np.random.default_rng(4909)
        X = rng.normal(size=(60, 12))
        y = 2 * X[:, 0] + 0.4 * X[:, 1] ** 2 + rng.normal(scale=0.08, size=60)
        np.savez(source, X=X, y=y, wv=np.linspace(900, 1700, 12))
        prompt = (
            "请根据上传文件 /mnt/user-data/uploads/synthetic.npz 建立单目标近红外回归模型。目标为 synthetic property，单位 a.u.，domain=default。"
            "请使用 llm_autonomous、sealed_holdout、autonomous_rounds=true、autonomous_delivery=true，完成初始方案选择、真实 Cal/Val 反馈决策、合适时停止、冻结、唯一 Test 和模型报告交付。"
            "预处理、模型、参数策略、每个后续动作和何时停止都由你根据实际证据自主决定，无固定模型或固定轮数要求。允许提前 STOP；不能根据 Test 继续优化。"
            "使用内部独立留出验证和 development 预算配置，task_budget 为以下 JSON：" + json.dumps(budget) + "。Cal CV 使用3折，自主分配累计18fits。"
            "比较至少两个合理候选，只提交选中方案。错误需显式修正。STOP 和预算后 RETAIN 应按返回合同提供证据及保留实验。"
            "可以选择历史成功实验，程序不会替你选最小RMSE。最终使用返回的报告/模型链接，等待真实用户审批，不自动注册。保持提案简洁。"
        )
        summary["human_task"] = prompt
        middleware = VerificationMiddleware()
        with (
            patch.object(lead, "create_chat_model", lambda **k: model),
            patch("deerflow.tools.get_available_tools", lambda **k: tools),
            patch.object(lead, "build_middlewares", lambda *a, **k: [DynamicContextMiddleware(app_config=app), middleware]),
            patch.object(lead, "build_tracing_callbacks", lambda: []),
        ):
            graph = lead._make_lead_agent({"configurable": {"thinking_enabled": False}}, app_config=app)
        seen = set()
        original_load = np.load
        owner_runtime = SimpleNamespace(config={"configurable": {"thread_id": thread}}, context={})
        search = SearchSessionStore(runtime_store(owner_runtime))

        def load_spy(path, *a, **k):
            opened.append(Path(path).name)
            if Path(path).name == "holdout.npz":
                with search.process._connection() as db:
                    session = json.loads(db.execute("SELECT payload FROM search_sessions").fetchone()[0])
                assert session["stage"] == "testing" and session["freeze"], "Test accessed before durable freeze/lease"
            return original_load(path, *a, **k)

        def resolve(runtime, path, **kwargs):
            return str(source) if kwargs.get("read_only") else str(outputs)

        with patch.object(search_tools, "_resolve", resolve), patch.object(np, "load", load_spy):
            for result in graph.stream({"messages": [HumanMessage(content=prompt)]}, {"configurable": {"thread_id": thread}, "recursion_limit": 65}, context={"run_id": "real-smoke-" + thread}, stream_mode="values"):
                for message in result.get("messages", []):
                    if message.id in seen:
                        continue
                    seen.add(message.id)
                    if isinstance(message, ToolMessage):
                        payload = json.loads(message.content)
                        summary["steps"].append(
                            {"tool": message.name, "status": payload.get("status"), "code": payload.get("code"), "error": payload.get("error"), "errors": payload.get("errors"), "state": result.get("nir_workflow", {}).get("stage")}
                        )
                    elif isinstance(message, AIMessage):
                        responses.append(
                            {
                                "tools": [call["name"] for call in message.tool_calls],
                                "tool_calls": message.tool_calls,
                                "workflow_calls": [call["args"] for call in message.tool_calls if call["name"] == "nir_workflow"],
                                "token_usage": message.usage_metadata,
                                "program_termination": message.additional_kwargs.get("nir_program_termination"),
                            }
                        )
        with search.process._connection() as db:
            session = search._session(db, result["nir_workflow"]["search_session_id"])
            records = InitialDecisionService(search)._records(db, session["id"])
        trained = [record for record in records if record["status"] == "succeeded"]
        assert trained and session["stage"] == "finalized" and session.get("export"), "No complete deliverable"
        assert all(record["origin"]["verification_kind"] == "provider_model" for record in records)
        assert all(record["execution"]["actual_configuration"] == record["accepted_spec"] for record in trained)
        for previous, current in zip(trained, trained[1:]):
            assert current["execution"]["previous_experiment_id"] == previous["experiment_id"]
        assert opened.count("holdout.npz") == 1
        export = session["export"]
        terminal = records[-1]
        assert terminal["action"] in {"STOP", "RETAIN"}, "No actual model-issued terminal selection"
        assert session["freeze"]["experiment_id"] == terminal["proposal"]["retained_experiment_id"]
        assert terminal["proposal"]["phase"] in {"feedback", "final_selection"}, "No feedback-based subsequent decision"
        from deerflow.community.nir._common import _load_trusted_model_artifact

        artifact_path = outputs / export["model"].removeprefix("/mnt/user-data/outputs/")
        artifact = _load_trusted_model_artifact(str(artifact_path), export["model"])
        import joblib

        frozen = joblib.load(search._dir(session["id"]) / (session["freeze"]["experiment_id"] + ".joblib"))
        prediction_input = rng.normal(size=(3, 12))
        predictions = artifact["model"].predict(artifact["preprocessing"]["pipeline"].transform(prediction_input)).ravel()
        np.testing.assert_allclose(predictions, frozen.predict(prediction_input))
        report_path = outputs / export["report"].removeprefix("/mnt/user-data/outputs/")
        report = report_path.read_text(encoding="utf-8")
        assert all(record["decision_id"] in report for record in records)
        assert all(label in report for label in ("Cal CV", "Val", "Test"))
        assert (outputs / export["bundle"].removeprefix("/mnt/user-data/outputs/")).is_file()
        original_fits = session["fits"]
        with patch.object(np, "load", load_spy):
            replay = search.finalize(session["id"])
        assert replay == session["final"] and opened.count("holdout.npz") == 1
        try:
            InitialDecisionService(search).context(session["id"], result["nir_workflow"], prompt)
        except ValueError:
            pass
        else:
            raise AssertionError("Search reopened after Test")
        assert session["fits"] == original_fits
        summary.update(
            status="passed",
            session_id=session["id"],
            stop_source=session["stop_source"],
            records=records,
            final=export,
            llm_budget=session["llm_budget"],
            provider_requests=middleware.requests,
            responses=responses,
            files_opened=opened,
            successful_experiments=len(trained),
            terminal_stage=result["nir_workflow"]["stage"],
            user_approved=False,
            registered=False,
            artifact_prediction_verified=True,
            prediction_rows=len(predictions),
            retained_experiment_matches_freeze=True,
            final_test_replay_cached=True,
            search_closed_after_test=True,
            all_decisions_present_in_report=True,
            chain="production Lead factory/prompt + registered tools + NIR/DynamicContext middleware; in-process configured provider, unrelated Gateway/UI services omitted",
        )
    except Exception as exc:
        summary.update(
            error_type=type(exc).__name__, http_status=getattr(exc, "status_code", None), trace_frames=[{"file": Path(frame.filename).name, "line": frame.lineno, "function": frame.name} for frame in traceback.extract_tb(exc.__traceback__)]
        )
    summary.update(provider_requests=middleware.requests if middleware else 0, responses=responses, files_opened=opened)
    if search:
        with search.process._connection() as db:
            row = db.execute("SELECT payload FROM search_sessions ORDER BY rowid DESC LIMIT 1").fetchone()
            if row:
                session = json.loads(row[0])
                summary.update(session_id=session["id"], session_stage=session["stage"], stop_source=session.get("stop_source"), llm_budget=session.get("llm_budget"), records=InitialDecisionService(search)._records(db, session["id"]))
                summary.update(successful_experiments=sum(record["status"] == "succeeded" for record in summary["records"]), actual_model_fits=session["fits"])
    summary["elapsed_seconds"] = time.monotonic() - started
    evidence.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: summary.get(key) for key in ("status", "error_type", "http_status", "actual_model_class", "successful_experiments", "stop_source", "provider_requests", "elapsed_seconds")}, ensure_ascii=False))
    print("Evidence: " + str(evidence))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
