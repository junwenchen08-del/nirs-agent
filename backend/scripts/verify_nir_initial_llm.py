"""One bounded real-provider Lead Agent/NIR integration, isolated user data."""

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
from deerflow.community.nir.workflow import start_workflow
from deerflow.config.app_config import AppConfig
from deerflow.config.paths import get_paths
from deerflow.models.factory import create_chat_model
from deerflow.reflection import resolve_variable


class BoundedNIRMiddleware(NIRWorkflowMiddleware):
    """Smoke-only provider call cap, before making a charged request."""

    def __init__(self):
        super().__init__()
        self.requests = 0

    def wrap_model_call(self, request, handler):
        if self.requests >= 9:
            raise ValueError("Real verification model response budget exhausted")
        self.requests += 1
        return super().wrap_model_call(request, handler)


def main():
    parser = argparse.ArgumentParser(description="One real-provider initial decision smoke (uses the configured model and incurs API cost).")
    parser.add_argument("--allow-real-provider", action="store_true", help="Explicitly authorize this opt-in provider test")
    parser.add_argument("--multi-round", action="store_true", help="Two executions / 14 fits / at most 9 model responses, using actual feedback")
    args = parser.parse_args()
    if not args.allow_real_provider:
        parser.error("--allow-real-provider is required")
    root = Path(__file__).resolve().parents[2]
    phase = "phase3" if args.multi_round else "phase2"
    isolated_home = root / ".tmp" / (phase + "-llm-smoke-home")
    os.environ.setdefault("DEER_FLOW_HOME", str(isolated_home))
    if not Path(os.environ["DEER_FLOW_HOME"]).resolve().is_relative_to(root / ".tmp"):
        parser.error("Use a fresh DEER_FLOW_HOME below this repository's .tmp directory")
    (root / ".tmp").mkdir(exist_ok=True)
    logging.disable(logging.CRITICAL)
    root = Path(__file__).resolve().parents[2]
    evidence_path = root / ".tmp" / (phase + "-real-llm-" + uuid.uuid4().hex[:12] + ".json")
    summary = {"verification": "real_provider_attempt", "started": time.time()}
    try:
        app = AppConfig.from_file(str(root / "config.yaml"))
        selected = app.models[0]
        summary.update(model_name=selected.name, provider=selected.use)
        app = app.model_copy(deep=True)
        app.memory.enabled = False
        app.memory.injection_enabled = False
        app.summarization.enabled = False
        app.title.enabled = False
        app.tool_search.enabled = False
        thread = phase + "-real-" + uuid.uuid4().hex[:12]
        summary["thread_id"] = thread
        get_paths().ensure_thread_dirs(thread)
        source = get_paths().sandbox_uploads_dir(thread) / "synthetic.npz"
        rng = np.random.default_rng(3907)
        X = rng.normal(size=(48, 12))
        y = 2 * X[:, 0] + 0.5 * X[:, 1] ** 2 + rng.normal(scale=0.03, size=48)
        np.savez(source, X=X, y=y, wv=np.linspace(900, 1700, 12))
        workflow = start_workflow(
            task_type="calibration",
            data_path="/mnt/user-data/uploads/synthetic.npz",
            domain="default",
            analyte="synthetic property",
            unit="a.u.",
            validation_goal="internal_holdout",
            search_protocol="sealed_holdout",
            decision_strategy="llm_autonomous",
            max_attempts=2 if args.multi_round else 1,
            autonomous_rounds=args.multi_round,
            max_fits=14 if args.multi_round else 200,
            max_decisions=4,
        )
        runtime = SimpleNamespace(state={"nir_workflow": workflow}, context={}, config={"configurable": {"thread_id": thread}}, tool_call_id="trusted-ingestion")
        # Reuse phase-1 trusted ingestion; set a small server-owned fit budget before the model runs.
        service = SearchSessionStore(runtime_store(runtime))
        original_prepare = service.prepare

        def bounded_prepare(*a, **k):
            k["max_fits"] = 14 if args.multi_round else 7
            return original_prepare(*a, **k)

        with patch.object(search_tools, "_resolve", lambda *a, **k: str(source)), patch.object(search_tools, "_service", lambda _: service), patch.object(service, "prepare", bounded_prepare):
            command = search_tools.nir_prepare_search_tool.func(runtime=runtime, cv_folds=3)
        workflow = command.update["nir_workflow"]
        summary["session_id"] = workflow["search_session_id"]
        app.models[0] = app.models[0].model_copy(update={"max_tokens": 2600 if args.multi_round else 1800, "timeout": 45, "max_retries": 0, "streaming": False})
        model = create_chat_model(name=selected.name, thinking_enabled=False, app_config=app, attach_tracing=False)
        summary["actual_model_class"] = type(model).__name__
        tool_names = {item.name: item.use for item in app.tools}
        names = ["nir_decision_context", "nir_submit_decision", "nir_execute_decision"]
        if args.multi_round:
            names.append("nir_decision_history")
        assert all(name in tool_names for name in names), "New tools absent from real config"
        tools = [resolve_variable(tool_names[name]) for name in names]
        assert [tool.name for tool in tools] == names
        middleware = [DynamicContextMiddleware(app_config=app), BoundedNIRMiddleware() if args.multi_round else NIRWorkflowMiddleware()]
        prompt = (
            "This is a real small-budget verification of autonomous INITIAL NIR regression pipeline selection. A fresh sealed session is already prepared. "
            "Read nir_decision_context, compare at least two meaningful alternatives and independently select preprocessing, one supported family and fixed/grid parameter strategy. "
            "Do not use a mandatory baseline or force any particular family. Use only returned evidence and runtime schemas. At most 2 parameter trials, 3 frozen folds and 7 total fits. "
            "Prefer a concise proposal within output limit. Submit it before executing; on validation rejection explicitly correct it. "
            "Call nir_execute_decision once, report actual Val performance and identifiers, then stop. Do not freeze/finalize or read Test. No subsequent optimization loop."
        )
        if args.multi_round:
            prompt = (
                "This explicitly authorized synthetic smoke verifies TWO autonomous NIR decisions and executions, using the existing sealed session. "
                "First read context, compare two meaningful initial alternatives, independently choose and submit one model/preprocessing/parameter plan, then execute. "
                "After actual execution, read a fresh context and independently compare feedback-based actions TUNE, CHANGE_PREPROCESS, SWITCH_MODEL, EXPLORE or STOP. "
                "For this two-execution verification prefer one informative legal follow-up experiment if budget supports it, but no model family, preprocessing or specific action is mandated. "
                "Use actual preceding Val/CV results, record issues/uncertainty, unselected candidate reasons, hypotheses and expected_fits. Name current base_experiment_id. "
                "Allocate the cumulative fit budget yourself across the two experiments; use the actual context's remaining candidates/fits. "
                "There are 3 frozen CV folds and at most TWO experiments and 14 cumulative fits. A fixed candidate costs 4 fits. "
                "Correct explicit validation errors, never substitute silently or call legacy tools. After the second execution report actual feedback and linkage. "
                "The hard experiment limit ends this verification distinctly from LLM STOP. Do not read Test or freeze/finalize. Keep each proposal concise. "
                "No more than 9 model responses are allowed, including repair turns. Raw [] is legal. Use only the schema and evidence actually returned."
            )
        summary["limits"] = {"experiments": 2 if args.multi_round else 1, "fits": 14 if args.multi_round else 7, "model_responses": 9 if args.multi_round else None}
        with (
            patch.object(lead, "create_chat_model", lambda **k: model),
            patch("deerflow.tools.get_available_tools", lambda **k: tools),
            patch.object(lead, "build_middlewares", lambda *a, **k: middleware),
            patch.object(lead, "build_tracing_callbacks", lambda: []),
        ):
            graph = lead._make_lead_agent({"configurable": {"thinking_enabled": False}}, app_config=app)
        original_load = np.load
        opened = []

        def guarded_load(path, *a, **k):
            opened.append(Path(path).name)
            if "holdout" in str(path):
                raise AssertionError("Search read Test")
            return original_load(path, *a, **k)

        summary["steps"] = []
        seen = set()
        model_responses = 0
        with patch.object(np, "load", guarded_load):
            for result in graph.stream(
                {"nir_workflow": workflow, "messages": [HumanMessage(content=prompt)]}, {"configurable": {"thread_id": thread}, "recursion_limit": 30}, context={"run_id": "real-smoke-" + thread}, stream_mode="values"
            ):
                for message in result.get("messages", []):
                    if message.id in seen:
                        continue
                    seen.add(message.id)
                    if isinstance(message, ToolMessage):
                        try:
                            payload = json.loads(message.content)
                        except Exception:
                            payload = {}
                        summary["steps"].append({"tool": message.name, "status": payload.get("status"), "code": payload.get("code"), "errors": payload.get("errors"), "state": result.get("nir_workflow", {}).get("stage")})
                    elif isinstance(message, AIMessage):
                        model_responses += 1
                        if args.multi_round and model_responses > 9:
                            raise ValueError("Real verification model response budget exhausted")
                        summary["steps"].append({"model_tool_calls": [call["name"] for call in message.tool_calls], "message_id": message.id, "token_usage": message.usage_metadata})
        with service.process._connection() as db:
            rows = list(db.execute("SELECT id FROM initial_decisions ORDER BY rowid"))
            row = rows[0] if rows else None
            experiment_count = db.execute("SELECT count(*) FROM search_experiments").fetchone()[0]
        assert row is not None and experiment_count == (2 if args.multi_round else 1), "Expected recorded experiments"
        record = InitialDecisionService(service).get(row[0])
        assert record["origin"]["verification_kind"] == "provider_model"
        assert record["status"] == "succeeded", record["status"]
        assert result["nir_workflow"]["stage"] == ("budget_exhausted" if args.multi_round else "initial_result")
        if args.multi_round:
            records = [InitialDecisionService(service).get(item[0]) for item in rows]
            assert len(records) == 2 and all(record["status"] == "succeeded" and record["origin"]["verification_kind"] == "provider_model" for record in records)
            assert records[1]["base_experiment_id"] == records[0]["experiment_id"]
            assert records[1]["execution"]["previous_experiment_id"] == records[0]["experiment_id"]
            assert all(record["execution"]["configuration"] == record["accepted_spec"] for record in records)
            summary.update(records=records, terminal_stage=result["nir_workflow"]["stage"], stop_source="hard_budget", multi_round=True)
        summary.update(
            status="passed",
            thread_id=thread,
            decision_id=row[0],
            context_id=record["context_id"],
            proposal=record["proposal"],
            accepted_spec=record["accepted_spec"],
            origin=record["origin"],
            execution=record["execution"],
            files_opened=opened,
            test_access=False,
            agent_chain="production Lead factory/prompt + actual NIR and dynamic-context middlewares; unrelated middleware services omitted",
            service_scope="in-process configured provider, no Gateway/UI deployment",
        )
    except Exception as exc:
        summary.update(status="unverified", error_type=type(exc).__name__, http_status=getattr(exc, "status_code", None))
        summary["trace_frames"] = [{"file": Path(frame.filename).name, "line": frame.lineno, "function": frame.name} for frame in traceback.extract_tb(exc.__traceback__)]
        if isinstance(exc, TypeError):
            summary["local_type_error"] = str(exc)[:500]
    finally:
        summary["elapsed_seconds"] = time.time() - summary["started"]
        evidence_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({key: summary.get(key) for key in ("status", "model_name", "actual_model_class", "error_type", "http_status", "elapsed_seconds", "local_type_error", "steps")}, ensure_ascii=False))

    print("Evidence: " + str(evidence_path))
    return 0 if summary.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
