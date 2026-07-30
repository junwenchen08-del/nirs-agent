"""Gateway export tests for captured NIR agent evaluation trajectories."""

from __future__ import annotations

import asyncio

from _router_auth_helpers import make_authed_test_app
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.memory import InMemorySaver

from app.gateway.routers import nir_evaluations
from deerflow.community.nir.workflow import start_workflow, transition_workflow
from deerflow.runtime.events.store.memory import MemoryRunEventStore
from deerflow.runtime.runs.manager import RunManager


def test_gateway_exports_checkpointed_workflow_as_cli_compatible_trace() -> None:
    app = make_authed_test_app()
    checkpointer = InMemorySaver()
    run_manager = RunManager()
    event_store = MemoryRunEventStore()
    app.state.checkpointer = checkpointer
    app.state.run_manager = run_manager
    app.state.run_event_store = event_store
    app.include_router(nir_evaluations.router)
    thread_id = "nir-eval-thread"

    async def _seed() -> None:
        run = await run_manager.create(thread_id)
        await run_manager.update_run_completion(
            run.run_id,
            total_input_tokens=700,
            total_output_tokens=300,
        )
        await event_store.put(
            thread_id=thread_id,
            run_id=run.run_id,
            event_type="middleware:skill_activation",
            category="middleware",
            content={
                "name": "SkillActivationMiddleware",
                "hook": "wrap_model_call",
                "action": "activate",
                "changes": {"skill_name": "nir-coordinator", "mode": "automatic"},
            },
        )
        workflow = start_workflow(
            task_type="calibration",
            data_path="corn.npz",
            analyte="protein",
            unit="%",
            domain="food_protein",
            validation_goal="internal_holdout",
        )
        workflow = transition_workflow(workflow, action="record_audit", audit_passed=True)
        workflow = transition_workflow(workflow, action="plan_ready")
        workflow = transition_workflow(
            workflow,
            action="record_attempt",
            attempt_passed=True,
            grade="A",
            model_path="model.pkl",
            metrics_path="metrics.json",
        )
        workflow = transition_workflow(workflow, action="approve")
        workflow = transition_workflow(workflow, action="registered")
        workflow = transition_workflow(workflow, action="complete")
        workflow["run_ids"] = [run.run_id]
        workflow["trace_ids"] = ["trace-gateway"]
        workflow["tool_observations"] = [
            {
                "name": "nir_train_model",
                "status": "success",
                "stage_before": "execution",
                "stage_after": "review",
                "code": None,
                "call_id": "call-model",
                "run_id": run.run_id,
                "trace_id": "trace-gateway",
            }
        ]
        checkpoint = empty_checkpoint()
        checkpoint["channel_values"] = {
            "nir_workflow": workflow,
            "messages": [
                AIMessage(
                    content="内部独立留出结果已记录；这不是外部验证。",
                    id="final-answer",
                )
            ],
        }
        version = checkpointer.get_next_version(None, None)
        checkpoint["channel_versions"] = {
            "nir_workflow": version,
            "messages": version,
        }
        await checkpointer.aput(
            {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}},
            checkpoint,
            {"step": 1, "source": "loop", "writes": {}, "parents": {}},
            {"nir_workflow": version, "messages": version},
        )

    asyncio.run(_seed())

    with TestClient(app) as client:
        response = client.get(
            f"/api/threads/{thread_id}/nir-evaluation-trace",
            params={"scenario_id": "calibration-register"},
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["version"] == 1
    assert len(payload["traces"]) == 1
    trace = payload["traces"][0]
    assert trace["scenario_id"] == "calibration-register"
    assert trace["routed_skill"] == "nir-coordinator"
    assert trace["workflow"]["stage"] == "completed"
    assert trace["tool_calls"][0]["name"] == "nir_train_model"
    assert trace["run_ids"]
    assert trace["trace_id"] == "trace-gateway"
    assert trace["input_tokens"] == 700
    assert trace["output_tokens"] == 300
    assert trace["response_text"] == "内部独立留出结果已记录；这不是外部验证。"


def test_gateway_returns_404_when_thread_has_no_nir_workflow() -> None:
    app = make_authed_test_app()
    checkpointer = InMemorySaver()
    app.state.checkpointer = checkpointer
    app.state.run_manager = RunManager()
    app.state.run_event_store = MemoryRunEventStore()
    app.include_router(nir_evaluations.router)
    checkpoint = empty_checkpoint()

    async def _seed() -> None:
        await checkpointer.aput(
            {"configurable": {"thread_id": "plain-thread", "checkpoint_ns": ""}},
            checkpoint,
            {"step": 1, "source": "loop", "writes": {}, "parents": {}},
            {},
        )

    asyncio.run(_seed())

    with TestClient(app) as client:
        response = client.get(
            "/api/threads/plain-thread/nir-evaluation-trace",
            params={"scenario_id": "inspection-complete"},
        )

    assert response.status_code == 404


def test_gateway_lists_versioned_nir_evaluation_scenarios() -> None:
    app = make_authed_test_app()
    app.include_router(nir_evaluations.router)

    with TestClient(app) as client:
        response = client.get("/api/nir/evaluations/scenarios")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["version"] == 1
    assert len(payload["scenarios"]) == 23
    assert payload["scenarios"][0]["id"]
    assert isinstance(payload["scenarios"][0]["tags"], list)


def test_gateway_evaluates_multiple_checkpointed_threads_in_one_request() -> None:
    app = make_authed_test_app()
    checkpointer = InMemorySaver()
    run_manager = RunManager()
    app.state.checkpointer = checkpointer
    app.state.run_manager = run_manager
    app.state.run_event_store = MemoryRunEventStore()
    app.include_router(nir_evaluations.router)

    async def _seed() -> None:
        inspection = start_workflow(task_type="inspection", data_path="sample.csv")
        inspection = transition_workflow(inspection, action="record_audit", audit_passed=True)
        inspection = transition_workflow(inspection, action="plan_ready")
        inspection = transition_workflow(inspection, action="complete")
        workflows = {
            "batch-inspection": inspection,
            "batch-intake": start_workflow(task_type="calibration"),
        }
        for thread_id, workflow in workflows.items():
            checkpoint = empty_checkpoint()
            checkpoint["channel_values"] = {"nir_workflow": workflow}
            version = checkpointer.get_next_version(None, None)
            checkpoint["channel_versions"] = {"nir_workflow": version}
            await checkpointer.aput(
                {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}},
                checkpoint,
                {"step": 1, "source": "loop", "writes": {}, "parents": {}},
                {"nir_workflow": version},
            )

    asyncio.run(_seed())

    with TestClient(app) as client:
        response = client.post(
            "/api/nir/evaluations/run",
            json={
                "entries": [
                    {"thread_id": "batch-inspection", "scenario_id": "inspection-complete"},
                    {"thread_id": "batch-intake", "scenario_id": "intake-missing-requirements"},
                ]
            },
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["version"] == 1
    assert payload["summary"]["total"] == 2
    assert [result["scenario_id"] for result in payload["summary"]["results"]] == [
        "inspection-complete",
        "intake-missing-requirements",
    ]
    assert len(payload["traces"]) == 2


def test_gateway_batch_evaluation_rejects_thread_without_owner_access() -> None:
    app = make_authed_test_app(owner_check_passes=False)
    app.state.checkpointer = InMemorySaver()
    app.state.run_manager = RunManager()
    app.state.run_event_store = MemoryRunEventStore()
    app.include_router(nir_evaluations.router)

    with TestClient(app) as client:
        response = client.post(
            "/api/nir/evaluations/run",
            json={"entries": [{"thread_id": "foreign-thread", "scenario_id": "inspection-complete"}]},
        )

    assert response.status_code == 404
