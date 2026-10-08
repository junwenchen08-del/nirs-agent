"""Registration persists the exact approved bundle without another user prompt."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import HumanMessage
from test_nir_model_service import _registered_package
from test_nir_model_service import model_library as _model_library_fixture
from test_nir_model_tools import _runtime
from test_nir_workflow_middleware import _execution_state, _request, _result

from deerflow.agents.middlewares.nir_workflow_middleware import NIRWorkflowMiddleware
from deerflow.community.nir import model_autosave, model_tools, modeling
from deerflow.community.nir.models.service import ModelError
from deerflow.community.nir.registration import nir_register_model_tool
from deerflow.community.nir.workflow import transition_workflow
from deerflow.config.nir_library_config import NIRLibraryConfig

model_library = _model_library_fixture


@pytest.fixture
def enabled_autosave(monkeypatch):
    policy = NIRLibraryConfig(enabled=True, min_free_disk_bytes=0)
    monkeypatch.setattr(model_autosave, "get_app_config", lambda: SimpleNamespace(nir_library=policy))
    return policy


@pytest.mark.asyncio
async def test_automatic_save_uses_approved_identity_and_reuses_existing(enabled_autosave, model_library, monkeypatch):
    service, paths = model_library
    version, workflow = _registered_package(paths)
    monkeypatch.setattr(model_tools, "_service", lambda: service)
    runtime = _runtime("同意注册模型", workflow=workflow)
    registration = {"status": "registered", "model_id": "tablet_assay", "version": version}
    saved = await model_autosave.save_registered_model(runtime, workflow, registration)
    reused = await model_autosave.save_registered_model(runtime, workflow, registration)
    assert saved["status"] == "saved"
    assert reused["status"] == "reused"
    rows = await service.list_models("alice")
    assert [(row["model_id"], row["version"]) for row in rows] == [("tablet_assay", version)]
    assert await service.list_models("bob") == []


@pytest.mark.asyncio
@pytest.mark.parametrize("message", ["同意注册，但不要保存到模型库", "Register it but don't save the model to the library"])
async def test_explicit_refusal_skips_automatic_storage(enabled_autosave, monkeypatch, message):
    service = MagicMock()
    monkeypatch.setattr(model_tools, "_service", lambda: service)
    result = await model_autosave.save_registered_model(_runtime(message), {"stage": "registered", "approval_status": "approved"}, {"status": "registered"})
    assert result["status"] == "skipped"
    service.promote_registered.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["enabled", "auto_save_registered_models"])
async def test_disabled_policy_does_not_write(enabled_autosave, monkeypatch, field):
    setattr(enabled_autosave, field, False)
    service = MagicMock()
    monkeypatch.setattr(model_tools, "_service", lambda: service)
    result = await model_autosave.save_registered_model(_runtime("同意注册"), {}, {})
    assert result["status"] == "disabled"
    service.promote_registered.assert_not_called()


@pytest.mark.asyncio
async def test_storage_failure_reports_a_separate_outcome(enabled_autosave, monkeypatch):
    service = MagicMock()
    service.promote_registered = AsyncMock(side_effect=ModelError("quota_exceeded", "Full"))
    monkeypatch.setattr(model_tools, "_service", lambda: service)
    result = await model_autosave.save_registered_model(_runtime("同意注册"), {"stage": "registered", "approval_status": "approved"}, {"status": "registered", "model_id": "tablet", "version": "tablet-v1"})
    assert result["status"] == "failed"
    assert result["error_code"] == "quota_exceeded"
    assert "容量不足" in result["message"]


@pytest.mark.asyncio
@pytest.mark.parametrize("workflow", [{"stage": "approved", "approval_status": "approved"}, {"stage": "registered", "approval_status": "pending"}])
async def test_unregistered_or_unapproved_workflow_cannot_auto_save(enabled_autosave, monkeypatch, workflow):
    service = MagicMock()
    monkeypatch.setattr(model_tools, "_service", lambda: service)
    result = await model_autosave.save_registered_model(_runtime("建模"), workflow, {"status": "registered"})
    assert result["error_code"] == "registration_required"
    service.promote_registered.assert_not_called()


@pytest.mark.asyncio
async def test_async_agent_registration_finishes_with_a_listable_model(enabled_autosave, model_library, monkeypatch):
    service, paths = model_library
    version, package = _registered_package(paths)
    monkeypatch.setattr(model_tools, "_service", lambda: service)
    monkeypatch.setattr(NIRWorkflowMiddleware, "_start_process", staticmethod(lambda _: None))
    monkeypatch.setattr(NIRWorkflowMiddleware, "_finish_process", staticmethod(lambda *args, **kwargs: None))
    output = paths.sandbox_outputs_dir("source-thread", user_id="alice")
    monkeypatch.setattr(modeling, "_resolve", lambda runtime, path, *, read_only: str(output / path.rsplit("/", 1)[-1]))
    workflow = transition_workflow(
        _execution_state(),
        action="record_attempt",
        attempt_passed=True,
        model_path=package["attempt_evidence"]["model_path"],
        metrics_path=package["attempt_evidence"]["metrics_path"],
        attempt_evidence={**package["attempt_evidence"], "tool_name": "nir_train_model"},
    )
    workflow = transition_workflow(workflow, action="approve")
    request = _request(
        "nir_register_model",
        {"nir_workflow": workflow, "messages": [HumanMessage(content="同意注册模型")]},
        context={"user_id": "alice", "thread_id": "source-thread", "run_id": "run-one"},
        args={"model_id": "tablet_new", "model_path": package["attempt_evidence"]["model_path"], "metrics_path": package["attempt_evidence"]["metrics_path"]},
    )

    async def registered(bound):
        payload = await asyncio.to_thread(nir_register_model_tool.func, runtime=bound.runtime, **bound.tool_call["args"])
        return _result("nir_register_model", json.loads(payload))

    result = await NIRWorkflowMiddleware().awrap_tool_call(request, registered)
    assert result.update["nir_workflow"]["stage"] == "registered"
    assert result.update["nir_workflow"]["registered_model"]["model_id"] == "tablet_new"
    assert result.update["nir_workflow"]["registered_model"]["version"] != version
    assert result.update["nir_workflow"]["model_library"]["status"] == "saved"
    assert json.loads(result.update["messages"][0].content)["model_library"]["status"] == "saved"
    assert len(await service.list_models("alice")) == 1


@pytest.mark.asyncio
async def test_failed_registration_never_invokes_automatic_save(monkeypatch):
    spy = AsyncMock()
    monkeypatch.setattr("deerflow.agents.middlewares.nir_workflow_middleware.save_registered_model", spy)
    workflow = transition_workflow(
        _execution_state(),
        action="record_attempt",
        attempt_passed=True,
        model_path="/mnt/user-data/outputs/model.pkl",
        metrics_path="/mnt/user-data/outputs/metrics.json",
        attempt_evidence={
            "schema_version": 1,
            "tool_name": "nir_train_model",
            "protocol": "random_three_way_holdout",
            "validation_scope": "independent_holdout_not_external",
            "model_path": "/mnt/user-data/outputs/model.pkl",
            "metrics_path": "/mnt/user-data/outputs/metrics.json",
        },
    )
    workflow = transition_workflow(workflow, action="approve")

    async def failed(_):
        return _result("nir_register_model", {"status": "error", "error": "integrity failed"})

    result = await NIRWorkflowMiddleware().awrap_tool_call(_request("nir_register_model", {"nir_workflow": workflow}, args={"model_path": workflow["model_path"], "metrics_path": workflow["metrics_path"]}), failed)
    assert result.update["nir_workflow"]["stage"] == "approved"
    spy.assert_not_awaited()
