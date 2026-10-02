"""Agent-facing persistent model tools remain explicit, bounded, and owner-scoped."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain.tools import ToolRuntime
from langchain_core.messages import HumanMessage

from deerflow.community.nir import model_tools


def _runtime(message: str, *, workflow: dict | None = None) -> ToolRuntime:
    state = {"messages": [HumanMessage(content=message)]}
    if workflow is not None:
        state["nir_workflow"] = workflow
    return ToolRuntime(
        state=state,
        context={"user_id": "alice", "thread_id": "source-thread", "run_id": "run-one"},
        config={"configurable": {"thread_id": "source-thread"}},
        stream_writer=lambda _: None,
        tools=[],
        tool_call_id="call-1",
        store=None,
    )


@pytest.mark.asyncio
async def test_promote_requires_latest_explicit_persistence_request(monkeypatch: pytest.MonkeyPatch) -> None:
    service = MagicMock()
    service.promote_registered = AsyncMock(return_value={"model_id": "tablet", "version": "tablet-v1", "status": "ready"})
    monkeypatch.setattr(model_tools, "_service", lambda: service)
    workflow = {"stage": "registered", "approval_status": "approved", "attempt_evidence": {}}

    denied = json.loads(
        await model_tools.nir_model_promote_tool.coroutine(
            runtime=_runtime("模型注册好了", workflow=workflow),
            model_id="tablet",
            version="tablet-v1",
        )
    )
    assert denied["code"] == "confirmation_required"
    service.promote_registered.assert_not_called()

    promoted = json.loads(
        await model_tools.nir_model_promote_tool.coroutine(
            runtime=_runtime("请把这个模型持久化到模型库，供跨会话使用", workflow=workflow),
            model_id="tablet",
            version="tablet-v1",
        )
    )
    assert promoted["status"] == "ok"
    service.promote_registered.assert_awaited_once()
    assert service.promote_registered.await_args.args[:4] == ("alice", "source-thread", "tablet", "tablet-v1")
    assert service.promote_registered.await_args.kwargs["workflow"] is workflow


@pytest.mark.asyncio
async def test_promote_accepts_explicit_chinese_confirmation_with_model_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    service = MagicMock()
    service.promote_registered = AsyncMock(return_value={"model_id": "tablet", "version": "tablet-v1", "status": "ready"})
    monkeypatch.setattr(model_tools, "_service", lambda: service)
    workflow = {"stage": "registered", "approval_status": "approved", "attempt_evidence": {}}

    payload = json.loads(
        await model_tools.nir_model_promote_tool.coroutine(
            runtime=_runtime(
                "确认持久化保存 local-real-tablets-pls（version: local-real-tablets-pls-v1）到本机跨会话模型库，仅用于本地开发验收",
                workflow=workflow,
            ),
            model_id="tablet",
            version="tablet-v1",
        )
    )

    assert payload["status"] == "ok"
    service.promote_registered.assert_awaited_once()


@pytest.mark.asyncio
async def test_promote_rejects_latest_user_negation(monkeypatch: pytest.MonkeyPatch) -> None:
    service = MagicMock()
    service.promote_registered = AsyncMock(return_value={"model_id": "tablet", "version": "tablet-v1", "status": "ready"})
    monkeypatch.setattr(model_tools, "_service", lambda: service)
    workflow = {"stage": "registered", "approval_status": "approved", "attempt_evidence": {}}

    payload = json.loads(
        await model_tools.nir_model_promote_tool.coroutine(
            runtime=_runtime("不要持久化保存这个模型到跨会话模型库", workflow=workflow),
            model_id="tablet",
            version="tablet-v1",
        )
    )

    assert payload["code"] == "confirmation_required"
    service.promote_registered.assert_not_called()


@pytest.mark.asyncio
async def test_attach_returns_prediction_ready_model_path(monkeypatch: pytest.MonkeyPatch) -> None:
    service = MagicMock()
    service.attach_to_thread = AsyncMock(
        return_value={
            "status": "attached",
            "model_id": "tablet",
            "version": "tablet-v1",
            "model_path": "/mnt/user-data/outputs/models/tablet/tablet-v1/model.pkl",
        }
    )
    monkeypatch.setattr(model_tools, "_service", lambda: service)

    payload = json.loads(
        await model_tools.nir_model_attach_tool.coroutine(
            runtime=_runtime("使用我以前保存的模型"),
            model_id="tablet",
            version="tablet-v1",
        )
    )
    assert payload["status"] == "ok"
    assert "nir_workflow" in payload["next_action"]
    assert "nir_predict" in payload["next_action"]
    kwargs = service.attach_to_thread.await_args.kwargs
    assert callable(kwargs["sync_callback"])


@pytest.mark.asyncio
async def test_list_and_get_return_only_service_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    service = MagicMock()
    service.list_models = AsyncMock(return_value=[{"model_id": "tablet", "version": "tablet-v1"}])
    service.get_model = AsyncMock(return_value={"model_id": "tablet", "version": "tablet-v1"})
    monkeypatch.setattr(model_tools, "_service", lambda: service)
    runtime = _runtime("查看模型")

    listed = json.loads(await model_tools.nir_model_list_tool.coroutine(runtime=runtime, include_archived=False))
    fetched = json.loads(await model_tools.nir_model_get_tool.coroutine(runtime=runtime, model_id="tablet", version="tablet-v1"))
    assert listed["count"] == 1
    assert fetched["model"]["version"] == "tablet-v1"
