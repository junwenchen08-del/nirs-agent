"""Agent-facing dataset tools keep results bounded and require save consent."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain.tools import ToolRuntime
from langchain_core.messages import HumanMessage

from deerflow.community.nir import dataset_tools


def _runtime(message: str) -> ToolRuntime:
    return ToolRuntime(
        state={"messages": [HumanMessage(content=message)]},
        context={"user_id": "alice", "thread_id": "thread1", "run_id": "run-one"},
        config={"configurable": {"thread_id": "thread1"}},
        stream_writer=lambda _: None,
        tools=[],
        tool_call_id="call-1",
        store=None,
    )


@pytest.mark.asyncio
async def test_save_tool_requires_latest_explicit_long_term_save_request(monkeypatch: pytest.MonkeyPatch) -> None:
    service = MagicMock()
    service.save_from_thread_upload = AsyncMock(return_value={"id": "ds_one", "reused_existing": False})
    monkeypatch.setattr(dataset_tools, "_service", lambda: service)

    denied = json.loads(
        await dataset_tools.nir_dataset_save_tool.coroutine(
            runtime=_runtime("请检查这个数据"),
            virtual_path="/mnt/user-data/uploads/data.csv",
            name="Data",
        )
    )
    assert denied["code"] == "confirmation_required"
    service.save_from_thread_upload.assert_not_called()

    saved = json.loads(
        await dataset_tools.nir_dataset_save_tool.coroutine(
            runtime=_runtime("请把这份数据长期保存到数据集库"),
            virtual_path="/mnt/user-data/uploads/data.csv",
            name="Data",
        )
    )
    assert saved["status"] == "ok"
    service.save_from_thread_upload.assert_awaited_once_with(
        "alice",
        "thread1",
        "/mnt/user-data/uploads/data.csv",
        "Data",
        save_confirmed=True,
    )


@pytest.mark.asyncio
async def test_get_tool_omits_full_profile_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    service = MagicMock()
    service.get_dataset = AsyncMock(return_value={"id": "ds_one", "status": "ready"})
    service.list_profiles = AsyncMock(
        return_value=[
            {
                "id": "dsp_one",
                "profile_status": "confirmed",
                "mapping": {"spectral_columns": [str(index) for index in range(1000)]},
                "mapping_sha256": "a" * 64,
            }
        ]
    )
    monkeypatch.setattr(dataset_tools, "_service", lambda: service)
    payload = json.loads(await dataset_tools.nir_dataset_get_tool.coroutine(runtime=_runtime("查看数据集"), dataset_id="ds_one"))
    assert payload["status"] == "ok"
    assert "mapping" not in payload["profiles"][0]
    assert payload["profiles"][0]["mapping_sha256"] == "a" * 64


@pytest.mark.asyncio
async def test_attach_tool_returns_fresh_inspection_instruction(monkeypatch: pytest.MonkeyPatch) -> None:
    service = MagicMock()
    service.attach_to_thread = AsyncMock(
        return_value={
            "attachment_id": "use_one",
            "dataset_id": "ds_one",
            "profile_id": None,
            "workflow_project_id": "nir-project-one",
            "virtual_path": "/mnt/user-data/uploads/dataset-ds_one.csv",
            "source_sha256": "b" * 64,
            "status": "attached",
        }
    )
    monkeypatch.setattr(dataset_tools, "_service", lambda: service)
    payload = json.loads(await dataset_tools.nir_dataset_attach_tool.coroutine(runtime=_runtime("使用之前的数据"), dataset_id="ds_one"))
    assert payload["status"] == "ok"
    assert "nir_inspect" in payload["next_action"]
    kwargs = service.attach_to_thread.await_args.kwargs
    assert kwargs["idempotency_key"] == "agent:run-one:ds_one:raw"
    assert kwargs["run_id"] == "run-one"
    assert kwargs["prepare_workflow"] is True
    assert callable(kwargs["sync_callback"])
