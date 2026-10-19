"""Gateway run configuration cannot rewind a sealed task or switch threads."""

import asyncio
import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from app.gateway import services


def checkpoint(identifier, workflow=None):
    values = {"nir_workflow": workflow} if workflow is not None else {}
    return SimpleNamespace(config={"configurable": {"thread_id": "owned", "checkpoint_id": identifier}}, checkpoint={"channel_values": values})


def sealed(**updates):
    return {
        "search_protocol": "sealed_holdout",
        "decision_strategy": "llm_autonomous",
        "autonomous_rounds": True,
        "autonomous_delivery": True,
        "search_session_id": "search_current",
        **updates,
    }


class Checkpointer:
    def __init__(self, latest, selected):
        self.latest = latest
        self.selected = selected
        self.reads = []

    async def aget_tuple(self, config):
        self.reads.append(copy.deepcopy(config))
        return self.selected if config["configurable"].get("checkpoint_id") else self.latest


def apply(monkeypatch, latest, selected, *, identifier="current", checkpoint_map=None, namespace=""):
    reader = Checkpointer(latest, selected)
    monkeypatch.setattr(services, "get_checkpointer", lambda _: reader)
    body = SimpleNamespace(checkpoint={"checkpoint_id": identifier, "checkpoint_ns": namespace, "checkpoint_map": checkpoint_map}, checkpoint_id=None)
    config = {"configurable": {"thread_id": "forged"}}
    asyncio.run(services.apply_checkpoint_to_run_config(config, body=body, thread_id="owned", request=SimpleNamespace()))
    return config, reader


@pytest.mark.parametrize(
    "request_config",
    [
        {"configurable": {"thread_id": "other", "model_name": "configured"}},
        {"context": {"thread_id": "other"}},
        {"context": {"thread_id": "other"}, "configurable": {"thread_id": "third"}},
    ],
)
def test_run_thread_identity_is_server_owned(request_config):
    config = services.build_run_config("owned", request_config, None)
    assert config["configurable"]["thread_id"] == "owned"
    if "context" in config:
        assert config["context"]["thread_id"] == "owned"


@pytest.mark.parametrize("container", [None, "configurable", "context"])
@pytest.mark.parametrize("field", ["checkpoint_id", "checkpoint_ns", "checkpoint_map"])
def test_config_cannot_supply_unvalidated_checkpoint_locators(container, field):
    request_config = {field: "old"} if container is None else {container: {field: "old"}}
    with pytest.raises(ValueError, match="checkpoint"):
        services.build_run_config("owned", request_config, None)


@pytest.mark.parametrize("old_workflow", [None, {"search_protocol": "legacy"}, sealed(), sealed(search_session_id="search_other")])
def test_sealed_task_rejects_older_checkpoint(monkeypatch, old_workflow):
    with pytest.raises(HTTPException) as exc:
        apply(monkeypatch, checkpoint("current", sealed()), checkpoint("old", old_workflow), identifier="old")
    assert exc.value.status_code == 409


@pytest.mark.parametrize("selected_workflow", [None, sealed(search_session_id="search_other"), sealed(decision_strategy="legacy"), sealed(autonomous_rounds=False), sealed(autonomous_delivery=False)])
def test_current_identifier_cannot_hide_changed_sealed_binding(monkeypatch, selected_workflow):
    with pytest.raises(HTTPException) as exc:
        apply(monkeypatch, checkpoint("current", sealed()), checkpoint("current", selected_workflow))
    assert exc.value.status_code == 409


def test_current_sealed_checkpoint_resume_remains_available(monkeypatch):
    current = checkpoint("current", sealed())
    config, reader = apply(monkeypatch, current, current, checkpoint_map={"": "current"})
    assert config["configurable"] == {"thread_id": "owned", "checkpoint_id": "current", "checkpoint_ns": "", "checkpoint_map": {"": "current"}}
    assert len(reader.reads) == 2
    assert all(item["configurable"]["thread_id"] == "owned" for item in reader.reads)


def test_legacy_checkpoint_history_remains_available(monkeypatch):
    config, _ = apply(monkeypatch, checkpoint("current", {"search_protocol": "legacy"}), checkpoint("old", {"search_protocol": "legacy"}), identifier="old")
    assert config["configurable"]["checkpoint_id"] == "old"


def test_sealed_resume_cannot_use_other_namespace_or_map(monkeypatch):
    current = checkpoint("current", sealed())
    for options in ({"namespace": "other"}, {"checkpoint_map": {"": "old"}}, {"checkpoint_map": {"other": "old"}}):
        with pytest.raises(HTTPException) as exc:
            apply(monkeypatch, current, current, **options)
        assert exc.value.status_code == 409


@pytest.mark.parametrize("invalid_config", [True, False])
def test_rejected_run_does_not_leave_pending_record(monkeypatch, invalid_config):
    manager = SimpleNamespace(create_or_reject=AsyncMock())
    context = SimpleNamespace(thread_store=SimpleNamespace(check_access=AsyncMock(return_value=True)))
    monkeypatch.setattr(services, "get_stream_bridge", lambda _: object())
    monkeypatch.setattr(services, "get_run_manager", lambda _: manager)
    monkeypatch.setattr(services, "get_run_context", lambda _: context)
    monkeypatch.setattr(services, "get_trusted_internal_owner_user_id", lambda _: "alice")
    reader = Checkpointer(checkpoint("current", sealed()), checkpoint("old", {"search_protocol": "legacy"}))
    monkeypatch.setattr(services, "get_checkpointer", lambda _: reader)
    request = SimpleNamespace(state=SimpleNamespace(user=SimpleNamespace(id="alice", system_role="user")))
    body = SimpleNamespace(
        on_disconnect="continue",
        context={},
        input=None,
        config={"configurable": {"checkpoint_id": "old"}} if invalid_config else {},
        metadata={},
        assistant_id="lead_agent",
        multitask_strategy="reject",
        checkpoint=None,
        checkpoint_id=None if invalid_config else "old",
    )
    with pytest.raises(HTTPException) as exc:
        asyncio.run(services.start_run(body, "owned", request))
    assert exc.value.status_code == (400 if invalid_config else 409)
    manager.create_or_reject.assert_not_awaited()
