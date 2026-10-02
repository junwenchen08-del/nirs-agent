"""Gateway ownership and checkpoint gates for the persistent NIR model library."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import httpx
import pytest
from _router_auth_helpers import make_authed_test_app

from app.gateway.auth.models import User
from app.gateway.routers import nir_models


@pytest.fixture
def model_router_client():
    owner = User(email="model-owner@example.com", password_hash="x", system_role="user", id=uuid4())
    service = MagicMock()
    service.list_models = AsyncMock(return_value=[{"model_id": "tablet", "version": "tablet-v1"}])
    service.get_model = AsyncMock(return_value={"model_id": "tablet", "version": "tablet-v1"})
    service.promote_registered = AsyncMock(return_value={"model_id": "tablet", "version": "tablet-v1", "status": "ready"})
    service.attach_to_thread = AsyncMock(return_value={"status": "attached", "model_path": "/mnt/user-data/outputs/models/tablet/tablet-v1/model.pkl"})
    service.archive_model = AsyncMock(return_value={"model_id": "tablet", "version": "tablet-v1", "status": "archived"})
    service.delete_model = AsyncMock(return_value={"model_id": "tablet", "version": "tablet-v1", "status": "deleted"})
    app = make_authed_test_app(user_factory=lambda: owner)
    app.state.checkpointer = MagicMock()
    app.state.checkpointer.aget_tuple = AsyncMock(
        return_value=SimpleNamespace(
            checkpoint={
                "channel_values": {
                    "nir_workflow": {
                        "stage": "registered",
                        "approval_status": "approved",
                        "attempt_evidence": {"run_id": "run-one"},
                    }
                }
            }
        )
    )
    app.include_router(nir_models.router)
    app.dependency_overrides[nir_models.get_model_service] = lambda: service
    return app, owner, service


@pytest.mark.asyncio
async def test_model_routes_use_authenticated_owner_and_checkpoint(model_router_client) -> None:
    app, owner, service = model_router_client
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        listed = await client.get("/api/nir/models")
        fetched = await client.get("/api/nir/models/tablet", params={"version": "tablet-v1"})
        promoted = await client.post(
            "/api/nir/models/promote",
            json={
                "source_thread_id": "source-thread",
                "model_id": "tablet",
                "version": "tablet-v1",
                "idempotency_key": "promote-router",
            },
        )
        attached = await client.post(
            "/api/nir/models/tablet/versions/tablet-v1/attach",
            json={"thread_id": "target-thread"},
        )
        archived = await client.post("/api/nir/models/tablet/versions/tablet-v1/archive")
        deleted = await client.request(
            "DELETE",
            "/api/nir/models/tablet/versions/tablet-v1",
            json={"confirmation": "tablet:tablet-v1"},
        )

    assert listed.json()["count"] == 1
    assert fetched.json()["version"] == "tablet-v1"
    assert promoted.status_code == 200
    assert attached.json()["status"] == "attached"
    assert archived.json()["status"] == "archived"
    assert deleted.json()["status"] == "deleted"
    service.promote_registered.assert_awaited_once()
    assert service.promote_registered.await_args.args[:4] == (str(owner.id), "source-thread", "tablet", "tablet-v1")
    assert service.attach_to_thread.await_args.args[:4] == (str(owner.id), "tablet", "tablet-v1", "target-thread")
    service.delete_model.assert_awaited_once_with(
        str(owner.id),
        "tablet",
        "tablet-v1",
        confirmation="tablet:tablet-v1",
    )


@pytest.mark.asyncio
async def test_promote_fails_closed_without_registered_checkpoint(model_router_client) -> None:
    app, _owner, service = model_router_client
    app.state.checkpointer.aget_tuple.return_value = None
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/nir/models/promote",
            json={"source_thread_id": "source-thread", "model_id": "tablet", "version": "tablet-v1"},
        )
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "registration_required"
    service.promote_registered.assert_not_called()
