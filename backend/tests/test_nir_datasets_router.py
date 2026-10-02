"""Gateway authentication and ownership gates for NIR dataset saves."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio
from _router_auth_helpers import make_authed_test_app
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.gateway.auth.models import User
from app.gateway.routers import nir_datasets
from deerflow.community.nir.datasets.service import DatasetService
from deerflow.config.app_config import AppConfig
from deerflow.config.nir_library_config import NIRLibraryConfig
from deerflow.config.paths import Paths
from deerflow.persistence.base import Base
from deerflow.persistence.thread_meta.model import ThreadMetaRow


def test_dataset_api_is_fail_closed_by_default() -> None:
    config = AppConfig.model_validate({"sandbox": {"use": "deerflow.sandbox.local:LocalSandboxProvider"}})
    with pytest.raises(HTTPException) as exc:
        nir_datasets.get_dataset_service(config)
    assert exc.value.status_code == 404
    assert exc.value.detail["code"] == "feature_disabled"


@pytest_asyncio.fixture
async def client_and_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    owner = User(email="nir-owner@example.com", password_hash="x", system_role="user", id=uuid4())
    engine = create_async_engine(f"sqlite+aiosqlite:///{(tmp_path / 'router.db').as_posix()}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    paths = Paths(tmp_path / "home")
    service = DatasetService(async_sessionmaker(engine, expire_on_commit=False), paths, NIRLibraryConfig(enabled=True, min_free_disk_bytes=0))
    async with service.session_factory() as session:
        session.add(ThreadMetaRow(thread_id="thread1", user_id=str(owner.id), status="idle", metadata_json={}))
        await session.commit()

    async def no_remote_sync(_owner: str, _thread_id: str, _virtual_path: str, _local_path: Path) -> None:
        return None

    monkeypatch.setattr(nir_datasets, "_sync_attachment_to_sandbox", no_remote_sync)
    app = make_authed_test_app(user_factory=lambda: owner)
    app.state.thread_store.get = AsyncMock(return_value={"thread_id": "thread1", "user_id": str(owner.id)})
    app.include_router(nir_datasets.router)
    app.include_router(nir_datasets.storage_router)
    app.dependency_overrides[nir_datasets.get_dataset_service] = lambda: service
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        yield client, paths, app, owner
    await engine.dispose()


@pytest.mark.asyncio
async def test_save_requires_owned_existing_thread_and_confirmation(client_and_paths) -> None:
    client, paths, app, owner = client_and_paths
    upload = paths.sandbox_uploads_dir("thread1", user_id=str(owner.id)) / "data.csv"
    upload.parent.mkdir(parents=True, exist_ok=True)
    upload.write_bytes(b"x,y\n1,2\n")
    body = {"thread_id": "thread1", "virtual_path": "/mnt/user-data/uploads/data.csv", "name": "Tablet"}

    app.state.thread_store.get.return_value = None
    denied = await client.post("/api/nir/datasets", json={**body, "save_confirmed": True})
    assert denied.status_code == 404
    assert denied.json()["detail"]["code"] == "thread_not_found"

    app.state.thread_store.get.return_value = {"thread_id": "thread1", "user_id": "another-user"}
    denied = await client.post("/api/nir/datasets", json={**body, "save_confirmed": True})
    assert denied.status_code == 404

    app.state.thread_store.get.return_value = {"thread_id": "thread1", "user_id": str(owner.id)}
    denied = await client.post("/api/nir/datasets", json=body)
    assert denied.status_code == 400
    assert denied.json()["detail"]["code"] == "confirmation_required"

    saved = await client.post("/api/nir/datasets", json={**body, "save_confirmed": True})
    assert saved.status_code == 200
    asset_id = saved.json()["id"]
    assert (await client.get("/api/nir/datasets")).json()["count"] == 1
    usage = (await client.get("/api/nir/storage/usage")).json()
    assert usage["dataset_bytes"] == len(b"x,y\n1,2\n")
    assert usage["strict_total_quota"] is False
    reconciliation = (await client.get("/api/nir/storage/reconciliation")).json()
    assert reconciliation["missing_assets"] == 0
    assert reconciliation["byte_discrepancy"] == 0
    assert (await client.get(f"/api/nir/datasets/{asset_id}")).status_code == 200
    draft = await client.post(
        f"/api/nir/datasets/{asset_id}/profiles",
        json={
            "task_type": "calibration",
            "schema_status": "confirmed_mapping",
            "mapping": {"target_column": "y", "spectral_columns": ["x"]},
        },
    )
    assert draft.status_code == 200
    profile_id = draft.json()["id"]
    assert (await client.post(f"/api/nir/datasets/{asset_id}/profiles/{profile_id}/confirm")).json()["profile_status"] == "confirmed"
    assert (await client.get(f"/api/nir/datasets/{asset_id}/profiles")).json()["count"] == 1
    attached = await client.post(
        f"/api/nir/datasets/{asset_id}/attach",
        json={"thread_id": "thread1", "profile_id": profile_id, "idempotency_key": "router-attach"},
    )
    assert attached.status_code == 200
    assert attached.json()["status"] == "attached"
    assert (await client.get(f"/api/nir/datasets/{asset_id}/uses")).json()["count"] == 1
    assert (await client.patch(f"/api/nir/datasets/{asset_id}", json={"name": "New name"})).json()["name"] == "New name"
    assert (await client.post(f"/api/nir/datasets/{asset_id}/archive")).json()["status"] == "archived"
    deleted = await client.request(
        "DELETE",
        f"/api/nir/datasets/{asset_id}",
        json={"confirmation": asset_id},
    )
    assert deleted.status_code == 200
    assert deleted.json()["status"] == "deleted"
