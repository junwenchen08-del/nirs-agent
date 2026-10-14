"""Gateway authentication and ownership gates for NIR dataset saves."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import httpx
import numpy as np
import pytest
import pytest_asyncio
from _router_auth_helpers import make_authed_test_app
from fastapi import HTTPException
from scipy.io import savemat
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.gateway.auth.models import User
from app.gateway.routers import nir_datasets
from deerflow.community.nir.datasets.service import DatasetService
from deerflow.config.app_config import AppConfig
from deerflow.config.nir_library_config import NIRLibraryConfig
from deerflow.config.paths import Paths
from deerflow.persistence.base import Base
from deerflow.persistence.thread_meta.model import ThreadMetaRow


@pytest.mark.asyncio
async def test_workspace_routes_save_paginate_and_reject_spectral_edits(client_and_paths):
    client, paths, app, owner = client_and_paths
    upload = paths.sandbox_uploads_dir("thread1", user_id=str(owner.id)) / "workspace.csv"
    upload.parent.mkdir(parents=True, exist_ok=True)
    upload.write_text("sample_id,moisture,1000,1100,1200\na,1,1,2,3\nb,2,2,3,4\nc,3,3,4,5\n", encoding="utf-8")
    saved = await client.post("/api/nir/datasets", json={"thread_id": "thread1", "virtual_path": "/mnt/user-data/uploads/workspace.csv", "name": "样本", "save_confirmed": True})
    id = saved.json()["id"]
    workspace = (await client.get(f"/api/nir/datasets/{id}/workspace")).json()
    assert workspace["sample_count"] == 3
    rows = await client.get(f"/api/nir/datasets/{id}/workspace/rows?offset=1&limit=1")
    assert rows.json()["rows"][0]["name"] == "b"
    assert (await client.get(f"/api/nir/datasets/{id}/workspace/spectra?indices=2")).json()["curves"][0]["values"] == [3, 4, 5]
    body = {"source_sha256": saved.json()["sha256"], "profile_id": None, "usages": [1, 2, 3], "range": [1100, 1200]}
    assert (await client.put(f"/api/nir/datasets/{id}/workspace", json={**body, "spectral_values": [[999]]})).status_code == 422
    assert (await client.put(f"/api/nir/datasets/{id}/workspace", json=body)).status_code == 200
    assert (await client.put(f"/api/nir/datasets/{id}/workspace", json=body)).status_code == 409
    assert (await client.get(f"/api/nir/datasets/{id}/workspace/spectra?indices=9")).status_code == 400
    assert (await client.get(f"/api/nir/datasets/{id}/workspace/rows?limit=200")).status_code == 422
    assert (await client.get("/api/nir/datasets/missing/workspace")).status_code == 404


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
async def test_local_upload_is_reusable_without_a_source_thread(client_and_paths):
    client, paths, app, owner = client_and_paths
    content = b"sample_id,moisture,1000,1100,1200\na,1,1,2,3\nb,2,2,3,4\n"
    app.state.thread_store.get.return_value = None
    saved = await client.post("/api/nir/datasets/upload", files={"file": ("玉米.csv", content, "text/csv")}, data={"name": "玉米水分"})
    assert saved.status_code == 200
    dataset = saved.json()
    assert dataset["name"] == "玉米水分"
    assert dataset["original_filename"] == "玉米.csv"
    assert dataset["size_bytes"] == len(content)
    app.state.thread_store.get.assert_not_awaited()
    assert not paths.sandbox_uploads_dir("thread1", user_id=str(owner.id)).exists()
    duplicate = await client.post("/api/nir/datasets/upload", files={"file": ("copy.txt", content)})
    assert duplicate.json()["id"] == dataset["id"]
    assert duplicate.json()["reused_existing"] is True
    assert (await client.get("/api/nir/datasets")).json()["count"] == 1
    assert (await client.get(f"/api/nir/datasets/{dataset['id']}/workspace")).json()["sample_count"] == 2
    app.state.thread_store.get.return_value = {"thread_id": "thread1", "user_id": str(owner.id)}
    mounted = await client.post(f"/api/nir/datasets/{dataset['id']}/attach", json={"thread_id": "thread1", "idempotency_key": "local-upload"})
    assert mounted.status_code == 200
    assert mounted.json()["status"] == "attached"
    filename = Path(mounted.json()["virtual_path"]).name
    assert (paths.sandbox_uploads_dir("thread1", user_id=str(owner.id)) / filename).read_bytes() == content


@pytest.mark.asyncio
@pytest.mark.parametrize("filename,content,code", [("data.exe", b"x", "invalid_filename"), ("../data.csv", b"x", "invalid_filename"), ("empty.csv", b"", "empty_file")])
async def test_local_upload_rejects_invalid_files_without_assets(client_and_paths, filename, content, code):
    client, _paths, _app, _owner = client_and_paths
    result = await client.post("/api/nir/datasets/upload", files={"file": (filename, content)})
    assert result.status_code == 400
    assert result.json()["detail"]["code"] == code
    assert (await client.get("/api/nir/datasets")).json()["count"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("extension", ["txt", "MAT"])
async def test_local_upload_formats_open_in_the_workspace(client_and_paths, extension):
    client, _paths, _app, _owner = client_and_paths
    if extension == "MAT":
        stream = io.BytesIO()
        savemat(stream, {"X": np.arange(12).reshape(4, 3), "y": np.arange(4), "wv": np.array([1000, 1100, 1200])})
        content = stream.getvalue()
    else:
        content = b"1 2 3\n2 3 4\n3 4 5\n4 5 6\n"
    saved = await client.post("/api/nir/datasets/upload", files={"file": (f"光谱.{extension}", content)})
    assert saved.status_code == 200
    assert saved.json()["name"] == f"光谱.{extension}"
    workspace = await client.get(f"/api/nir/datasets/{saved.json()['id']}/workspace")
    assert workspace.status_code == 200
    assert workspace.json()["sample_count"] == 4
    assert workspace.json()["feature_count"] == 3


@pytest.mark.asyncio
async def test_visual_parsing_routes_preview_without_writes_and_save_owned_version(client_and_paths):
    from test_nir_dataset_parsing import choose, nested_mat

    client, paths, app, owner = client_and_paths
    source = paths.user_dir(str(owner.id)) / "synthetic.mat"
    source.parent.mkdir(parents=True, exist_ok=True)
    nested_mat(source)
    uploaded = await client.post("/api/nir/datasets/upload", files={"file": ("nested.mat", source.read_bytes())})
    dataset = uploaded.json()
    schema = (await client.get(f"/api/nir/datasets/{dataset['id']}/parsing")).json()
    body = {"source_sha256": dataset["sha256"], "profile_id": None, "selection": choose(schema), "task_type": "calibration"}
    preview = await client.post(f"/api/nir/datasets/{dataset['id']}/parsing/preview", json=body)
    assert preview.status_code == 200 and preview.json()["sample_count"] == 6
    assert (await client.get(f"/api/nir/datasets/{dataset['id']}/profiles")).json()["count"] == 0
    invalid = await client.put(f"/api/nir/datasets/{dataset['id']}/parsing", json={**body, "selection": {"raw_spectra": [999]}})
    assert invalid.status_code == 422
    saved = await client.put(f"/api/nir/datasets/{dataset['id']}/parsing", json=body)
    assert saved.status_code == 200 and saved.json()["profile_status"] == "confirmed"
    assert (await client.put(f"/api/nir/datasets/{dataset['id']}/parsing", json=body)).status_code == 409
    assert (await client.get("/api/nir/datasets/missing/parsing")).status_code == 404


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
    preview = await client.get(f"/api/nir/datasets/{asset_id}/preview")
    assert preview.status_code == 200
    assert preview.json()["dataset_id"] == asset_id
    assert preview.json()["source_sha256"] == saved.json()["sha256"]
    workspace = await client.get(f"/api/nir/datasets/{asset_id}/workspace")
    assert workspace.status_code == 422  # Ambiguous x/y columns are never guessed.
    assert (await client.get("/api/nir/datasets/missing/preview")).status_code == 404
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
