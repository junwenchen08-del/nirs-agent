"""Upload-to-library integration using isolated SQLite and synthetic bytes."""

from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio
from _router_auth_helpers import make_authed_test_app
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.gateway.auth.models import User
from app.gateway.deps import get_config
from app.gateway.routers import nir_datasets, uploads
from deerflow.community.nir.datasets.service import DatasetService
from deerflow.config.nir_library_config import NIRLibraryConfig
from deerflow.config.paths import Paths
from deerflow.persistence.base import Base
from deerflow.persistence.thread_meta.sql import ThreadMetaRepository
from deerflow.uploads import manager


@pytest_asyncio.fixture
async def upload_library(tmp_path, monkeypatch):
    owner = User(email="upload-owner@example.com", password_hash="x", system_role="user", id=uuid4())
    engine = create_async_engine(f"sqlite+aiosqlite:///{(tmp_path / 'library.db').as_posix()}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    paths = Paths(tmp_path / "home")
    policy = NIRLibraryConfig(enabled=True, min_free_disk_bytes=0)
    config = SimpleNamespace(nir_library=policy)
    service = DatasetService(factory, paths, policy)
    app = make_authed_test_app(user_factory=lambda: owner)
    app.state.thread_store = ThreadMetaRepository(factory)
    app.include_router(uploads.router)
    app.include_router(nir_datasets.router)
    app.dependency_overrides[get_config] = lambda: config
    app.dependency_overrides[nir_datasets.get_dataset_service] = lambda: service
    monkeypatch.setattr(uploads, "get_paths", lambda: paths)
    monkeypatch.setattr(manager, "get_paths", lambda: paths)
    monkeypatch.setattr(uploads, "get_session_factory", lambda: factory, raising=False)
    monkeypatch.setattr(uploads, "get_effective_user_id", lambda: str(owner.id))
    monkeypatch.setattr(uploads, "get_sandbox_provider", lambda: SimpleNamespace(uses_thread_data_mounts=True))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        yield client, service, paths, owner, config, app
    await engine.dispose()


async def _upload(client, thread="new-thread", name="光谱.CSV", content=b"x,y\n1,2\n"):
    return await client.post(f"/api/threads/{thread}/uploads", files={"files": (name, content, "text/csv")})


@pytest.mark.asyncio
async def test_upload_autosaves_without_agent_or_explicit_save_and_survives_thread_copy_deletion(upload_library):
    client, service, paths, owner, _, app = upload_library
    response = await _upload(client)
    assert response.status_code == 200
    saved = response.json()["files"][0]["dataset_library"]
    assert saved["status"] == "saved"
    datasets = (await client.get("/api/nir/datasets")).json()["datasets"]
    assert [row["id"] for row in datasets] == [saved["dataset_id"]]
    assert datasets[0]["name"] == "光谱.CSV"
    assert (await app.state.thread_store.get("new-thread", user_id=None))["user_id"] == str(owner.id)
    assert await service.list_profiles(str(owner.id), saved["dataset_id"]) == []
    paths.sandbox_uploads_dir("new-thread", user_id=str(owner.id)).joinpath("光谱.CSV").unlink()
    assert (await service.get_dataset(str(owner.id), saved["dataset_id"]))["status"] == "ready"


@pytest.mark.asyncio
async def test_duplicate_content_reuses_library_asset_and_changed_content_creates_another(upload_library):
    client, _, _, _, _, _ = upload_library
    first = (await _upload(client)).json()["files"][0]["dataset_library"]
    duplicate = (await _upload(client, thread="another-thread", name="renamed.csv")).json()["files"][0]["dataset_library"]
    assert duplicate["status"] == "reused"
    assert duplicate["dataset_id"] == first["dataset_id"]
    changed = (await _upload(client, content=b"x,y\n1,3\n")).json()["files"][0]["dataset_library"]
    assert changed["status"] == "saved"
    assert changed["dataset_id"] != first["dataset_id"]
    assert (await client.get("/api/nir/datasets")).json()["count"] == 2


@pytest.mark.asyncio
async def test_autosave_failure_preserves_current_upload_and_reports_failure(upload_library):
    client, _, paths, owner, config, _ = upload_library
    config.nir_library.max_user_dataset_bytes = 1
    response = await _upload(client)
    assert response.status_code == 200
    assert response.json()["success"] is True
    result = response.json()["files"][0]["dataset_library"]
    assert result == {"status": "failed", "dataset_id": None, "error_code": "quota_exceeded"}
    assert paths.sandbox_uploads_dir("new-thread", user_id=str(owner.id)).joinpath("光谱.CSV").is_file()
    assert (await client.get("/api/nir/datasets")).json()["count"] == 0


@pytest.mark.asyncio
async def test_missing_database_is_reported_without_losing_upload(upload_library, monkeypatch):
    client, _, _, _, _, _ = upload_library
    monkeypatch.setattr(uploads, "get_session_factory", lambda: None)
    response = await _upload(client)
    assert response.status_code == 200
    assert response.json()["files"][0]["dataset_library"]["error_code"] == "database_unavailable"


@pytest.mark.asyncio
async def test_disabled_policy_and_other_file_types_do_not_autosave(upload_library):
    client, _, _, _, config, _ = upload_library
    response = await _upload(client, name="report.pdf")
    assert response.json()["files"][0].get("dataset_library") is None
    config.nir_library.enabled = False
    response = await _upload(client)
    assert response.json()["files"][0].get("dataset_library") is None
    config.nir_library.enabled = True
    config.nir_library.auto_save_uploads = False
    response = await _upload(client)
    assert response.json()["files"][0].get("dataset_library") is None
    assert (await client.get("/api/nir/datasets")).json()["count"] == 0


@pytest.mark.asyncio
async def test_other_owner_is_denied_and_legacy_unowned_thread_is_not_claimed(upload_library):
    client, _, _, _, _, app = upload_library
    await app.state.thread_store.create("other-thread", user_id="other-owner")
    response = await _upload(client, thread="other-thread")
    assert response.status_code == 404
    await app.state.thread_store.create("legacy-thread", user_id=None)
    response = await _upload(client, thread="legacy-thread")
    assert response.status_code == 200
    assert response.json()["files"][0]["dataset_library"]["error_code"] == "thread_not_owned"
    assert (await app.state.thread_store.get("legacy-thread", user_id=None))["user_id"] is None
    assert (await client.get("/api/nir/datasets")).json()["count"] == 0


@pytest.mark.asyncio
async def test_rejected_batch_does_not_save_earlier_part(upload_library):
    client, _, paths, owner, config, _ = upload_library
    config.uploads = SimpleNamespace(max_file_size=10)
    response = await client.post(
        "/api/threads/new-thread/uploads",
        files=[("files", ("first.csv", b"x,y\n1,2\n")), ("files", ("large.csv", b"x" * 11))],
    )
    assert response.status_code == 413
    assert (await client.get("/api/nir/datasets")).json()["count"] == 0
    assert not paths.sandbox_uploads_dir("new-thread", user_id=str(owner.id)).joinpath("first.csv").exists()


@pytest.mark.asyncio
async def test_archived_duplicate_is_not_silently_restored(upload_library):
    client, service, _, owner, _, _ = upload_library
    saved = (await _upload(client)).json()["files"][0]["dataset_library"]
    await service.archive_dataset(str(owner.id), saved["dataset_id"])
    repeated = (await _upload(client)).json()["files"][0]["dataset_library"]
    assert repeated["status"] == "failed"
    assert repeated["error_code"] == "dataset_archived"


@pytest.mark.asyncio
@pytest.mark.parametrize("same_owner", [True, False])
async def test_concurrent_thread_creation_rechecks_winning_owner(upload_library, monkeypatch, same_owner):
    client, _, _, owner, _, app = upload_library
    original_create = app.state.thread_store.create

    async def raced_create(thread_id, **kwargs):
        await original_create(thread_id, user_id=str(owner.id) if same_owner else "other-owner")
        return await original_create(thread_id, **kwargs)

    monkeypatch.setattr(app.state.thread_store, "create", raced_create)
    response = await _upload(client)
    assert response.status_code == 200
    outcome = response.json()["files"][0]["dataset_library"]
    assert outcome["status"] == ("saved" if same_owner else "failed")
    if not same_owner:
        assert outcome["error_code"] == "thread_not_owned"
    assert (await client.get("/api/nir/datasets")).json()["count"] == (1 if same_owner else 0)
