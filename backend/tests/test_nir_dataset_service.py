"""Dataset-library save contracts; all spectra here are synthetic bytes."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from deerflow.community.nir.datasets import service as dataset_module
from deerflow.community.nir.datasets.service import DatasetError, DatasetService
from deerflow.config.nir_library_config import NIRLibraryConfig
from deerflow.config.paths import Paths
from deerflow.persistence.base import Base
from deerflow.persistence.nir_library.model import NIRDatasetUseRow, NIRModelVersionRow
from deerflow.persistence.thread_meta.model import ThreadMetaRow


@pytest_asyncio.fixture
async def library(tmp_path: Path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{(tmp_path / 'library.db').as_posix()}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    paths = Paths(tmp_path / "home")
    policy = NIRLibraryConfig(enabled=True, min_free_disk_bytes=0)
    service = DatasetService(async_sessionmaker(engine, expire_on_commit=False), paths, policy)
    async with service.session_factory() as session:
        session.add_all(
            [
                ThreadMetaRow(thread_id="thread1", user_id="alice", status="idle", metadata_json={}),
                ThreadMetaRow(thread_id="thread2", user_id="bob", status="idle", metadata_json={}),
                ThreadMetaRow(thread_id="source-thread", user_id="alice", status="idle", metadata_json={}),
            ]
        )
        await session.commit()
    try:
        yield service, paths
    finally:
        await engine.dispose()


def _upload(paths: Paths, owner: str, thread: str, filename: str, content: bytes) -> str:
    target = paths.sandbox_uploads_dir(thread, user_id=owner) / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    return f"/mnt/user-data/uploads/{filename}"


async def _owned_thread(service: DatasetService, owner: str, thread_id: str) -> None:
    async with service.session_factory() as session:
        session.add(ThreadMetaRow(thread_id=thread_id, user_id=owner, status="idle", metadata_json={}))
        await session.commit()


@pytest.mark.asyncio
async def test_confirmed_save_is_immutable_deduplicated_and_owner_scoped(library) -> None:
    service, paths = library
    data = b"wv,x\n1000,0.12\n1001,0.14\n"
    source = _upload(paths, "alice", "thread1", "spectrum.csv", data)
    first = await service.save_from_thread_upload("alice", "thread1", source, "Tablets", save_confirmed=True)
    assert first["name"] == "Tablets"
    assert first["size_bytes"] == len(data)
    assert first["reused_existing"] is False
    assert "storage_relpath" not in first
    assert (paths.user_nir_dataset_dir("alice", first["id"]) / "source.csv").read_bytes() == data
    assert (paths.user_nir_dataset_dir("alice", first["id"]) / "manifest.json").is_file()
    repeated = await service.save_from_thread_upload("alice", "thread1", source, "Again", save_confirmed=True)
    assert repeated == {**first, "reused_existing": True}
    assert len(await service.list_datasets("alice")) == 1
    assert await service.list_datasets("bob") == []

    bob_source = _upload(paths, "bob", "thread2", "spectrum.csv", data)
    bob = await service.save_from_thread_upload("bob", "thread2", bob_source, "Bob", save_confirmed=True)
    assert bob["id"] != first["id"]
    with pytest.raises(DatasetError) as exc:
        await service.get_dataset("bob", first["id"])
    assert exc.value.code == "dataset_not_found"


@pytest.mark.asyncio
async def test_save_rejects_unconfirmed_or_non_upload_paths(library) -> None:
    service, paths = library
    source = _upload(paths, "alice", "thread1", "spectrum.csv", b"x,y\n1,2\n")
    with pytest.raises(DatasetError) as exc:
        await service.save_from_thread_upload("alice", "thread1", source, "A", save_confirmed=False)
    assert exc.value.code == "confirmation_required"
    for bad in ("/mnt/user-data/outputs/spectrum.csv", "/mnt/user-data/uploads/../spectrum.csv", "/mnt/user-data/uploads/dir/spectrum.csv"):
        with pytest.raises(DatasetError) as exc:
            await service.save_from_thread_upload("alice", "thread1", bad, "A", save_confirmed=True)
        assert exc.value.code == "invalid_source"
    assert await service.list_datasets("alice") == []


@pytest.mark.asyncio
async def test_save_enforces_quota_and_detects_tampering(library) -> None:
    service, paths = library
    service.policy.max_dataset_file_size = 6
    source = _upload(paths, "alice", "thread1", "spectrum.csv", b"x,y\n1,2\n")
    with pytest.raises(DatasetError) as exc:
        await service.save_from_thread_upload("alice", "thread1", source, "A", save_confirmed=True)
    assert exc.value.code == "file_too_large"
    assert await service.list_datasets("alice") == []

    service.policy.max_dataset_file_size = 100
    saved = await service.save_from_thread_upload("alice", "thread1", source, "A", save_confirmed=True)
    target = paths.user_nir_dataset_dir("alice", saved["id"]) / "source.csv"
    target.write_bytes(b"tampered")
    with pytest.raises(DatasetError) as exc:
        await service.get_dataset("alice", saved["id"])
    assert exc.value.code == "source_hash_mismatch"


@pytest.mark.asyncio
async def test_archive_preserves_bytes_and_requires_explicit_recovery(library) -> None:
    service, paths = library
    source = _upload(paths, "alice", "thread1", "spectrum.csv", b"x,y\n1,2\n")
    saved = await service.save_from_thread_upload("alice", "thread1", source, "A", save_confirmed=True)
    archived = await service.archive_dataset("alice", saved["id"])
    assert archived["status"] == "archived"
    assert (paths.user_nir_dataset_dir("alice", saved["id"]) / "source.csv").exists()
    with pytest.raises(DatasetError) as exc:
        await service.save_from_thread_upload("alice", "thread1", source, "A", save_confirmed=True)
    assert exc.value.code == "dataset_archived"


@pytest.mark.asyncio
async def test_delete_requires_archive_confirmation_and_no_live_model_reference(library) -> None:
    service, paths = library
    source = _upload(paths, "alice", "thread1", "spectrum.csv", b"x,y\n1,2\n")
    saved = await service.save_from_thread_upload("alice", "thread1", source, "A", save_confirmed=True)

    with pytest.raises(DatasetError) as exc:
        await service.delete_dataset("alice", saved["id"], confirmation=saved["id"])
    assert exc.value.code == "archive_required"

    await service.archive_dataset("alice", saved["id"])
    with pytest.raises(DatasetError) as exc:
        await service.delete_dataset("alice", saved["id"], confirmation="wrong")
    assert exc.value.code == "confirmation_required"

    async with service.session_factory() as session:
        referenced = NIRModelVersionRow(
            id="nmv_reference",
            owner_user_id="alice",
            model_id="tablet",
            version="v1",
            artifact_relpath="tablet/v1/model.pkl",
            artifact_size_bytes=1,
            artifact_sha256="a" * 64,
            metrics_sha256="b" * 64,
            training_data_sha256="c" * 64,
            validation_scope="independent_holdout_not_external",
            preprocessing_json={},
            source_dataset_id=saved["id"],
            source_thread_id="source-thread",
            source_attempt=1,
            status="archived",
            metadata_json={},
        )
        session.add(referenced)
        await session.commit()

    with pytest.raises(DatasetError) as exc:
        await service.delete_dataset("alice", saved["id"], confirmation=saved["id"])
    assert exc.value.code == "referenced_by_models"

    async with service.session_factory() as session:
        referenced = await session.get(NIRModelVersionRow, "nmv_reference")
        assert referenced is not None
        referenced.status = "deleted"
        await session.commit()

    deleted = await service.delete_dataset("alice", saved["id"], confirmation=saved["id"])
    assert deleted["status"] == "deleted"
    assert deleted["reclaimed_bytes"] == saved["size_bytes"]
    assert deleted["attached_thread_copies_retained"] is True
    assert not paths.user_nir_dataset_dir("alice", saved["id"]).exists()
    assert (await service.get_usage("alice"))["dataset_bytes"] == 0
    assert await service.list_datasets("alice") == []

    repeated = await service.delete_dataset("alice", saved["id"], confirmation=saved["id"])
    assert repeated["already_deleted"] is True
    assert repeated["reclaimed_bytes"] == 0

    resaved = await service.save_from_thread_upload("alice", "thread1", source, "A", save_confirmed=True)
    assert resaved["id"] != saved["id"]
    assert (await service.get_dataset("alice", resaved["id"]))["status"] == "ready"


@pytest.mark.asyncio
async def test_delete_dataset_recovers_from_interrupted_deleting_state(library) -> None:
    service, paths = library
    source = _upload(paths, "alice", "thread1", "spectrum.csv", b"x,y\n1,2\n")
    saved = await service.save_from_thread_upload("alice", "thread1", source, "A", save_confirmed=True)
    await service.archive_dataset("alice", saved["id"])
    asset_dir = paths.user_nir_dataset_dir("alice", saved["id"])
    unexpected = asset_dir / "unexpected.bin"
    unexpected.write_bytes(b"do not remove implicitly")

    with pytest.raises(DatasetError) as exc:
        await service.delete_dataset("alice", saved["id"], confirmation=saved["id"])
    assert exc.value.code == "deletion_failed"
    unexpected.unlink()

    deleted = await service.delete_dataset("alice", saved["id"], confirmation=saved["id"])
    assert deleted["status"] == "deleted"


@pytest.mark.asyncio
async def test_invalid_manifest_is_quarantined_without_leaking_parser_errors(library) -> None:
    service, paths = library
    source = _upload(paths, "alice", "thread1", "spectrum.csv", b"x,y\n1,2\n")
    saved = await service.save_from_thread_upload("alice", "thread1", source, "A", save_confirmed=True)
    manifest = paths.user_nir_dataset_dir("alice", saved["id"]) / "manifest.json"
    manifest.write_text("{not-json", encoding="utf-8")
    with pytest.raises(DatasetError) as exc:
        await service.get_dataset("alice", saved["id"])
    assert exc.value.code == "source_hash_mismatch"
    listed = await service.list_datasets("alice")
    assert listed[0]["status"] == "quarantined"


@pytest.mark.asyncio
async def test_concurrent_identical_saves_publish_only_one_asset(library) -> None:
    service, paths = library
    source = _upload(paths, "alice", "thread1", "spectrum.csv", b"x,y\n1,2\n")
    first, second = await asyncio.gather(
        service.save_from_thread_upload("alice", "thread1", source, "A", save_confirmed=True),
        service.save_from_thread_upload("alice", "thread1", source, "B", save_confirmed=True),
    )
    assert first["id"] == second["id"]
    assert len(await service.list_datasets("alice")) == 1


@pytest.mark.asyncio
async def test_copy_failure_leaves_no_ready_row_or_staging_file(library, monkeypatch: pytest.MonkeyPatch) -> None:
    service, paths = library
    source = _upload(paths, "alice", "thread1", "spectrum.csv", b"x,y\n1,2\n")

    def fail_copy(*_args):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(dataset_module, "_copy_upload", fail_copy)
    with pytest.raises(OSError):
        await service.save_from_thread_upload("alice", "thread1", source, "A", save_confirmed=True)
    assert await service.list_datasets("alice") == []
    assert list((paths.user_nir_datasets_dir("alice") / ".staging").iterdir()) == []


@pytest.mark.asyncio
async def test_symlinked_upload_is_rejected(library, tmp_path: Path) -> None:
    service, paths = library
    outside = tmp_path / "outside.csv"
    outside.write_bytes(b"private")
    linked = paths.sandbox_uploads_dir("thread1", user_id="alice") / "linked.csv"
    linked.parent.mkdir(parents=True, exist_ok=True)
    try:
        linked.symlink_to(outside)
    except OSError:
        pytest.skip("This Windows account cannot create symlinks")
    with pytest.raises(DatasetError) as exc:
        await service.save_from_thread_upload("alice", "thread1", "/mnt/user-data/uploads/linked.csv", "A", save_confirmed=True)
    assert exc.value.code == "invalid_source"


@pytest.mark.asyncio
async def test_profiles_are_versioned_and_need_resolved_mapping_to_confirm(library) -> None:
    service, paths = library
    source = _upload(paths, "alice", "thread1", "spectrum.csv", b"target,1000,1001\n1,0.1,0.2\n")
    saved = await service.save_from_thread_upload("alice", "thread1", source, "A", save_confirmed=True)
    draft = await service.create_profile(
        "alice",
        saved["id"],
        task_type="calibration",
        schema_status="needs_user_mapping",
        mapping={"candidate_target_columns": ["target"]},
    )
    assert draft["profile_version"] == 1
    assert draft["profile_status"] == "draft"
    with pytest.raises(DatasetError) as exc:
        await service.confirm_profile("alice", saved["id"], draft["id"])
    assert exc.value.code == "profile_unconfirmed"

    resolved = await service.create_profile(
        "alice",
        saved["id"],
        task_type="calibration",
        schema_status="confirmed_mapping",
        mapping={"target_column": "target", "spectral_columns": ["1000", "1001"]},
    )
    assert resolved["profile_version"] == 2
    confirmed = await service.confirm_profile("alice", saved["id"], resolved["id"])
    assert confirmed["profile_status"] == "confirmed"
    assert confirmed["mapping_sha256"]

    replacement = await service.create_profile(
        "alice",
        saved["id"],
        task_type="classification",
        schema_status="auto",
        mapping={"target_column": "target", "spectral_columns": ["1000", "1001"]},
    )
    await service.confirm_profile("alice", saved["id"], replacement["id"])
    profiles = await service.list_profiles("alice", saved["id"])
    assert [profile["profile_version"] for profile in profiles] == [3, 2, 1]
    assert next(profile for profile in profiles if profile["id"] == resolved["id"])["profile_status"] == "superseded"


@pytest.mark.asyncio
async def test_profile_cannot_cross_owner_or_store_raw_values(library) -> None:
    service, paths = library
    source = _upload(paths, "alice", "thread1", "spectrum.mat", b"MATLAB synthetic fixture")
    saved = await service.save_from_thread_upload("alice", "thread1", source, "A", save_confirmed=True)
    with pytest.raises(DatasetError) as exc:
        await service.create_profile("bob", saved["id"], task_type="calibration", schema_status="auto", mapping={})
    assert exc.value.code == "dataset_not_found"
    with pytest.raises(DatasetError) as exc:
        await service.create_profile(
            "alice",
            saved["id"],
            task_type="calibration",
            schema_status="auto",
            mapping={"raw_spectra": [[0.1, 0.2]]},
        )
    assert exc.value.code == "invalid_profile"


@pytest.mark.asyncio
async def test_staging_cleanup_is_ttl_scoped_and_ignores_unmarked_files(library, tmp_path: Path) -> None:
    service, paths = library
    service.policy.staging_ttl_hours = 1
    staging = paths.user_nir_datasets_dir("alice") / ".staging"
    staging.mkdir(parents=True)
    stale = staging / "nir-stage-abandoned.csv"
    unmarked = staging / "keep.csv"
    recent = staging / "nir-stage-recent.csv"
    for item in (stale, unmarked, recent):
        item.write_bytes(b"synthetic")
    os.utime(stale, (0, 0))
    os.utime(unmarked, (0, 0))

    result = await service.cleanup_stale_staging("alice", now=7200)
    assert result == {"stale_found": 1, "removed": 1}
    assert not stale.exists()
    assert unmarked.exists()
    assert recent.exists()

    outside = tmp_path / "outside.csv"
    outside.write_bytes(b"keep")
    linked = staging / "nir-stage-linked.csv"
    try:
        linked.symlink_to(outside)
    except OSError:
        return
    assert await service.cleanup_stale_staging("alice", now=7200) == {"stale_found": 0, "removed": 0}
    assert outside.read_bytes() == b"keep"


@pytest.mark.asyncio
async def test_database_commit_failure_compensates_published_files(library) -> None:
    service, paths = library
    source = _upload(paths, "alice", "thread1", "spectrum.csv", b"x,y\n1,2\n")
    real_factory = service.session_factory
    calls = 0

    @asynccontextmanager
    async def failing_session():
        nonlocal calls
        calls += 1
        async with real_factory() as session:
            if calls >= 2:
                session.commit = AsyncMock(side_effect=RuntimeError("simulated database failure"))
            yield session

    service.session_factory = failing_session  # type: ignore[assignment]
    with pytest.raises(RuntimeError, match="simulated database failure"):
        await service.save_from_thread_upload("alice", "thread1", source, "A", save_confirmed=True)
    assert await service.list_datasets("alice") == []
    assert not any(item.name.startswith("ds_") for item in paths.user_nir_datasets_dir("alice").iterdir())


def test_copy_detects_source_replacement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.csv"
    stage = tmp_path / "stage.csv"
    source.write_bytes(b"x,y\n1,2\n")
    real_lstat = Path.lstat
    calls = 0

    def changing_lstat(path: Path):
        nonlocal calls
        result = real_lstat(path)
        if path == source:
            calls += 1
            if calls == 2:
                return SimpleNamespace(
                    st_dev=result.st_dev,
                    st_ino=result.st_ino,
                    st_size=result.st_size,
                    st_mtime_ns=result.st_mtime_ns + 1,
                    st_mode=result.st_mode,
                )
        return result

    monkeypatch.setattr(Path, "lstat", changing_lstat)
    with pytest.raises(DatasetError) as exc:
        dataset_module._copy_upload(source, stage, 1024)
    assert exc.value.code == "source_changed"


@pytest.mark.asyncio
async def test_dataset_and_total_admission_limits_are_distinct(library) -> None:
    service, paths = library
    source = _upload(paths, "alice", "thread1", "spectrum.csv", b"x,y\n1,2\n")
    service.policy.max_user_dataset_bytes = 4
    with pytest.raises(DatasetError) as exc:
        await service.save_from_thread_upload("alice", "thread1", source, "A", save_confirmed=True)
    assert exc.value.code == "quota_exceeded"

    service.policy.max_user_dataset_bytes = 100
    service.policy.max_total_bytes_for_library_writes = 15
    with pytest.raises(DatasetError) as exc:
        await service.save_from_thread_upload("alice", "thread1", source, "A", save_confirmed=True)
    assert exc.value.code == "library_write_limit"


@pytest.mark.asyncio
async def test_minimum_free_space_is_checked_before_copy(library, monkeypatch: pytest.MonkeyPatch) -> None:
    service, paths = library
    source = _upload(paths, "alice", "thread1", "spectrum.csv", b"x,y\n1,2\n")
    service.policy.min_free_disk_bytes = 100
    monkeypatch.setattr(dataset_module.shutil, "disk_usage", lambda _path: SimpleNamespace(free=0))
    with pytest.raises(DatasetError) as exc:
        await service.save_from_thread_upload("alice", "thread1", source, "A", save_confirmed=True)
    assert exc.value.code == "insufficient_free_space"
    assert await service.list_datasets("alice") == []


@pytest.mark.asyncio
async def test_reconciliation_reports_missing_untracked_and_stale_without_paths(library) -> None:
    service, paths = library
    source = _upload(paths, "alice", "thread1", "spectrum.csv", b"x,y\n1,2\n")
    saved = await service.save_from_thread_upload("alice", "thread1", source, "A", save_confirmed=True)
    clean = await service.reconcile_storage("alice")
    assert clean["database_dataset_bytes"] == len(b"x,y\n1,2\n")
    assert clean["byte_discrepancy"] == 0
    assert clean["missing_assets"] == 0
    assert all("path" not in key for key in clean)

    (paths.user_nir_dataset_dir("alice", saved["id"]) / "source.csv").unlink()
    orphan = paths.user_nir_datasets_dir("alice") / "ds_untracked"
    orphan.mkdir()
    staging = paths.user_nir_datasets_dir("alice") / ".staging" / "nir-stage-old.csv"
    staging.write_bytes(b"orphan")
    os.utime(staging, (0, 0))
    report = await service.reconcile_storage("alice", now=service.policy.staging_ttl_hours * 3600 + 1)
    assert report["missing_assets"] == 1
    assert report["untracked_asset_directories"] == 1
    assert report["stale_staging_files"] == 1


@pytest.mark.asyncio
async def test_attach_requires_owned_thread_and_is_idempotent(library) -> None:
    service, paths = library
    source = _upload(paths, "alice", "source-thread", "spectrum.csv", b"target,1000\n1,0.2\n")
    saved = await service.save_from_thread_upload("alice", "source-thread", source, "A", save_confirmed=True)
    profile = await service.create_profile(
        "alice",
        saved["id"],
        task_type="calibration",
        schema_status="confirmed_mapping",
        mapping={"target_column": "target", "spectral_columns": ["1000"]},
    )
    profile = await service.confirm_profile("alice", saved["id"], profile["id"])
    await _owned_thread(service, "alice", "target-thread")

    first = await service.attach_to_thread(
        "alice",
        saved["id"],
        "target-thread",
        profile_id=profile["id"],
        idempotency_key="attach-once",
    )
    assert first["status"] == "attached"
    assert first["virtual_path"].startswith("/mnt/user-data/uploads/dataset-ds_")
    target = paths.sandbox_uploads_dir("target-thread", user_id="alice") / Path(first["virtual_path"]).name
    assert target.read_bytes() == b"target,1000\n1,0.2\n"

    repeated = await service.attach_to_thread(
        "alice",
        saved["id"],
        "target-thread",
        profile_id=profile["id"],
        idempotency_key="attach-once",
    )
    assert repeated["attachment_id"] == first["attachment_id"]
    assert repeated["reused_existing"] is True
    assert len(await service.list_uses("alice", saved["id"])) == 1

    with pytest.raises(DatasetError) as exc:
        await service.attach_to_thread("alice", saved["id"], "missing-thread", profile_id=None, idempotency_key="missing")
    assert exc.value.code == "thread_not_owned"


@pytest.mark.asyncio
async def test_attached_dataset_is_readable_by_fresh_nir_inspection(library) -> None:
    from deerflow.community.nir.io_tools import nir_inspect_tool

    service, paths = library
    content = b"target,1000,1002\n1.0,0.20,0.22\n1.1,0.21,0.23\n"
    source = _upload(paths, "alice", "source-thread", "spectrum.csv", content)
    saved = await service.save_from_thread_upload("alice", "source-thread", source, "A", save_confirmed=True)
    await _owned_thread(service, "alice", "target-thread")
    attached = await service.attach_to_thread(
        "alice",
        saved["id"],
        "target-thread",
        profile_id=None,
        idempotency_key="inspect-attached",
        run_id="run-inspect",
        prepare_workflow=True,
    )
    target = paths.sandbox_uploads_dir("target-thread", user_id="alice") / Path(attached["virtual_path"]).name

    with patch("deerflow.community.nir.io_tools._resolve", return_value=str(target)):
        payload = json.loads(
            nir_inspect_tool.func(
                runtime=MagicMock(context={"run_id": "run-inspect"}),
                file_path=attached["virtual_path"],
            )
        )

    assert payload["format"] == "csv"
    assert payload["shape"] == [2, 3]
    assert attached["run_id"] == "run-inspect"
    assert attached["workflow_project_id"].startswith("nir-")


@pytest.mark.asyncio
async def test_attach_rejects_unconfirmed_profile_and_compensates_sync_failure(library) -> None:
    service, paths = library
    source = _upload(paths, "alice", "source-thread", "spectrum.mat", b"synthetic mat bytes")
    saved = await service.save_from_thread_upload("alice", "source-thread", source, "A", save_confirmed=True)
    draft = await service.create_profile(
        "alice",
        saved["id"],
        task_type="calibration",
        schema_status="needs_user_mapping",
        mapping={"candidates": ["X1", "X2"]},
    )
    await _owned_thread(service, "alice", "target-thread")
    with pytest.raises(DatasetError) as exc:
        await service.attach_to_thread(
            "alice",
            saved["id"],
            "target-thread",
            profile_id=draft["id"],
            idempotency_key="draft",
        )
    assert exc.value.code == "profile_unconfirmed"

    async def fail_sync(_virtual_path: str, _local_path: Path) -> None:
        raise RuntimeError("remote sandbox unavailable")

    with pytest.raises(DatasetError) as exc:
        await service.attach_to_thread(
            "alice",
            saved["id"],
            "target-thread",
            profile_id=None,
            idempotency_key="sync-fails",
            desired_filename="attached.mat",
            sync_callback=fail_sync,
        )
    assert exc.value.code == "sandbox_sync_failed"
    assert not (paths.sandbox_uploads_dir("target-thread", user_id="alice") / "attached.mat").exists()
    async with service.session_factory() as session:
        row = await session.scalar(select(NIRDatasetUseRow).where(NIRDatasetUseRow.idempotency_key_sha256 == hashlib.sha256(b"sync-fails").hexdigest()))
        assert row is not None
        assert row.status == "failed"
