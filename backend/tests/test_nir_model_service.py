"""Persistent NIR model-library contracts using synthetic model packages."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import pytest_asyncio
from nir_core.utils.registry import ModelRegistry
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from deerflow.community.nir._common import (
    _bind_model_metrics,
    _load_trusted_model_artifact,
    _verify_trusted_model_bundle,
    _write_trusted_model_artifact,
)
from deerflow.community.nir.models import service as model_service_module
from deerflow.community.nir.models.service import ModelError, ModelService
from deerflow.config.nir_library_config import NIRLibraryConfig
from deerflow.config.paths import Paths
from deerflow.persistence.base import Base
from deerflow.persistence.thread_meta.model import ThreadMetaRow


@pytest_asyncio.fixture
async def model_library(tmp_path: Path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{(tmp_path / 'models.db').as_posix()}")
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    paths = Paths(tmp_path / "home")
    policy = NIRLibraryConfig(enabled=True, min_free_disk_bytes=0)
    service = ModelService(async_sessionmaker(engine, expire_on_commit=False), paths, policy)
    async with service.session_factory() as session:
        session.add_all(
            [
                ThreadMetaRow(thread_id="source-thread", user_id="alice", status="idle", metadata_json={}),
                ThreadMetaRow(thread_id="target-thread", user_id="alice", status="idle", metadata_json={}),
                ThreadMetaRow(thread_id="bob-thread", user_id="bob", status="idle", metadata_json={}),
            ]
        )
        await session.commit()
    try:
        yield service, paths
    finally:
        await engine.dispose()


def _registered_package(paths: Paths, *, owner: str = "alice", thread_id: str = "source-thread", model_id: str = "tablet_assay") -> tuple[str, dict]:
    output = paths.sandbox_outputs_dir(thread_id, user_id=owner)
    output.mkdir(parents=True, exist_ok=True)
    model_path = output / "candidate.pkl"
    metrics_path = output / "candidate.metrics.json"
    training_hash = "a" * 64
    metrics = {
        "method": "pls",
        "training_data_hash": training_hash,
        "validation_scope": "independent_holdout_not_external",
        "protocol": "deterministic_auto_split_holdout",
        "preprocessing_steps": [{"method": "snv", "params": {}}],
        "scientific_validation": {
            "passed": True,
            "dataset": {"passed": True},
            "partition_separation": {"passed": True},
        },
        "reproducibility": {
            "schema_version": 1,
            "input_sha256": training_hash,
            "random_state": 42,
            "protocol": "deterministic_auto_split_holdout",
        },
        "quality": {"passed": True},
        "R2_val": 0.91,
        "RMSEP": 0.32,
        "RPD": 3.1,
    }
    _write_trusted_model_artifact({"format": "nir_model_artifact", "model": {"kind": "synthetic"}}, str(model_path))
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    _bind_model_metrics(str(model_path), str(metrics_path), training_data_hash=training_hash)
    verified = _verify_trusted_model_bundle(str(model_path), str(metrics_path))
    registry = ModelRegistry(str(output / "registry.json"))
    version = registry.register(
        model_id=model_id,
        method="pls",
        metrics=metrics,
        preprocessing_steps=metrics["preprocessing_steps"],
        data_hash=training_hash,
        model_path="/mnt/user-data/outputs/candidate.pkl",
        artifact_hash=str(verified["model_sha256"]),
    )
    workflow = {
        "project_id": "nir-project",
        "stage": "registered",
        "task_type": "calibration",
        "approval_status": "approved",
        "validation_goal": "internal_holdout",
        "attempt": 1,
        "dataset_id": None,
        "dataset_profile_id": None,
        "attempt_evidence": {
            "schema_version": 1,
            "model_path": "/mnt/user-data/outputs/candidate.pkl",
            "metrics_path": "/mnt/user-data/outputs/candidate.metrics.json",
            "model_sha256": verified["model_sha256"],
            "metrics_sha256": verified["metrics_sha256"],
            "training_data_sha256": verified["training_data_sha256"],
            "validation_scope": metrics["validation_scope"],
            "protocol": metrics["protocol"],
        },
    }
    return version, workflow


@pytest.mark.asyncio
async def test_promote_is_verified_owner_scoped_and_idempotent(model_library) -> None:
    service, paths = model_library
    version, workflow = _registered_package(paths)

    first = await service.promote_registered(
        "alice",
        "source-thread",
        "tablet_assay",
        version,
        workflow=workflow,
        source_run_id="run-1",
        idempotency_key="promote-once",
    )
    second = await service.promote_registered(
        "alice",
        "source-thread",
        "tablet_assay",
        version,
        workflow=workflow,
        source_run_id="run-1",
        idempotency_key="promote-once",
    )

    assert first["status"] == "ready"
    assert first["validation_scope"] == "independent_holdout_not_external"
    assert first["source_dataset_id"] is None
    assert second == {**first, "reused_existing": True}
    assert len(await service.list_models("alice")) == 1
    assert await service.list_models("bob") == []
    with pytest.raises(ModelError) as exc:
        await service.get_model("bob", "tablet_assay", version)
    assert exc.value.code == "model_not_found"

    stored = paths.user_nir_model_version_dir("alice", "tablet_assay", version)
    _verify_trusted_model_bundle(
        str(stored / "model.pkl"),
        str(stored / "metrics.json"),
        expected_model_sha256=first["artifact_sha256"],
    )


@pytest.mark.asyncio
async def test_promote_rejects_unregistered_or_scientifically_invalid_state(model_library) -> None:
    service, paths = model_library
    version, workflow = _registered_package(paths)
    workflow["stage"] = "approved"
    with pytest.raises(ModelError) as exc:
        await service.promote_registered("alice", "source-thread", "tablet_assay", version, workflow=workflow)
    assert exc.value.code == "registration_required"

    workflow["stage"] = "registered"
    metrics_path = paths.sandbox_outputs_dir("source-thread", user_id="alice") / "candidate.metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["quality"]["passed"] = False
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ModelError) as exc:
        await service.promote_registered("alice", "source-thread", "tablet_assay", version, workflow=workflow)
    assert exc.value.code == "artifact_unverified"


@pytest.mark.asyncio
async def test_promote_rechecks_scientific_gate_after_valid_rebinding(model_library) -> None:
    service, paths = model_library
    version, workflow = _registered_package(paths)
    output = paths.sandbox_outputs_dir("source-thread", user_id="alice")
    metrics_path = output / "candidate.metrics.json"
    model_path = output / "candidate.pkl"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["quality"]["passed"] = False
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    _bind_model_metrics(str(model_path), str(metrics_path), training_data_hash=metrics["training_data_hash"])
    verified = _verify_trusted_model_bundle(str(model_path), str(metrics_path))
    workflow["attempt_evidence"].update(
        {
            "model_sha256": verified["model_sha256"],
            "metrics_sha256": verified["metrics_sha256"],
            "training_data_sha256": verified["training_data_sha256"],
        }
    )
    registry = json.loads((output / "registry.json").read_text(encoding="utf-8"))
    registry["tablet_assay"][0]["metrics"] = metrics
    (output / "registry.json").write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(ModelError) as exc:
        await service.promote_registered("alice", "source-thread", "tablet_assay", version, workflow=workflow)
    assert exc.value.code == "scientific_gate_failed"


@pytest.mark.asyncio
async def test_attach_copies_verified_bundle_for_prediction_after_source_thread_is_gone(model_library) -> None:
    service, paths = model_library
    version, workflow = _registered_package(paths)
    promoted = await service.promote_registered("alice", "source-thread", "tablet_assay", version, workflow=workflow)

    source_output = paths.sandbox_outputs_dir("source-thread", user_id="alice")
    for filename in ("candidate.pkl", "candidate.pkl.manifest.json", "candidate.metrics.json", "registry.json"):
        (source_output / filename).unlink()

    attached = await service.attach_to_thread("alice", "tablet_assay", version, "target-thread")
    assert attached["model_path"] == f"/mnt/user-data/outputs/models/tablet_assay/{version}/model.pkl"
    local_model = paths.resolve_virtual_path("target-thread", attached["model_path"], user_id="alice")
    loaded = _load_trusted_model_artifact(str(local_model), attached["model_path"])
    assert loaded["format"] == "nir_model_artifact"
    assert attached["artifact_sha256"] == promoted["artifact_sha256"]

    repeated = await service.attach_to_thread("alice", "tablet_assay", version, "target-thread")
    assert repeated["reused_existing"] is True
    with pytest.raises(ModelError) as exc:
        await service.attach_to_thread("alice", "tablet_assay", version, "bob-thread")
    assert exc.value.code == "thread_not_owned"


@pytest.mark.asyncio
async def test_attach_rejects_library_tampering(model_library) -> None:
    service, paths = model_library
    version, workflow = _registered_package(paths)
    await service.promote_registered("alice", "source-thread", "tablet_assay", version, workflow=workflow)
    stored = paths.user_nir_model_version_dir("alice", "tablet_assay", version)
    (stored / "metrics.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ModelError) as exc:
        await service.attach_to_thread("alice", "tablet_assay", version, "target-thread")
    assert exc.value.code == "artifact_unverified"


@pytest.mark.asyncio
async def test_attach_rejects_library_metadata_tampering(model_library) -> None:
    service, paths = model_library
    version, workflow = _registered_package(paths)
    await service.promote_registered("alice", "source-thread", "tablet_assay", version, workflow=workflow)
    stored = paths.user_nir_model_version_dir("alice", "tablet_assay", version)
    metadata_path = stored / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["model_id"] = "another_model"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ModelError) as exc:
        await service.attach_to_thread("alice", "tablet_assay", version, "target-thread")
    assert exc.value.code == "artifact_unverified"


@pytest.mark.asyncio
async def test_repeated_attach_rejects_target_metadata_tampering(model_library) -> None:
    service, paths = model_library
    version, workflow = _registered_package(paths)
    await service.promote_registered("alice", "source-thread", "tablet_assay", version, workflow=workflow)
    await service.attach_to_thread("alice", "tablet_assay", version, "target-thread")
    attached = paths.sandbox_outputs_dir("target-thread", user_id="alice") / "models" / "tablet_assay" / version
    (attached / "metadata.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ModelError) as exc:
        await service.attach_to_thread("alice", "tablet_assay", version, "target-thread")
    assert exc.value.code == "destination_conflict"


@pytest.mark.asyncio
async def test_promotion_checks_source_thread_ownership_before_artifacts(model_library) -> None:
    service, paths = model_library
    version, workflow = _registered_package(paths)

    with pytest.raises(ModelError) as exc:
        await service.promote_registered("bob", "source-thread", "tablet_assay", version, workflow=workflow)
    assert exc.value.code == "thread_not_owned"


@pytest.mark.asyncio
async def test_concurrent_promotion_publishes_one_version(model_library) -> None:
    service, paths = model_library
    version, workflow = _registered_package(paths)
    first, second = await asyncio.gather(
        service.promote_registered("alice", "source-thread", "tablet_assay", version, workflow=workflow),
        service.promote_registered("alice", "source-thread", "tablet_assay", version, workflow=workflow),
    )
    assert first["id"] == second["id"]
    assert {first["reused_existing"], second["reused_existing"]} == {False, True}


@pytest.mark.asyncio
async def test_promotion_copy_failure_leaves_no_ready_row_or_package(model_library, monkeypatch: pytest.MonkeyPatch) -> None:
    service, paths = model_library
    version, workflow = _registered_package(paths)

    def fail_copy(*_args) -> None:
        raise ModelError("storage_error", "simulated copy failure")

    monkeypatch.setattr(model_service_module, "_copy_regular_file", fail_copy)
    with pytest.raises(ModelError) as exc:
        await service.promote_registered("alice", "source-thread", "tablet_assay", version, workflow=workflow)
    assert exc.value.code == "storage_error"
    assert await service.list_models("alice") == []
    assert not paths.user_nir_model_version_dir("alice", "tablet_assay", version).exists()


@pytest.mark.asyncio
async def test_model_quota_version_cap_and_archive_are_enforced(model_library) -> None:
    service, paths = model_library
    version, workflow = _registered_package(paths)
    service.policy.max_user_model_bytes = 1
    with pytest.raises(ModelError) as exc:
        await service.promote_registered("alice", "source-thread", "tablet_assay", version, workflow=workflow)
    assert exc.value.code == "quota_exceeded"

    service.policy.max_user_model_bytes = 1024 * 1024
    service.policy.max_model_versions_per_id = 1
    await service.promote_registered("alice", "source-thread", "tablet_assay", version, workflow=workflow)
    second_version, second_workflow = _registered_package(paths)
    with pytest.raises(ModelError) as exc:
        await service.promote_registered(
            "alice",
            "source-thread",
            "tablet_assay",
            second_version,
            workflow=second_workflow,
        )
    assert exc.value.code == "version_limit"
    archived = await service.archive_model("alice", "tablet_assay", version)
    assert archived["status"] == "archived"


@pytest.mark.asyncio
async def test_delete_model_requires_archive_and_exact_confirmation(model_library) -> None:
    service, paths = model_library
    version, workflow = _registered_package(paths)
    promoted = await service.promote_registered("alice", "source-thread", "tablet_assay", version, workflow=workflow)

    with pytest.raises(ModelError) as exc:
        await service.delete_model("alice", "tablet_assay", version, confirmation=f"tablet_assay:{version}")
    assert exc.value.code == "archive_required"

    await service.archive_model("alice", "tablet_assay", version)
    with pytest.raises(ModelError) as exc:
        await service.delete_model("alice", "tablet_assay", version, confirmation="wrong")
    assert exc.value.code == "confirmation_required"

    deleted = await service.delete_model("alice", "tablet_assay", version, confirmation=f"tablet_assay:{version}")
    assert deleted["status"] == "deleted"
    assert deleted["reclaimed_bytes"] == promoted["artifact_size_bytes"]
    assert deleted["attached_thread_copies_retained"] is True
    assert not paths.user_nir_model_version_dir("alice", "tablet_assay", version).exists()
    assert await service.list_models("alice", include_archived=True) == []

    repeated = await service.delete_model("alice", "tablet_assay", version, confirmation=f"tablet_assay:{version}")
    assert repeated["already_deleted"] is True
    assert repeated["reclaimed_bytes"] == 0


@pytest.mark.asyncio
async def test_delete_model_recovers_from_interrupted_deleting_state(model_library) -> None:
    service, paths = model_library
    version, workflow = _registered_package(paths)
    await service.promote_registered("alice", "source-thread", "tablet_assay", version, workflow=workflow)
    await service.archive_model("alice", "tablet_assay", version)
    model_dir = paths.user_nir_model_version_dir("alice", "tablet_assay", version)
    unexpected = model_dir / "unexpected.bin"
    unexpected.write_bytes(b"do not remove implicitly")

    with pytest.raises(ModelError) as exc:
        await service.delete_model("alice", "tablet_assay", version, confirmation=f"tablet_assay:{version}")
    assert exc.value.code == "deletion_failed"
    unexpected.unlink()

    deleted = await service.delete_model("alice", "tablet_assay", version, confirmation=f"tablet_assay:{version}")
    assert deleted["status"] == "deleted"


@pytest.mark.asyncio
async def test_promotion_rejects_missing_source_storage(model_library) -> None:
    service, paths = model_library
    version, workflow = _registered_package(paths)
    (paths.sandbox_outputs_dir("source-thread", user_id="alice") / "candidate.pkl").unlink()
    with pytest.raises(ModelError) as exc:
        await service.promote_registered("alice", "source-thread", "tablet_assay", version, workflow=workflow)
    assert exc.value.code == "artifact_unverified"
