"""Promote verified registered NIR models for owner-scoped reuse.

The service never accepts an arbitrary host path. Promotion is bound to the
current thread's approved workflow evidence and JSON registry record, and the
complete model bundle is verified before and after every copy.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import shutil
import stat
import tempfile
from collections.abc import Awaitable, Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from deerflow.community.nir._common import _read_regular_bytes, _verify_trusted_model_bundle
from deerflow.community.nir.datasets.service import _owner_admission_lock, _size_of_regular_files
from deerflow.community.nir.registration import _registration_gate, _validation_goal_gate
from deerflow.config.nir_library_config import NIRLibraryConfig
from deerflow.config.paths import Paths
from deerflow.persistence.nir_library.model import (
    NIRDatasetProfileRow,
    NIRDatasetRow,
    NIRModelVersionRow,
)
from deerflow.persistence.thread_meta.model import ThreadMetaRow

logger = logging.getLogger(__name__)
_COPY_CHUNK_SIZE = 1024 * 1024
_BUNDLE_FILES = ("model.pkl", "model.pkl.manifest.json", "metrics.json", "metadata.json")


class ModelError(Exception):
    """Stable model-library error that never reveals a host path."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _timestamp(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.isoformat()


def _public(row: NIRModelVersionRow, *, reused_existing: bool = False) -> dict:
    metadata = row.metadata_json if isinstance(row.metadata_json, dict) else {}
    return {
        "id": row.id,
        "model_id": row.model_id,
        "version": row.version,
        "status": row.status,
        "method": row.method,
        "preprocessing": row.preprocessing_json if isinstance(row.preprocessing_json, dict) else {},
        "validation_scope": row.validation_scope,
        "artifact_size_bytes": row.artifact_size_bytes,
        "artifact_sha256": row.artifact_sha256,
        "metrics_sha256": row.metrics_sha256,
        "training_data_sha256": row.training_data_sha256,
        "metrics_summary": metadata.get("metrics_summary", {}),
        "source_dataset_id": row.source_dataset_id,
        "source_profile_id": row.source_profile_id,
        "source_thread_id": row.source_thread_id,
        "source_run_id": row.source_run_id,
        "source_attempt": row.source_attempt,
        "created_at": _timestamp(row.created_at),
        "updated_at": _timestamp(row.updated_at),
        "last_used_at": _timestamp(row.last_used_at),
        "reused_existing": reused_existing,
    }


def _source_output_path(paths: Paths, owner: str, thread_id: str, virtual_path: object) -> Path:
    if not isinstance(virtual_path, str) or "\\" in virtual_path:
        raise ModelError("artifact_unverified", "Registered model paths must be under the source thread outputs directory")
    parsed = PurePosixPath(virtual_path)
    if parsed.as_posix() != virtual_path or parsed.parts[:4] != ("/", "mnt", "user-data", "outputs"):
        raise ModelError("artifact_unverified", "Registered model paths must be under the source thread outputs directory")
    try:
        return paths.resolve_virtual_path(thread_id, virtual_path, user_id=owner)
    except ValueError as exc:
        raise ModelError("artifact_unverified", "Registered model path is invalid") from exc


def _file_identity(info: os.stat_result) -> tuple[int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _copy_regular_file(source: Path, destination: Path) -> None:
    """Copy a regular file without following symlinks or accepting mutation."""

    try:
        before = source.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise ModelError("artifact_unverified", "Model package contains a non-regular file")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        with os.fdopen(os.open(source, flags), "rb") as input_file, destination.open("xb") as output_file:
            opened = os.fstat(input_file.fileno())
            if _file_identity(opened) != _file_identity(before):
                raise ModelError("artifact_unverified", "Model package changed during copy")
            while block := input_file.read(_COPY_CHUNK_SIZE):
                output_file.write(block)
            output_file.flush()
            os.fsync(output_file.fileno())
            after_open = os.fstat(input_file.fileno())
        after_path = source.lstat()
    except FileNotFoundError as exc:
        raise ModelError("artifact_unverified", "Registered model package is missing") from exc
    except ModelError:
        raise
    except OSError as exc:
        raise ModelError("storage_error", "Could not copy the verified model package") from exc
    if _file_identity(before) != _file_identity(after_open) or _file_identity(before) != _file_identity(after_path):
        raise ModelError("artifact_unverified", "Model package changed during copy")


def _write_metadata(path: Path, payload: dict) -> bytes:
    encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    with path.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    return encoded


def _remove_known_bundle(directory: Path, root: Path) -> None:
    """Compensate only a verified service-created package directory."""

    try:
        directory.resolve().relative_to(root.resolve())
    except ValueError:
        raise RuntimeError("Refusing to clean a model directory outside its owner root")
    if not directory.exists() or directory.is_symlink():
        return
    for filename in _BUNDLE_FILES:
        (directory / filename).unlink(missing_ok=True)
    directory.rmdir()


def _metrics_summary(metrics: Mapping[str, object]) -> dict:
    keys = ("R2_val", "RPD", "RMSEP", "grade", "passed", "n_targets", "component_names")
    return {key: metrics[key] for key in keys if key in metrics}


def _verify_complete_package(
    directory: Path,
    *,
    expected_metadata: Mapping[str, object],
    expected_package_size: int,
    expected_model_sha256: str,
    expected_metrics_sha256: str,
    expected_training_data_sha256: str,
    virtual_model_path: str | None = None,
) -> dict[str, object]:
    """Verify every persisted bundle file and its database binding."""

    verified = _verify_trusted_model_bundle(
        str(directory / "model.pkl"),
        str(directory / "metrics.json"),
        virtual_model_path=virtual_model_path,
        expected_model_sha256=expected_model_sha256,
        expected_metrics_sha256=expected_metrics_sha256,
        expected_training_data_sha256=expected_training_data_sha256,
    )
    metadata_bytes = _read_regular_bytes(
        directory / "metadata.json",
        label="Model library metadata",
        limit=256 * 1024,
    )
    try:
        metadata = json.loads(metadata_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid model library metadata.") from exc
    if not isinstance(metadata, dict) or metadata != dict(expected_metadata):
        raise ValueError("Model library metadata does not match its database record.")

    manifest_bytes = _read_regular_bytes(
        directory / "model.pkl.manifest.json",
        label="Model integrity manifest",
        limit=64 * 1024,
    )
    if hashlib.sha256(manifest_bytes).hexdigest() != verified["manifest_sha256"]:
        raise ValueError("Model integrity manifest changed during package verification.")
    metrics_bytes = _read_regular_bytes(
        directory / "metrics.json",
        label="Model metrics",
        limit=16 * 1024 * 1024,
    )
    if hashlib.sha256(metrics_bytes).hexdigest() != verified["metrics_sha256"]:
        raise ValueError("Model metrics changed during package verification.")
    package_size = int(verified["model_size_bytes"]) + len(manifest_bytes) + len(metrics_bytes) + len(metadata_bytes)
    if package_size != expected_package_size:
        raise ValueError("Model package size does not match its database record.")
    return verified


class ModelService:
    """Owner-scoped storage and reuse boundary for approved model packages."""

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        paths: Paths,
        policy: NIRLibraryConfig,
    ) -> None:
        self.session_factory = session_factory
        self.paths = paths
        self.policy = policy

    def _enabled(self) -> None:
        if not self.policy.enabled:
            raise ModelError("feature_disabled", "NIR model library is disabled")

    async def _owned_thread(self, session: AsyncSession, owner: str, thread_id: str) -> None:
        owned = await session.scalar(
            select(ThreadMetaRow.thread_id).where(
                ThreadMetaRow.thread_id == thread_id,
                ThreadMetaRow.user_id == owner,
            )
        )
        if owned is None:
            raise ModelError("thread_not_owned", "Thread not found")

    async def _used_model_bytes(self, session: AsyncSession, owner: str) -> int:
        result = await session.scalar(
            select(func.coalesce(func.sum(NIRModelVersionRow.artifact_size_bytes), 0)).where(
                NIRModelVersionRow.owner_user_id == owner,
                NIRModelVersionRow.status != "deleted",
            )
        )
        return int(result or 0)

    async def _used_dataset_bytes(self, session: AsyncSession, owner: str) -> int:
        result = await session.scalar(
            select(func.coalesce(func.sum(NIRDatasetRow.size_bytes), 0)).where(
                NIRDatasetRow.owner_user_id == owner,
                NIRDatasetRow.status != "deleted",
            )
        )
        return int(result or 0)

    async def _admit_model(self, session: AsyncSession, owner: str, additional: int) -> None:
        model_bytes = await self._used_model_bytes(session, owner)
        if model_bytes + additional > self.policy.max_user_model_bytes:
            raise ModelError("quota_exceeded", "Model library quota would be exceeded")
        dataset_bytes = await self._used_dataset_bytes(session, owner)
        user_root = self.paths.user_dir(owner)
        thread_bytes = await asyncio.to_thread(_size_of_regular_files, user_root / "threads")
        if dataset_bytes + model_bytes + thread_bytes + additional > self.policy.max_total_bytes_for_library_writes:
            raise ModelError("library_write_limit", "NIR library write-admission limit would be exceeded")
        free = await asyncio.to_thread(lambda: shutil.disk_usage(user_root).free)
        if free < self.policy.min_free_disk_bytes + additional:
            raise ModelError("insufficient_free_space", "Not enough free disk space for model-library write")

    async def _admit_attachment(self, session: AsyncSession, owner: str, additional: int) -> None:
        model_bytes = await self._used_model_bytes(session, owner)
        dataset_bytes = await self._used_dataset_bytes(session, owner)
        user_root = self.paths.user_dir(owner)
        thread_bytes = await asyncio.to_thread(_size_of_regular_files, user_root / "threads")
        if dataset_bytes + model_bytes + thread_bytes + additional > self.policy.max_total_bytes_for_library_writes:
            raise ModelError("library_write_limit", "NIR library write-admission limit would be exceeded")
        free = await asyncio.to_thread(lambda: shutil.disk_usage(user_root).free)
        if free < self.policy.min_free_disk_bytes + additional:
            raise ModelError("insufficient_free_space", "Not enough free disk space for model attachment")

    async def _validate_lineage(self, session: AsyncSession, owner: str, workflow: Mapping[str, object]) -> tuple[str | None, str | None]:
        dataset_id = str(workflow.get("dataset_id")) if workflow.get("dataset_id") else None
        profile_id = str(workflow.get("dataset_profile_id")) if workflow.get("dataset_profile_id") else None
        if profile_id is not None and dataset_id is None:
            raise ModelError("lineage_mismatch", "A source Profile requires its source Dataset")
        if dataset_id is not None:
            dataset = await session.scalar(
                select(NIRDatasetRow).where(
                    NIRDatasetRow.id == dataset_id,
                    NIRDatasetRow.owner_user_id == owner,
                    NIRDatasetRow.status != "deleted",
                )
            )
            if dataset is None:
                raise ModelError("lineage_mismatch", "Source Dataset lineage does not belong to this owner")
        if profile_id is not None:
            profile = await session.scalar(
                select(NIRDatasetProfileRow).where(
                    NIRDatasetProfileRow.id == profile_id,
                    NIRDatasetProfileRow.dataset_id == dataset_id,
                    NIRDatasetProfileRow.owner_user_id == owner,
                )
            )
            if profile is None:
                raise ModelError("lineage_mismatch", "Source Profile does not match the source Dataset and owner")
        return dataset_id, profile_id

    async def list_models(self, owner: str, *, include_archived: bool = False) -> list[dict]:
        self._enabled()
        async with self.session_factory() as session:
            query = select(NIRModelVersionRow).where(NIRModelVersionRow.owner_user_id == owner)
            if not include_archived:
                query = query.where(NIRModelVersionRow.status == "ready")
            else:
                query = query.where(NIRModelVersionRow.status != "deleted")
            rows = (await session.scalars(query.order_by(NIRModelVersionRow.created_at.desc()).limit(200))).all()
            return [_public(row) for row in rows]

    async def get_model(self, owner: str, model_id: str, version: str | None = None) -> dict:
        self._enabled()
        async with self.session_factory() as session:
            query = select(NIRModelVersionRow).where(
                NIRModelVersionRow.owner_user_id == owner,
                NIRModelVersionRow.model_id == model_id,
                NIRModelVersionRow.status != "deleted",
            )
            if version is not None:
                query = query.where(NIRModelVersionRow.version == version)
            row = await session.scalar(query.order_by(NIRModelVersionRow.created_at.desc()).limit(1))
            if row is None:
                raise ModelError("model_not_found", "Model version not found")
            return _public(row)

    async def promote_registered(
        self,
        owner: str,
        source_thread_id: str,
        model_id: str,
        version: str,
        *,
        workflow: Mapping[str, object],
        source_run_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> dict:
        """Promote exactly one workflow-approved and thread-registered bundle."""

        self._enabled()
        if workflow.get("stage") != "registered" or workflow.get("approval_status") != "approved":
            raise ModelError("registration_required", "Only an approved, registered workflow model can be promoted")
        evidence = workflow.get("attempt_evidence")
        if not isinstance(evidence, Mapping):
            raise ModelError("registration_required", "Registered workflow evidence is missing")
        async with self.session_factory() as session:
            await self._owned_thread(session, owner, source_thread_id)
        try:
            final_dir = self.paths.user_nir_model_version_dir(owner, model_id, version)
        except ValueError as exc:
            raise ModelError("invalid_identifier", "Model ID or version is invalid") from exc
        clean_run_id = source_run_id.strip() if isinstance(source_run_id, str) else None
        if clean_run_id == "":
            clean_run_id = None
        if clean_run_id is not None and len(clean_run_id) > 64:
            raise ModelError("invalid_run_id", "Run ID must be at most 64 characters")

        model_virtual = evidence.get("model_path")
        metrics_virtual = evidence.get("metrics_path")
        model_source = _source_output_path(self.paths, owner, source_thread_id, model_virtual)
        metrics_source = _source_output_path(self.paths, owner, source_thread_id, metrics_virtual)
        registry_source = self.paths.sandbox_outputs_dir(source_thread_id, user_id=owner) / "registry.json"
        try:
            registry_payload = json.loads(_read_regular_bytes(registry_source, label="Thread model registry", limit=4 * 1024 * 1024))
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ModelError("registration_required", "Thread model registry is missing or invalid") from exc
        versions = registry_payload.get(model_id) if isinstance(registry_payload, dict) else None
        record = next((item for item in versions if isinstance(item, dict) and item.get("version") == version), None) if isinstance(versions, list) else None
        if record is None or record.get("model_path") != model_virtual:
            raise ModelError("registration_required", "Requested model version is not registered in this source thread")

        try:
            verified = _verify_trusted_model_bundle(
                str(model_source),
                str(metrics_source),
                virtual_model_path=str(model_virtual),
                expected_model_sha256=str(evidence.get("model_sha256") or ""),
                expected_metrics_sha256=str(evidence.get("metrics_sha256") or ""),
                expected_training_data_sha256=str(evidence.get("training_data_sha256") or ""),
            )
        except ValueError as exc:
            raise ModelError("artifact_unverified", "Registered model package failed integrity verification") from exc
        if record.get("artifact_sha256") != verified["model_sha256"] or record.get("training_data_hash") != verified["training_data_sha256"]:
            raise ModelError("registration_required", "Thread registry does not match the approved model package")
        metrics = verified["metrics"]
        assert isinstance(metrics, dict)
        scientific_ok, _scientific_reason = _registration_gate(metrics)
        validation_ok, _validation_reason = _validation_goal_gate(metrics, dict(workflow))
        if not scientific_ok or not validation_ok:
            raise ModelError("scientific_gate_failed", "Registered model no longer satisfies its scientific and validation gates")
        validation_scope = str(metrics.get("validation_scope") or "")
        attempt = int(workflow.get("attempt") or 0)
        if attempt < 1:
            raise ModelError("lineage_mismatch", "Workflow attempt lineage is missing")
        preprocessing = {"steps": metrics.get("preprocessing_steps", [])}
        metadata = {
            "schema_version": 1,
            "model_id": model_id,
            "version": version,
            "artifact_sha256": verified["model_sha256"],
            "metrics_sha256": verified["metrics_sha256"],
            "training_data_sha256": verified["training_data_sha256"],
            "validation_scope": validation_scope,
            "method": metrics.get("method"),
            "preprocessing": preprocessing,
            "metrics_summary": _metrics_summary(metrics),
            "source": {
                "thread_id": source_thread_id,
                "run_id": clean_run_id,
                "attempt": attempt,
                "workflow_project_id": workflow.get("project_id"),
                "dataset_id": workflow.get("dataset_id"),
                "profile_id": workflow.get("dataset_profile_id"),
            },
        }
        metadata_bytes = (json.dumps(metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
        try:
            manifest_size = (Path(str(model_source) + ".manifest.json")).lstat().st_size
            metrics_size = metrics_source.lstat().st_size
        except OSError as exc:
            raise ModelError("artifact_unverified", "Registered model package changed during verification") from exc
        package_size = int(verified["model_size_bytes"]) + manifest_size + metrics_size + len(metadata_bytes)
        key_source = idempotency_key or f"{source_thread_id}:{model_id}:{version}:{verified['model_sha256']}"
        clean_key = key_source.strip() if isinstance(key_source, str) else ""
        if not clean_key or len(clean_key) > 256:
            raise ModelError("invalid_idempotency_key", "Idempotency key must be 1–256 characters")
        key_digest = hashlib.sha256(clean_key.encode("utf-8")).hexdigest()

        lock_root = self.paths.user_nir_datasets_dir(owner)
        model_root = self.paths.user_nir_models_dir(owner)
        async with _owner_admission_lock(lock_root):
            async with self.session_factory() as session:
                await self._owned_thread(session, owner, source_thread_id)
                existing_key = await session.scalar(
                    select(NIRModelVersionRow).where(
                        NIRModelVersionRow.owner_user_id == owner,
                        NIRModelVersionRow.promotion_key_sha256 == key_digest,
                    )
                )
                existing = await session.scalar(
                    select(NIRModelVersionRow).where(
                        NIRModelVersionRow.owner_user_id == owner,
                        NIRModelVersionRow.model_id == model_id,
                        NIRModelVersionRow.version == version,
                    )
                )
                candidate = existing_key or existing
                if candidate is not None:
                    if candidate.model_id != model_id or candidate.version != version or candidate.source_thread_id != source_thread_id or candidate.artifact_sha256 != verified["model_sha256"]:
                        raise ModelError("idempotency_conflict", "Promotion key or model version is already bound to another package")
                    await self._verify_stored(candidate)
                    return _public(candidate, reused_existing=True)
                version_count = await session.scalar(
                    select(func.count())
                    .select_from(NIRModelVersionRow)
                    .where(
                        NIRModelVersionRow.owner_user_id == owner,
                        NIRModelVersionRow.model_id == model_id,
                        NIRModelVersionRow.status == "ready",
                    )
                )
                if int(version_count or 0) >= self.policy.max_model_versions_per_id:
                    raise ModelError("version_limit", "Archive an older model version before promoting another")
                dataset_id, profile_id = await self._validate_lineage(session, owner, workflow)
                await self._admit_model(session, owner, package_size)

                stage_root = model_root / ".staging"
                await asyncio.to_thread(stage_root.mkdir, parents=True, exist_ok=True)
                stage_dir = Path(await asyncio.to_thread(tempfile.mkdtemp, prefix="nir-model-stage-", dir=stage_root))
                published = False
                try:
                    await asyncio.to_thread(_copy_regular_file, model_source, stage_dir / "model.pkl")
                    await asyncio.to_thread(
                        _copy_regular_file,
                        Path(str(model_source) + ".manifest.json"),
                        stage_dir / "model.pkl.manifest.json",
                    )
                    await asyncio.to_thread(_copy_regular_file, metrics_source, stage_dir / "metrics.json")
                    await asyncio.to_thread(_write_metadata, stage_dir / "metadata.json", metadata)
                    staged_verified = await asyncio.to_thread(
                        _verify_complete_package,
                        stage_dir,
                        expected_metadata=metadata,
                        expected_package_size=package_size,
                        expected_model_sha256=str(verified["model_sha256"]),
                        expected_metrics_sha256=str(verified["metrics_sha256"]),
                        expected_training_data_sha256=str(verified["training_data_sha256"]),
                    )
                    if staged_verified["manifest_sha256"] != verified["manifest_sha256"]:
                        raise ModelError("artifact_unverified", "Model manifest changed during promotion")
                    await asyncio.to_thread(final_dir.parent.mkdir, parents=True, exist_ok=True)
                    if final_dir.exists():
                        raise ModelError("storage_conflict", "Model version storage already exists without a database record")
                    await asyncio.to_thread(os.replace, stage_dir, final_dir)
                    published = True
                    row = NIRModelVersionRow(
                        id=f"nmv_{uuid4().hex}",
                        owner_user_id=owner,
                        model_id=model_id,
                        version=version,
                        artifact_relpath=PurePosixPath(model_id, version, "model.pkl").as_posix(),
                        artifact_size_bytes=package_size,
                        artifact_sha256=str(verified["model_sha256"]),
                        metrics_sha256=str(verified["metrics_sha256"]),
                        training_data_sha256=str(verified["training_data_sha256"]),
                        validation_scope=validation_scope,
                        method=str(metrics.get("method")) if metrics.get("method") else None,
                        preprocessing_json=preprocessing,
                        source_dataset_id=dataset_id,
                        source_profile_id=profile_id,
                        source_thread_id=source_thread_id,
                        source_run_id=clean_run_id,
                        source_attempt=attempt,
                        promotion_key_sha256=key_digest,
                        status="ready",
                        metadata_json=metadata,
                    )
                    session.add(row)
                    await session.commit()
                    return _public(row)
                except Exception:
                    await session.rollback()
                    try:
                        if published:
                            await asyncio.to_thread(_remove_known_bundle, final_dir, model_root)
                        elif stage_dir.exists():
                            await asyncio.to_thread(_remove_known_bundle, stage_dir, model_root)
                    except OSError:
                        logger.exception("Failed to compensate an incomplete NIR model promotion")
                    raise

    async def _verify_stored(self, row: NIRModelVersionRow) -> dict[str, object]:
        expected_relpath = PurePosixPath(row.model_id, row.version, "model.pkl").as_posix()
        if row.artifact_relpath != expected_relpath:
            raise ModelError("artifact_unverified", "Stored model path does not match its database identity")
        try:
            directory = self.paths.user_nir_model_version_dir(row.owner_user_id, row.model_id, row.version)
            return await asyncio.to_thread(
                _verify_complete_package,
                directory,
                expected_metadata=row.metadata_json if isinstance(row.metadata_json, dict) else {},
                expected_package_size=row.artifact_size_bytes,
                expected_model_sha256=row.artifact_sha256,
                expected_metrics_sha256=row.metrics_sha256,
                expected_training_data_sha256=row.training_data_sha256,
            )
        except (ValueError, OSError) as exc:
            raise ModelError("artifact_unverified", "Stored model package failed integrity verification") from exc

    async def attach_to_thread(
        self,
        owner: str,
        model_id: str,
        version: str,
        target_thread_id: str,
        *,
        sync_callback: Callable[[str, Path], Awaitable[None]] | None = None,
    ) -> dict:
        """Copy an owned verified model package into another owned thread."""

        self._enabled()
        lock_root = self.paths.user_nir_datasets_dir(owner)
        async with _owner_admission_lock(lock_root):
            async with self.session_factory() as session:
                await self._owned_thread(session, owner, target_thread_id)
                row = await session.scalar(
                    select(NIRModelVersionRow).where(
                        NIRModelVersionRow.owner_user_id == owner,
                        NIRModelVersionRow.model_id == model_id,
                        NIRModelVersionRow.version == version,
                        NIRModelVersionRow.status == "ready",
                    )
                )
                if row is None:
                    raise ModelError("model_not_found", "Ready model version not found")
                await self._verify_stored(row)
                source = self.paths.user_nir_model_version_dir(owner, model_id, version)
                output_root = self.paths.sandbox_outputs_dir(target_thread_id, user_id=owner)
                final_dir = output_root / "models" / model_id / version
                virtual_dir = f"/mnt/user-data/outputs/models/{model_id}/{version}"
                if final_dir.exists():
                    try:
                        existing = await asyncio.to_thread(
                            _verify_complete_package,
                            final_dir,
                            expected_metadata=row.metadata_json if isinstance(row.metadata_json, dict) else {},
                            expected_package_size=row.artifact_size_bytes,
                            virtual_model_path=f"{virtual_dir}/model.pkl",
                            expected_model_sha256=row.artifact_sha256,
                            expected_metrics_sha256=row.metrics_sha256,
                            expected_training_data_sha256=row.training_data_sha256,
                        )
                    except ValueError as exc:
                        raise ModelError("destination_conflict", "Target thread already contains a different or damaged model package") from exc
                    row.last_used_at = datetime.now(UTC)
                    row.updated_at = datetime.now(UTC)
                    await session.commit()
                    return self._attachment_result(row, reused_existing=True, model_size=int(existing["model_size_bytes"]))

                await self._admit_attachment(session, owner, row.artifact_size_bytes)
                await asyncio.to_thread(final_dir.parent.mkdir, parents=True, exist_ok=True)
                stage_dir = Path(await asyncio.to_thread(tempfile.mkdtemp, prefix="nir-attach-stage-", dir=final_dir.parent))
                published = False
                synced = sync_callback is None
                try:
                    for filename in _BUNDLE_FILES:
                        await asyncio.to_thread(_copy_regular_file, source / filename, stage_dir / filename)
                    await asyncio.to_thread(
                        _verify_complete_package,
                        stage_dir,
                        expected_metadata=row.metadata_json if isinstance(row.metadata_json, dict) else {},
                        expected_package_size=row.artifact_size_bytes,
                        expected_model_sha256=row.artifact_sha256,
                        expected_metrics_sha256=row.metrics_sha256,
                        expected_training_data_sha256=row.training_data_sha256,
                    )
                    await asyncio.to_thread(os.replace, stage_dir, final_dir)
                    published = True
                    await asyncio.to_thread(
                        _verify_complete_package,
                        final_dir,
                        expected_metadata=row.metadata_json if isinstance(row.metadata_json, dict) else {},
                        expected_package_size=row.artifact_size_bytes,
                        virtual_model_path=f"{virtual_dir}/model.pkl",
                        expected_model_sha256=row.artifact_sha256,
                        expected_metrics_sha256=row.metrics_sha256,
                        expected_training_data_sha256=row.training_data_sha256,
                    )
                    if sync_callback is not None:
                        for filename in _BUNDLE_FILES:
                            await sync_callback(f"{virtual_dir}/{filename}", final_dir / filename)
                        synced = True
                    row.last_used_at = datetime.now(UTC)
                    row.updated_at = datetime.now(UTC)
                    await session.commit()
                    return self._attachment_result(row, reused_existing=False, model_size=(final_dir / "model.pkl").stat().st_size)
                except Exception as exc:
                    await session.rollback()
                    try:
                        if published:
                            await asyncio.to_thread(_remove_known_bundle, final_dir, output_root)
                        elif stage_dir.exists():
                            await asyncio.to_thread(_remove_known_bundle, stage_dir, output_root)
                    except OSError:
                        logger.exception("Failed to compensate an incomplete NIR model attachment")
                    if isinstance(exc, ModelError):
                        raise
                    if sync_callback is not None and not synced:
                        raise ModelError("sandbox_sync_failed", "Could not synchronize the model into the target sandbox") from exc
                    raise ModelError("storage_error", "Could not attach the model package") from exc

    def _attachment_result(self, row: NIRModelVersionRow, *, reused_existing: bool, model_size: int) -> dict:
        virtual_dir = f"/mnt/user-data/outputs/models/{row.model_id}/{row.version}"
        return {
            "status": "attached",
            "model_id": row.model_id,
            "version": row.version,
            "model_path": f"{virtual_dir}/model.pkl",
            "metrics_path": f"{virtual_dir}/metrics.json",
            "manifest_path": f"{virtual_dir}/model.pkl.manifest.json",
            "artifact_sha256": row.artifact_sha256,
            "metrics_sha256": row.metrics_sha256,
            "training_data_sha256": row.training_data_sha256,
            "validation_scope": row.validation_scope,
            "model_size_bytes": model_size,
            "reused_existing": reused_existing,
        }

    async def archive_model(self, owner: str, model_id: str, version: str) -> dict:
        self._enabled()
        async with self.session_factory() as session:
            row = await session.scalar(
                select(NIRModelVersionRow).where(
                    NIRModelVersionRow.owner_user_id == owner,
                    NIRModelVersionRow.model_id == model_id,
                    NIRModelVersionRow.version == version,
                    NIRModelVersionRow.status.in_(["ready", "archived"]),
                )
            )
            if row is None:
                raise ModelError("model_not_found", "Model version not found")
            await self._verify_stored(row)
            row.status = "archived"
            row.updated_at = datetime.now(UTC)
            await session.commit()
            return _public(row)

    async def delete_model(self, owner: str, model_id: str, version: str, *, confirmation: str) -> dict:
        """Reclaim an archived model package while retaining its SQL lineage."""

        self._enabled()
        expected_confirmation = f"{model_id}:{version}"
        if confirmation != expected_confirmation:
            raise ModelError("confirmation_required", "Type the exact model ID and version to confirm permanent deletion")
        try:
            directory = self.paths.user_nir_model_version_dir(owner, model_id, version)
        except ValueError as exc:
            raise ModelError("model_not_found", "Model version not found") from exc
        model_root = self.paths.user_nir_models_dir(owner)
        lock_root = self.paths.user_nir_datasets_dir(owner)
        async with _owner_admission_lock(lock_root):
            async with self.session_factory() as session:
                row = await session.scalar(
                    select(NIRModelVersionRow).where(
                        NIRModelVersionRow.owner_user_id == owner,
                        NIRModelVersionRow.model_id == model_id,
                        NIRModelVersionRow.version == version,
                    )
                )
                if row is None:
                    raise ModelError("model_not_found", "Model version not found")
                if row.status == "deleted":
                    return {
                        **_public(row),
                        "already_deleted": True,
                        "reclaimed_bytes": 0,
                        "attached_thread_copies_retained": True,
                    }
                if row.status not in {"archived", "deleting"}:
                    raise ModelError("archive_required", "Archive the model version before permanent deletion")
                reclaimed_bytes = row.artifact_size_bytes
                row.status = "deleting"
                row.updated_at = datetime.now(UTC)
                await session.commit()
                try:
                    await asyncio.to_thread(_remove_known_bundle, directory, model_root)
                except OSError as exc:
                    raise ModelError("deletion_failed", "Model deletion is incomplete and can be retried") from exc
                row.status = "deleted"
                row.updated_at = datetime.now(UTC)
                await session.commit()
                return {
                    **_public(row),
                    "already_deleted": False,
                    "reclaimed_bytes": reclaimed_bytes,
                    "attached_thread_copies_retained": True,
                }
