"""Explicitly save immutable thread uploads in the NIR dataset library.

Only the gateway should call ``save_from_thread_upload`` after checking that
the authenticated owner owns the existing source thread. This service repeats
the filesystem boundary check and never accepts an arbitrary host path.
"""

from __future__ import annotations

import asyncio
import errno
import hashlib
import json
import logging
import os
import shutil
import stat
import tempfile
import threading
from collections.abc import Awaitable, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from uuid import uuid4

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from deerflow.config.nir_library_config import NIRLibraryConfig
from deerflow.config.paths import Paths
from deerflow.persistence.nir_library.model import NIRDatasetProfileRow, NIRDatasetRow, NIRDatasetUseRow, NIRModelVersionRow
from deerflow.persistence.thread_meta.model import ThreadMetaRow
from deerflow.uploads.manager import normalize_filename

_ALLOWED_EXTENSIONS = {".csv", ".txt", ".mat"}
_COPY_CHUNK_SIZE = 1024 * 1024
_MAX_MAPPING_JSON_BYTES = 64 * 1024
_PROFILE_SCHEMA_STATUSES = {"auto", "confirmed_mapping", "legacy_auto_layout", "needs_user_mapping", "spectra_only"}
_CONFIRMABLE_SCHEMA_STATUSES = {"auto", "confirmed_mapping", "legacy_auto_layout"}
_PROFILE_TASK_TYPES = {"calibration", "classification", "inspection"}
_STAGING_PREFIX = "nir-stage-"
_LOCKS_GUARD = threading.Lock()
_LOCKS: dict[str, threading.Lock] = {}
logger = logging.getLogger(__name__)


class DatasetError(Exception):
    """Stable, non-path-disclosing service error suitable for an HTTP adapter."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _thread_lock(path: Path) -> threading.Lock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(str(path.resolve()), threading.Lock())


@contextmanager
def _process_lock(path: Path) -> Iterator[None]:
    """An advisory cross-process lock on one owner's library admission path."""
    with path.open("a+b") as handle:
        if handle.seek(0, os.SEEK_END) == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:  # pragma: no cover - Linux deployment
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:  # pragma: no cover - Linux deployment
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


@asynccontextmanager
async def _owner_admission_lock(root: Path):
    await asyncio.to_thread(root.mkdir, parents=True, exist_ok=True)
    local_lock = await asyncio.to_thread(_thread_lock, root / ".admission")
    await asyncio.to_thread(local_lock.acquire)
    process_lock = _process_lock(root / ".admission")
    try:
        await asyncio.to_thread(process_lock.__enter__)
        try:
            yield
        finally:
            await asyncio.to_thread(process_lock.__exit__, None, None, None)
    finally:
        local_lock.release()


def _source_path(paths: Paths, owner: str, thread_id: str, virtual_path: str) -> Path:
    # Reject backslashes, alternate prefixes and nested directories before
    # resolving anything. The upload router stores files directly here.
    if not isinstance(virtual_path, str) or "\\" in virtual_path:
        raise DatasetError("invalid_source", "Choose a file from this thread's uploads")
    path = PurePosixPath(virtual_path)
    if path.as_posix() != virtual_path or len(path.parts) != 5 or path.parts[:4] != ("/", "mnt", "user-data", "uploads"):
        raise DatasetError("invalid_source", "Choose a file from this thread's uploads")
    filename = path.name
    if filename in {"", ".", ".."} or len(filename) > 256 or filename.lower().endswith(".md") or Path(filename).suffix.lower() not in _ALLOWED_EXTENSIONS:
        raise DatasetError("invalid_source", "Supported upload formats are CSV, TXT and MAT")
    try:
        return paths.sandbox_uploads_dir(thread_id, user_id=owner) / filename
    except ValueError as exc:
        raise DatasetError("invalid_source", "Invalid thread or owner") from exc


def _file_identity(info: os.stat_result) -> tuple[int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _copy_upload(source: Path, stage: Path, limit: int) -> tuple[str, int]:
    """Copy without following a source symlink; detect replacement or mutation."""
    try:
        before = source.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise DatasetError("invalid_source", "Upload is not a regular file")
        if before.st_size > limit:
            raise DatasetError("file_too_large", "Upload exceeds the dataset file limit")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(source, flags)
        with os.fdopen(fd, "rb") as input_file, stage.open("wb") as output_file:
            opened = os.fstat(input_file.fileno())
            if not stat.S_ISREG(opened.st_mode) or _file_identity(opened) != _file_identity(before):
                raise DatasetError("source_changed", "Upload changed while being saved")
            digest = hashlib.sha256()
            size = 0
            while chunk := input_file.read(_COPY_CHUNK_SIZE):
                size += len(chunk)
                if size > limit:
                    raise DatasetError("file_too_large", "Upload exceeds the dataset file limit")
                digest.update(chunk)
                output_file.write(chunk)
            output_file.flush()
            os.fsync(output_file.fileno())
            after_open = os.fstat(input_file.fileno())
        after_path = source.lstat()
        if _file_identity(before) != _file_identity(after_open) or _file_identity(before) != _file_identity(after_path) or size != before.st_size:
            raise DatasetError("source_changed", "Upload changed while being saved")
        return digest.hexdigest(), size
    except FileNotFoundError as exc:
        raise DatasetError("source_not_found", "Uploaded source file was not found") from exc
    except OSError as exc:
        raise DatasetError("storage_error", "Could not copy the uploaded file") from exc


def _size_of_regular_files(root: Path) -> int:
    if not root.exists():
        return 0
    total = 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [name for name in dirs if not (Path(directory) / name).is_symlink()]
        for name in files:
            item = Path(directory) / name
            info = item.lstat()
            if stat.S_ISREG(info.st_mode):
                total += info.st_size
    return total


def _public(row: NIRDatasetRow) -> dict:
    created_at = row.created_at
    if created_at is not None and created_at.tzinfo is None:
        # SQLite drops timezone info from DateTime(timezone=True) on reload.
        created_at = created_at.replace(tzinfo=UTC)
    return {
        "id": row.id,
        "name": row.name,
        "original_filename": row.original_filename,
        "sha256": row.sha256,
        "size_bytes": row.size_bytes,
        "media_type": row.media_type,
        "status": row.status,
        "created_at": created_at.isoformat() if created_at else None,
    }


def _public_profile(row: NIRDatasetProfileRow) -> dict:
    created_at = row.created_at
    confirmed_at = row.confirmed_at
    if created_at is not None and created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=UTC)
    if confirmed_at is not None and confirmed_at.tzinfo is None:
        confirmed_at = confirmed_at.replace(tzinfo=UTC)
    return {
        "id": row.id,
        "dataset_id": row.dataset_id,
        "profile_version": row.profile_version,
        "profile_status": row.profile_status,
        "task_type": row.task_type,
        "schema_status": row.schema_status,
        "mapping": row.mapping_json,
        "mapping_sha256": row.mapping_sha256,
        "confirmed_at": confirmed_at.isoformat() if confirmed_at else None,
        "created_at": created_at.isoformat() if created_at else None,
    }


def _validated_mapping(mapping: dict) -> tuple[dict, str]:
    if not isinstance(mapping, dict):
        raise DatasetError("invalid_profile", "Profile mapping must be a JSON object")
    try:
        canonical = json.dumps(mapping, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DatasetError("invalid_profile", "Profile mapping must contain finite JSON values") from exc
    if len(canonical) > _MAX_MAPPING_JSON_BYTES:
        raise DatasetError("invalid_profile", "Profile mapping is too large")
    # Profiles describe how to locate arrays/columns; raw spectra and row-level
    # target values belong only in the immutable source asset.
    forbidden = {"raw_spectra", "spectra_values", "target_values", "sample_rows", "matrix_values"}
    if forbidden.intersection(mapping):
        raise DatasetError("invalid_profile", "Profile mapping must not contain raw sample values")
    return json.loads(canonical), hashlib.sha256(canonical).hexdigest()


def _write_manifest(path: Path, *, asset_id: str, filename: str, digest: str, size: int) -> None:
    payload = {
        "schema_version": 1,
        "dataset_id": asset_id,
        "source_filename": filename,
        "sha256": digest,
        "size_bytes": size,
    }
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _create_stage(stage_root: Path, suffix: str) -> Path:
    stage_root.mkdir(exist_ok=True)
    stage_fd, stage_name = tempfile.mkstemp(prefix=_STAGING_PREFIX, suffix=suffix, dir=stage_root)
    os.close(stage_fd)
    return Path(stage_name)


def _publish_stage(stage: Path, final_dir: Path, *, filename: str, digest: str, size: int) -> None:
    """Publish source and manifest together, compensating any partial failure."""
    final_dir.mkdir()
    try:
        os.replace(stage, final_dir / filename)
        _write_manifest(final_dir / "manifest.json", asset_id=final_dir.name, filename=filename, digest=digest, size=size)
    except Exception:
        (final_dir / filename).unlink(missing_ok=True)
        (final_dir / "manifest.json").unlink(missing_ok=True)
        final_dir.rmdir()
        raise


def _remove_uncommitted(final_dir: Path, filename: str) -> None:
    (final_dir / filename).unlink(missing_ok=True)
    (final_dir / "manifest.json").unlink(missing_ok=True)
    final_dir.rmdir()


def _reconcile_storage(root: Path, rows: list[tuple[str, str | None, int]], cutoff: float) -> dict:
    """Compare owner database records with exact library directories, without mutation."""
    known_ids = {row_id for row_id, _relpath, _size in rows}
    missing_assets = 0
    source_bytes = 0
    untracked_asset_dirs = 0
    if root.exists():
        for item in root.iterdir():
            try:
                info = item.lstat()
            except FileNotFoundError:
                continue
            if item.name.startswith("ds_") and stat.S_ISDIR(info.st_mode) and item.name not in known_ids:
                untracked_asset_dirs += 1
    for row_id, relpath, _size in rows:
        relative = PurePosixPath(relpath) if relpath else None
        if relative is None or len(relative.parts) != 2 or relative.parts[0] != row_id:
            missing_assets += 1
            continue
        source = root / row_id / relative.name
        manifest = root / row_id / "manifest.json"
        try:
            source_info = source.lstat()
            manifest_info = manifest.lstat()
        except FileNotFoundError:
            missing_assets += 1
            continue
        if not stat.S_ISREG(source_info.st_mode) or not stat.S_ISREG(manifest_info.st_mode):
            missing_assets += 1
            continue
        source_bytes += source_info.st_size

    staging_bytes = 0
    stale_staging_files = 0
    stage_root = root / ".staging"
    if stage_root.exists():
        for item in stage_root.iterdir():
            try:
                info = item.lstat()
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(info.st_mode):
                continue
            staging_bytes += info.st_size
            if item.name.startswith(_STAGING_PREFIX) and info.st_mtime <= cutoff:
                stale_staging_files += 1
    database_bytes = sum(size for _row_id, _relpath, size in rows)
    return {
        "database_dataset_bytes": database_bytes,
        "asset_source_bytes": source_bytes,
        "staging_bytes": staging_bytes,
        "byte_discrepancy": source_bytes - database_bytes,
        "missing_assets": missing_assets,
        "untracked_asset_directories": untracked_asset_dirs,
        "stale_staging_files": stale_staging_files,
    }


def _clean_staging(stage_root: Path, cutoff: float) -> dict:
    if not stage_root.exists():
        return {"stale_found": 0, "removed": 0}
    stale_found = 0
    removed = 0
    for item in stage_root.iterdir():
        try:
            info = item.lstat()
        except FileNotFoundError:
            continue
        if not item.name.startswith(_STAGING_PREFIX) or not stat.S_ISREG(info.st_mode) or info.st_mtime > cutoff:
            continue
        stale_found += 1
        try:
            item.unlink()
            removed += 1
        except FileNotFoundError:
            pass
    return {"stale_found": stale_found, "removed": removed}


def _remove_dataset_asset(directory: Path, root: Path, source_filename: str) -> None:
    """Remove only the two files created for one verified dataset asset."""

    try:
        directory.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise OSError("Dataset asset is outside its owner library") from exc
    if not directory.exists():
        return
    if directory.is_symlink() or not directory.is_dir():
        raise OSError("Dataset asset directory is unsafe")
    allowed = {source_filename, "manifest.json"}
    for item in directory.iterdir():
        info = item.lstat()
        if item.name not in allowed or not stat.S_ISREG(info.st_mode):
            raise OSError("Dataset asset contains an unexpected entry")
    for filename in allowed:
        (directory / filename).unlink(missing_ok=True)
    directory.rmdir()


def _hash_regular_file(path: Path) -> str | None:
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            return None
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        with os.fdopen(os.open(path, flags), "rb") as handle:
            if _file_identity(os.fstat(handle.fileno())) != _file_identity(before):
                return None
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if _file_identity(path.lstat()) != _file_identity(before):
            return None
        return digest
    except OSError:
        return None


def _attach_local_file(source: Path, destination: Path, expected_sha256: str, size_limit: int) -> bool:
    """Atomically attach without overwriting an existing upload; return created."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.parent.chmod(0o777)
    except OSError:
        logger.debug("Could not widen NIR attachment directory permissions", exc_info=True)
    if destination.exists() or destination.is_symlink():
        if _hash_regular_file(destination) == expected_sha256:
            return False
        raise DatasetError("filename_conflict", "Attachment filename already exists with different content")
    fd, stage_name = tempfile.mkstemp(prefix=".nir-attach-", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    stage = Path(stage_name)
    try:
        digest, _size = _copy_upload(source, stage, size_limit)
        if digest != expected_sha256:
            raise DatasetError("source_hash_mismatch", "Dataset changed while being attached")
        try:
            os.link(stage, destination, follow_symlinks=False)
        except FileExistsError:
            if _hash_regular_file(destination) == expected_sha256:
                return False
            raise DatasetError("filename_conflict", "Attachment filename already exists with different content") from None
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                if _hash_regular_file(destination) == expected_sha256:
                    return False
                raise DatasetError("filename_conflict", "Attachment filename already exists with different content") from exc
            raise
        mode = stat.S_IMODE(destination.stat().st_mode) | stat.S_IRGRP | stat.S_IROTH
        destination.chmod(mode)
        return True
    finally:
        stage.unlink(missing_ok=True)


def _public_use(row: NIRDatasetUseRow, *, reused_existing: bool) -> dict:
    return {
        "attachment_id": row.id,
        "dataset_id": row.dataset_id,
        "profile_id": row.profile_id,
        "thread_id": row.thread_id,
        "run_id": row.run_id,
        "workflow_project_id": row.workflow_project_id,
        "source_sha256": row.source_sha256,
        "virtual_path": f"/mnt/user-data/uploads/{row.attachment_filename}" if row.attachment_filename else None,
        "status": row.status,
        "reused_existing": reused_existing,
    }


class DatasetService:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession], paths: Paths, policy: NIRLibraryConfig) -> None:
        self.session_factory = session_factory
        self.paths = paths
        self.policy = policy

    def _enabled(self) -> None:
        if not self.policy.enabled:
            raise DatasetError("feature_disabled", "NIR dataset library is disabled")

    async def _used_dataset_bytes(self, session: AsyncSession, owner: str) -> int:
        result = await session.scalar(
            select(func.coalesce(func.sum(NIRDatasetRow.size_bytes), 0)).where(
                NIRDatasetRow.owner_user_id == owner,
                NIRDatasetRow.status != "deleted",
            )
        )
        return int(result or 0)

    async def _admit(self, session: AsyncSession, owner: str, additional: int, *, staged: bool) -> None:
        current = await self._used_dataset_bytes(session, owner)
        if current + additional > self.policy.max_user_dataset_bytes:
            raise DatasetError("quota_exceeded", "Dataset library quota would be exceeded")
        user_root = self.paths.user_dir(owner)
        # This is a write-admission ceiling, not a hard limit on existing
        # uploads/sandbox writes. The source thread copy remains counted.
        thread_bytes = await asyncio.to_thread(_size_of_regular_files, user_root / "threads")
        model_bytes = await asyncio.to_thread(_size_of_regular_files, user_root / "nir-models")
        if current + model_bytes + thread_bytes + additional > self.policy.max_total_bytes_for_library_writes:
            raise DatasetError("library_write_limit", "NIR library write-admission limit would be exceeded")
        free = await asyncio.to_thread(lambda: shutil.disk_usage(user_root).free)
        required = self.policy.min_free_disk_bytes + (0 if staged else additional)
        if free < required:
            raise DatasetError("insufficient_free_space", "Not enough free disk space for library save")

    async def save_from_thread_upload(
        self,
        owner: str,
        thread_id: str,
        virtual_path: str,
        name: str,
        *,
        save_confirmed: bool,
    ) -> dict:
        self._enabled()
        if save_confirmed is not True:
            raise DatasetError("confirmation_required", "Explicit save confirmation is required")
        clean_name = name.strip() if isinstance(name, str) else ""
        if not clean_name or len(clean_name) > 256:
            raise DatasetError("invalid_name", "Dataset name must be 1–256 characters")
        async with self.session_factory() as session:
            source_thread = await session.scalar(
                select(ThreadMetaRow.thread_id).where(
                    ThreadMetaRow.thread_id == thread_id,
                    ThreadMetaRow.user_id == owner,
                )
            )
        if source_thread is None:
            raise DatasetError("thread_not_owned", "Source thread not found")
        source = _source_path(self.paths, owner, thread_id, virtual_path)
        root = self.paths.user_nir_datasets_dir(owner)
        async with _owner_admission_lock(root):
            stage_root = root / ".staging"
            cutoff = datetime.now(UTC).timestamp() - self.policy.staging_ttl_hours * 3600
            await asyncio.to_thread(_clean_staging, stage_root, cutoff)
            stage = await asyncio.to_thread(_create_stage, stage_root, source.suffix.lower())
            final_dir: Path | None = None
            filename: str | None = None
            final_created = False
            committed = False
            try:
                try:
                    source_info = await asyncio.to_thread(source.lstat)
                except FileNotFoundError as exc:
                    raise DatasetError("source_not_found", "Uploaded source file was not found") from exc
                if not stat.S_ISREG(source_info.st_mode):
                    raise DatasetError("invalid_source", "Upload is not a regular file")
                initial_size = source_info.st_size
                if initial_size > self.policy.max_dataset_file_size:
                    raise DatasetError("file_too_large", "Upload exceeds the dataset file limit")
                async with self.session_factory() as session:
                    await self._admit(session, owner, initial_size, staged=False)
                digest, size = await asyncio.to_thread(_copy_upload, source, stage, self.policy.max_dataset_file_size)
                async with self.session_factory() as session:
                    existing = await session.scalar(select(NIRDatasetRow).where(NIRDatasetRow.owner_user_id == owner, NIRDatasetRow.sha256 == digest, NIRDatasetRow.status != "deleted"))
                    if existing is not None:
                        if existing.status == "archived":
                            raise DatasetError("dataset_archived", "Identical dataset is archived; restore it explicitly")
                        if existing.status != "ready":
                            raise DatasetError("dataset_unavailable", "Identical dataset requires repair before reuse")
                        await self._verify_row(session, owner, existing)
                        return {**_public(existing), "reused_existing": True}
                    await self._admit(session, owner, size, staged=True)
                    asset_id = f"ds_{uuid4().hex}"
                    final_dir = self.paths.user_nir_dataset_dir(owner, asset_id)
                    filename = f"source{source.suffix.lower()}"
                    await asyncio.to_thread(_publish_stage, stage, final_dir, filename=filename, digest=digest, size=size)
                    final_created = True
                    row = NIRDatasetRow(
                        id=asset_id,
                        owner_user_id=owner,
                        name=clean_name,
                        original_filename=source.name,
                        sha256=digest,
                        size_bytes=size,
                        media_type={".csv": "text/csv", ".txt": "text/plain", ".mat": "application/octet-stream"}[source.suffix.lower()],
                        storage_relpath=f"{asset_id}/{filename}",
                        status="ready",
                        metadata_json={},
                    )
                    session.add(row)
                    try:
                        await session.commit()
                    except Exception:
                        await session.rollback()
                        raise
                    committed = True
                    return {**_public(row), "reused_existing": False}
            except Exception:
                if not committed and final_created and final_dir is not None and filename is not None:
                    try:
                        await asyncio.to_thread(_remove_uncommitted, final_dir, filename)
                    except OSError:
                        logger.exception("Failed to compensate an uncommitted NIR dataset publication")
                raise
            finally:
                try:
                    await asyncio.to_thread(stage.unlink, missing_ok=True)
                except OSError:
                    logger.exception("Failed to remove an NIR dataset staging file")

    async def _verify_row(self, session: AsyncSession, owner: str, row: NIRDatasetRow) -> None:
        if not row.storage_relpath:
            row.status = "quarantined"
            await session.commit()
            raise DatasetError("source_hash_mismatch", "Dataset asset is missing")
        relative = PurePosixPath(row.storage_relpath)
        if len(relative.parts) != 2 or relative.parts[0] != row.id:
            row.status = "quarantined"
            await session.commit()
            raise DatasetError("source_hash_mismatch", "Dataset asset path is invalid")
        target = self.paths.user_nir_dataset_dir(owner, row.id) / relative.name
        manifest_path = target.parent / "manifest.json"

        def verify() -> bool:
            try:
                manifest_info = manifest_path.lstat()
                if not stat.S_ISREG(manifest_info.st_mode) or manifest_info.st_size > 4096:
                    return False
                flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
                with os.fdopen(os.open(manifest_path, flags), "rb") as manifest_handle:
                    if _file_identity(os.fstat(manifest_handle.fileno())) != _file_identity(manifest_info):
                        return False
                    manifest = json.loads(manifest_handle.read().decode("utf-8"))
                if manifest != {
                    "schema_version": 1,
                    "dataset_id": row.id,
                    "source_filename": relative.name,
                    "sha256": row.sha256,
                    "size_bytes": row.size_bytes,
                }:
                    return False
                info = target.lstat()
                if not stat.S_ISREG(info.st_mode) or info.st_size != row.size_bytes:
                    return False
                with os.fdopen(os.open(target, flags), "rb") as handle:
                    if _file_identity(os.fstat(handle.fileno())) != _file_identity(info):
                        return False
                    digest = hashlib.file_digest(handle, "sha256").hexdigest()
                return digest == row.sha256 and _file_identity(target.lstat()) == _file_identity(info)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                return False

        if not await asyncio.to_thread(verify):
            row.status = "quarantined"
            await session.commit()
            raise DatasetError("source_hash_mismatch", "Dataset asset failed integrity verification")

    async def get_dataset(self, owner: str, dataset_id: str) -> dict:
        self._enabled()
        try:
            self.paths.user_nir_dataset_dir(owner, dataset_id)
        except ValueError as exc:
            raise DatasetError("dataset_not_found", "Dataset not found") from exc
        async with self.session_factory() as session:
            row = await session.scalar(select(NIRDatasetRow).where(NIRDatasetRow.id == dataset_id, NIRDatasetRow.owner_user_id == owner, NIRDatasetRow.status != "deleted"))
            if row is None:
                raise DatasetError("dataset_not_found", "Dataset not found")
            if row.status in {"ready", "archived"}:
                await self._verify_row(session, owner, row)
            return _public(row)

    async def list_datasets(self, owner: str) -> list[dict]:
        self._enabled()
        async with self.session_factory() as session:
            rows = (await session.scalars(select(NIRDatasetRow).where(NIRDatasetRow.owner_user_id == owner, NIRDatasetRow.status != "deleted").order_by(NIRDatasetRow.created_at.desc()).limit(100))).all()
            return [_public(row) for row in rows]

    async def get_usage(self, owner: str) -> dict:
        """Return owner-only capacity numbers; admission limit is not a hard quota."""
        self._enabled()
        root = self.paths.user_dir(owner)
        await asyncio.to_thread(root.mkdir, parents=True, exist_ok=True)
        async with self.session_factory() as session:
            dataset_bytes = await self._used_dataset_bytes(session, owner)
        thread_bytes, model_bytes, free = await asyncio.gather(
            asyncio.to_thread(_size_of_regular_files, root / "threads"),
            asyncio.to_thread(_size_of_regular_files, root / "nir-models"),
            asyncio.to_thread(lambda: shutil.disk_usage(root).free),
        )
        return {
            "dataset_bytes": dataset_bytes,
            "model_bytes": model_bytes,
            "thread_bytes": thread_bytes,
            "accounted_total_bytes": dataset_bytes + model_bytes + thread_bytes,
            "max_user_dataset_bytes": self.policy.max_user_dataset_bytes,
            "max_user_model_bytes": self.policy.max_user_model_bytes,
            "library_write_admission_bytes": self.policy.max_total_bytes_for_library_writes,
            "strict_total_quota": False,
            "disk_free_bytes": free,
            "min_free_disk_bytes": self.policy.min_free_disk_bytes,
        }

    async def cleanup_stale_staging(self, owner: str, *, now: float | None = None) -> dict:
        """Remove only old service-created regular staging files for one owner."""
        self._enabled()
        stage_root = self.paths.user_nir_datasets_dir(owner) / ".staging"
        cutoff = (datetime.now(UTC).timestamp() if now is None else now) - self.policy.staging_ttl_hours * 3600

        async with _owner_admission_lock(self.paths.user_nir_datasets_dir(owner)):
            return await asyncio.to_thread(_clean_staging, stage_root, cutoff)

    async def reconcile_storage(self, owner: str, *, now: float | None = None) -> dict:
        """Return owner-only DB/filesystem diagnostics without exposing host paths."""
        self._enabled()
        root = self.paths.user_nir_datasets_dir(owner)
        cutoff = (datetime.now(UTC).timestamp() if now is None else now) - self.policy.staging_ttl_hours * 3600
        async with self.session_factory() as session:
            records = (
                await session.execute(
                    select(NIRDatasetRow.id, NIRDatasetRow.storage_relpath, NIRDatasetRow.size_bytes).where(
                        NIRDatasetRow.owner_user_id == owner,
                        NIRDatasetRow.status != "deleted",
                    )
                )
            ).all()
        async with _owner_admission_lock(root):
            return await asyncio.to_thread(_reconcile_storage, root, list(records), cutoff)

    async def create_profile(
        self,
        owner: str,
        dataset_id: str,
        *,
        task_type: str,
        schema_status: str,
        mapping: dict,
    ) -> dict:
        """Create a new immutable mapping version in draft state."""
        self._enabled()
        if task_type not in _PROFILE_TASK_TYPES:
            raise DatasetError("invalid_profile", "Unsupported NIR profile task type")
        if schema_status not in _PROFILE_SCHEMA_STATUSES:
            raise DatasetError("invalid_profile", "Unsupported profile schema status")
        clean_mapping, mapping_digest = _validated_mapping(mapping)
        root = self.paths.user_nir_datasets_dir(owner)
        async with _owner_admission_lock(root):
            async with self.session_factory() as session:
                dataset = await session.scalar(
                    select(NIRDatasetRow).where(
                        NIRDatasetRow.id == dataset_id,
                        NIRDatasetRow.owner_user_id == owner,
                        NIRDatasetRow.status == "ready",
                    )
                )
                if dataset is None:
                    raise DatasetError("dataset_not_found", "Ready dataset not found")
                await self._verify_row(session, owner, dataset)
                latest = await session.scalar(
                    select(func.max(NIRDatasetProfileRow.profile_version)).where(
                        NIRDatasetProfileRow.dataset_id == dataset_id,
                        NIRDatasetProfileRow.owner_user_id == owner,
                    )
                )
                row = NIRDatasetProfileRow(
                    id=f"dsp_{uuid4().hex}",
                    owner_user_id=owner,
                    dataset_id=dataset_id,
                    profile_version=int(latest or 0) + 1,
                    profile_status="draft",
                    task_type=task_type,
                    schema_status=schema_status,
                    mapping_json=clean_mapping,
                    mapping_sha256=mapping_digest,
                )
                session.add(row)
                await session.commit()
                return _public_profile(row)

    async def list_profiles(self, owner: str, dataset_id: str) -> list[dict]:
        self._enabled()
        async with self.session_factory() as session:
            dataset = await session.scalar(
                select(NIRDatasetRow.id).where(
                    NIRDatasetRow.id == dataset_id,
                    NIRDatasetRow.owner_user_id == owner,
                    NIRDatasetRow.status != "deleted",
                )
            )
            if dataset is None:
                raise DatasetError("dataset_not_found", "Dataset not found")
            rows = (
                await session.scalars(
                    select(NIRDatasetProfileRow)
                    .where(
                        NIRDatasetProfileRow.dataset_id == dataset_id,
                        NIRDatasetProfileRow.owner_user_id == owner,
                        NIRDatasetProfileRow.profile_status != "archived",
                    )
                    .order_by(NIRDatasetProfileRow.profile_version.desc())
                    .limit(100)
                )
            ).all()
            return [_public_profile(row) for row in rows]

    async def confirm_profile(self, owner: str, dataset_id: str, profile_id: str) -> dict:
        """Confirm one draft and supersede the prior confirmed interpretation."""
        self._enabled()
        root = self.paths.user_nir_datasets_dir(owner)
        async with _owner_admission_lock(root):
            async with self.session_factory() as session:
                dataset = await session.scalar(
                    select(NIRDatasetRow).where(
                        NIRDatasetRow.id == dataset_id,
                        NIRDatasetRow.owner_user_id == owner,
                        NIRDatasetRow.status == "ready",
                    )
                )
                if dataset is None:
                    raise DatasetError("dataset_not_found", "Ready dataset not found")
                await self._verify_row(session, owner, dataset)
                profile = await session.scalar(
                    select(NIRDatasetProfileRow).where(
                        NIRDatasetProfileRow.id == profile_id,
                        NIRDatasetProfileRow.dataset_id == dataset_id,
                        NIRDatasetProfileRow.owner_user_id == owner,
                    )
                )
                if profile is None:
                    raise DatasetError("profile_not_found", "Profile not found")
                if profile.profile_status == "confirmed":
                    return _public_profile(profile)
                if profile.profile_status != "draft":
                    raise DatasetError("profile_unavailable", "Only a draft profile can be confirmed")
                _mapping, mapping_digest = _validated_mapping(profile.mapping_json)
                if mapping_digest != profile.mapping_sha256:
                    raise DatasetError("profile_unavailable", "Profile mapping failed integrity verification")
                if profile.schema_status not in _CONFIRMABLE_SCHEMA_STATUSES:
                    raise DatasetError("profile_unconfirmed", "Resolve the dataset mapping before confirming this profile")
                previous = (
                    await session.scalars(
                        select(NIRDatasetProfileRow).where(
                            NIRDatasetProfileRow.dataset_id == dataset_id,
                            NIRDatasetProfileRow.owner_user_id == owner,
                            NIRDatasetProfileRow.profile_status == "confirmed",
                        )
                    )
                ).all()
                for old in previous:
                    old.profile_status = "superseded"
                profile.profile_status = "confirmed"
                profile.confirmed_by = owner
                profile.confirmed_at = datetime.now(UTC)
                await session.commit()
                return _public_profile(profile)

    async def attach_to_thread(
        self,
        owner: str,
        dataset_id: str,
        thread_id: str,
        *,
        profile_id: str | None,
        idempotency_key: str,
        desired_filename: str | None = None,
        run_id: str | None = None,
        prepare_workflow: bool = False,
        sync_callback: Callable[[str, Path], Awaitable[None]] | None = None,
    ) -> dict:
        """Copy a verified asset into an owned thread and record one attachment."""
        self._enabled()
        clean_key = idempotency_key.strip() if isinstance(idempotency_key, str) else ""
        if not clean_key or len(clean_key) > 256:
            raise DatasetError("invalid_idempotency_key", "Idempotency key must be 1–256 characters")
        clean_run_id = run_id.strip() if isinstance(run_id, str) else None
        if clean_run_id == "":
            clean_run_id = None
        if clean_run_id is not None and len(clean_run_id) > 64:
            raise DatasetError("invalid_run_id", "Run ID must be at most 64 characters")
        key_digest = hashlib.sha256(clean_key.encode("utf-8")).hexdigest()
        root = self.paths.user_nir_datasets_dir(owner)
        async with _owner_admission_lock(root):
            async with self.session_factory() as session:
                thread = await session.scalar(
                    select(ThreadMetaRow).where(
                        ThreadMetaRow.thread_id == thread_id,
                        ThreadMetaRow.user_id == owner,
                    )
                )
                if thread is None:
                    raise DatasetError("thread_not_owned", "Target thread not found")
                dataset = await session.scalar(
                    select(NIRDatasetRow).where(
                        NIRDatasetRow.id == dataset_id,
                        NIRDatasetRow.owner_user_id == owner,
                        NIRDatasetRow.status == "ready",
                    )
                )
                if dataset is None or not dataset.sha256 or not dataset.storage_relpath:
                    raise DatasetError("dataset_not_found", "Ready dataset not found")
                await self._verify_row(session, owner, dataset)

                profile: NIRDatasetProfileRow | None = None
                if profile_id is not None:
                    profile = await session.scalar(
                        select(NIRDatasetProfileRow).where(
                            NIRDatasetProfileRow.id == profile_id,
                            NIRDatasetProfileRow.dataset_id == dataset_id,
                            NIRDatasetProfileRow.owner_user_id == owner,
                        )
                    )
                    if profile is None:
                        raise DatasetError("profile_not_found", "Profile not found")
                    if profile.profile_status != "confirmed":
                        raise DatasetError("profile_unconfirmed", "Only a confirmed Profile can be attached for reuse")
                    _mapping, profile_digest = _validated_mapping(profile.mapping_json)
                    if profile_digest != profile.mapping_sha256:
                        raise DatasetError("profile_unavailable", "Profile mapping failed integrity verification")

                relative = PurePosixPath(dataset.storage_relpath)
                source = self.paths.user_nir_dataset_dir(owner, dataset_id) / relative.name
                suffix = source.suffix.lower()
                if desired_filename is None:
                    filename = f"dataset-{dataset_id}{suffix}"
                else:
                    try:
                        filename = normalize_filename(desired_filename)
                    except ValueError as exc:
                        raise DatasetError("invalid_filename", "Attachment filename is invalid") from exc
                    if filename != desired_filename or Path(filename).suffix.lower() != suffix:
                        raise DatasetError("invalid_filename", "Attachment filename must be a basename with the original extension")

                existing = await session.scalar(
                    select(NIRDatasetUseRow).where(
                        NIRDatasetUseRow.owner_user_id == owner,
                        NIRDatasetUseRow.thread_id == thread_id,
                        NIRDatasetUseRow.idempotency_key_sha256 == key_digest,
                    )
                )
                if existing is not None:
                    if existing.dataset_id != dataset_id or existing.profile_id != profile_id or existing.attachment_filename != filename or existing.source_sha256 != dataset.sha256:
                        raise DatasetError("idempotency_conflict", "Idempotency key was already used for another attachment")
                    destination = self.paths.sandbox_uploads_dir(thread_id, user_id=owner) / filename
                    if existing.status == "attached" and await asyncio.to_thread(_hash_regular_file, destination) == dataset.sha256:
                        return _public_use(existing, reused_existing=True)
                    use = existing
                    use.status = "preparing"
                    if clean_run_id is not None:
                        use.run_id = clean_run_id
                    if prepare_workflow and use.workflow_project_id is None:
                        use.workflow_project_id = f"nir-{uuid4().hex[:12]}"
                    use.updated_at = datetime.now(UTC)
                else:
                    use = NIRDatasetUseRow(
                        id=f"use_{uuid4().hex}",
                        owner_user_id=owner,
                        dataset_id=dataset_id,
                        profile_id=profile_id,
                        thread_id=thread_id,
                        run_id=clean_run_id,
                        workflow_project_id=f"nir-{uuid4().hex[:12]}" if prepare_workflow else None,
                        use_kind="attach",
                        idempotency_key_sha256=key_digest,
                        attachment_filename=filename,
                        source_sha256=dataset.sha256,
                        status="preparing",
                    )
                    session.add(use)
                await session.commit()

                destination = self.paths.sandbox_uploads_dir(thread_id, user_id=owner) / filename
                created = False
                sync_completed = sync_callback is None
                try:
                    created = await asyncio.to_thread(_attach_local_file, source, destination, dataset.sha256, self.policy.max_dataset_file_size)
                    virtual_path = f"/mnt/user-data/uploads/{filename}"
                    if sync_callback is not None:
                        await sync_callback(virtual_path, destination)
                        sync_completed = True
                    use.status = "attached"
                    use.updated_at = datetime.now(UTC)
                    await session.commit()
                    return _public_use(use, reused_existing=not created)
                except Exception as exc:
                    if created:
                        try:
                            await asyncio.to_thread(destination.unlink, missing_ok=True)
                        except OSError:
                            logger.exception("Failed to compensate a failed NIR dataset attachment")
                    try:
                        await session.rollback()
                        failed_use = await session.get(NIRDatasetUseRow, use.id)
                        if failed_use is not None:
                            failed_use.status = "failed"
                            failed_use.updated_at = datetime.now(UTC)
                        await session.commit()
                    except Exception:
                        logger.exception("Failed to record a failed NIR dataset attachment")
                    if isinstance(exc, DatasetError):
                        raise
                    if sync_callback is not None and not sync_completed:
                        raise DatasetError("sandbox_sync_failed", "Could not synchronize the dataset into the target sandbox") from exc
                    raise DatasetError("storage_error", "Could not attach the dataset") from exc

    async def list_uses(self, owner: str, dataset_id: str) -> list[dict]:
        self._enabled()
        async with self.session_factory() as session:
            dataset = await session.scalar(
                select(NIRDatasetRow.id).where(
                    NIRDatasetRow.id == dataset_id,
                    NIRDatasetRow.owner_user_id == owner,
                    NIRDatasetRow.status != "deleted",
                )
            )
            if dataset is None:
                raise DatasetError("dataset_not_found", "Dataset not found")
            rows = (
                await session.scalars(
                    select(NIRDatasetUseRow)
                    .where(
                        NIRDatasetUseRow.owner_user_id == owner,
                        NIRDatasetUseRow.dataset_id == dataset_id,
                        NIRDatasetUseRow.status == "attached",
                    )
                    .order_by(NIRDatasetUseRow.created_at.desc())
                    .limit(100)
                )
            ).all()
            return [_public_use(row, reused_existing=True) for row in rows]

    async def rename_dataset(self, owner: str, dataset_id: str, name: str) -> dict:
        self._enabled()
        clean_name = name.strip() if isinstance(name, str) else ""
        if not clean_name or len(clean_name) > 256:
            raise DatasetError("invalid_name", "Dataset name must be 1–256 characters")
        async with self.session_factory() as session:
            row = await session.scalar(select(NIRDatasetRow).where(NIRDatasetRow.id == dataset_id, NIRDatasetRow.owner_user_id == owner, NIRDatasetRow.status.in_(["ready", "archived"])))
            if row is None:
                raise DatasetError("dataset_not_found", "Dataset not found")
            row.name = clean_name
            row.updated_at = datetime.now(UTC)
            await session.commit()
            return _public(row)

    async def archive_dataset(self, owner: str, dataset_id: str) -> dict:
        self._enabled()
        async with self.session_factory() as session:
            row = await session.scalar(select(NIRDatasetRow).where(NIRDatasetRow.id == dataset_id, NIRDatasetRow.owner_user_id == owner, NIRDatasetRow.status.in_(["ready", "archived"])))
            if row is None:
                raise DatasetError("dataset_not_found", "Dataset not found")
            await self._verify_row(session, owner, row)
            row.status = "archived"
            row.updated_at = datetime.now(UTC)
            await session.commit()
            return _public(row)

    async def delete_dataset(self, owner: str, dataset_id: str, *, confirmation: str) -> dict:
        """Reclaim an archived asset while retaining its SQL audit tombstone."""

        self._enabled()
        if confirmation != dataset_id:
            raise DatasetError("confirmation_required", "Type the exact dataset ID to confirm permanent deletion")
        try:
            directory = self.paths.user_nir_dataset_dir(owner, dataset_id)
        except ValueError as exc:
            raise DatasetError("dataset_not_found", "Dataset not found") from exc
        root = self.paths.user_nir_datasets_dir(owner)
        async with _owner_admission_lock(root):
            async with self.session_factory() as session:
                row = await session.scalar(
                    select(NIRDatasetRow).where(
                        NIRDatasetRow.id == dataset_id,
                        NIRDatasetRow.owner_user_id == owner,
                    )
                )
                if row is None:
                    raise DatasetError("dataset_not_found", "Dataset not found")
                if row.status == "deleted":
                    return {
                        **_public(row),
                        "already_deleted": True,
                        "reclaimed_bytes": 0,
                        "attached_thread_copies_retained": True,
                    }
                if row.status not in {"archived", "deleting"}:
                    raise DatasetError("archive_required", "Archive the dataset before permanent deletion")
                referenced = await session.scalar(
                    select(func.count())
                    .select_from(NIRModelVersionRow)
                    .where(
                        NIRModelVersionRow.owner_user_id == owner,
                        NIRModelVersionRow.source_dataset_id == dataset_id,
                        NIRModelVersionRow.status != "deleted",
                    )
                )
                if int(referenced or 0) > 0:
                    raise DatasetError("referenced_by_models", "Delete every referencing model version before this dataset")
                relative = PurePosixPath(row.storage_relpath or "")
                if len(relative.parts) != 2 or relative.parts[0] != dataset_id:
                    raise DatasetError("deletion_failed", "Dataset storage identity is invalid")
                reclaimed_bytes = row.size_bytes
                row.status = "deleting"
                row.updated_at = datetime.now(UTC)
                await session.commit()
                try:
                    await asyncio.to_thread(_remove_dataset_asset, directory, root, relative.name)
                except OSError as exc:
                    raise DatasetError("deletion_failed", "Dataset deletion is incomplete and can be retried") from exc
                row.status = "deleted"
                row.updated_at = datetime.now(UTC)
                await session.commit()
                return {
                    **_public(row),
                    "already_deleted": False,
                    "reclaimed_bytes": reclaimed_bytes,
                    "attached_thread_copies_retained": True,
                }
