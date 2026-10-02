"""Opt-in, owner-scoped NIR dataset-library API."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.gateway.authz import get_auth_context, require_permission
from app.gateway.deps import get_config, get_thread_store
from deerflow.community.nir.datasets.sandbox_sync import sync_attachment_to_sandbox as _sync_attachment_to_sandbox
from deerflow.community.nir.datasets.service import DatasetError, DatasetService
from deerflow.config.app_config import AppConfig
from deerflow.config.paths import get_paths
from deerflow.persistence.engine import get_session_factory

router = APIRouter(prefix="/api/nir/datasets", tags=["nir-datasets"])
storage_router = APIRouter(prefix="/api/nir/storage", tags=["nir-storage"])


class SaveDatasetRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=64)
    virtual_path: str = Field(min_length=1, max_length=1024)
    name: str = Field(min_length=1, max_length=256)
    save_confirmed: bool = False


class RenameDatasetRequest(BaseModel):
    name: str = Field(min_length=1, max_length=256)


class DeleteDatasetRequest(BaseModel):
    confirmation: str = Field(min_length=1, max_length=64)


class CreateProfileRequest(BaseModel):
    task_type: str = Field(min_length=1, max_length=64)
    schema_status: str = Field(min_length=1, max_length=32)
    mapping: dict = Field(default_factory=dict)


class AttachDatasetRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=64)
    profile_id: str | None = Field(default=None, min_length=1, max_length=64)
    idempotency_key: str = Field(min_length=1, max_length=256)
    desired_filename: str | None = Field(default=None, min_length=1, max_length=256)


def get_dataset_service(config: AppConfig = Depends(get_config)) -> DatasetService:
    if not config.nir_library.enabled:
        raise HTTPException(status_code=404, detail={"code": "feature_disabled", "message": "NIR dataset library is disabled"})
    session_factory = get_session_factory()
    if session_factory is None:
        raise HTTPException(status_code=503, detail={"code": "database_unavailable", "message": "Persistent database is unavailable"})
    return DatasetService(session_factory, get_paths(), config.nir_library)


def _owner(request: Request) -> str:
    auth = get_auth_context(request)
    if auth is None or not auth.is_authenticated:
        raise HTTPException(status_code=401, detail="Authentication required")
    return str(auth.require_user().id)


def _raise_dataset_error(exc: DatasetError) -> None:
    status = {
        "feature_disabled": 404,
        "source_not_found": 404,
        "dataset_not_found": 404,
        "invalid_source": 400,
        "invalid_name": 400,
        "confirmation_required": 400,
        "source_changed": 409,
        "dataset_archived": 409,
        "dataset_unavailable": 409,
        "source_hash_mismatch": 409,
        "profile_not_found": 404,
        "profile_unavailable": 409,
        "profile_unconfirmed": 409,
        "invalid_profile": 400,
        "invalid_idempotency_key": 400,
        "invalid_filename": 400,
        "thread_not_owned": 404,
        "idempotency_conflict": 409,
        "filename_conflict": 409,
        "sandbox_sync_failed": 503,
        "storage_error": 500,
        "file_too_large": 413,
        "quota_exceeded": 413,
        "library_write_limit": 413,
        "insufficient_free_space": 507,
        "archive_required": 409,
        "referenced_by_models": 409,
        "deletion_failed": 409,
    }.get(exc.code, 500)
    raise HTTPException(status_code=status, detail={"code": exc.code, "message": str(exc)}) from exc


@router.post("")
@require_permission("threads", "write")
async def save_dataset(body: SaveDatasetRequest, request: Request, service: DatasetService = Depends(get_dataset_service)) -> dict:
    owner = _owner(request)
    # Unlike legacy read routes, saving an uploaded source requires a real,
    # explicitly owned thread row. NULL-owner legacy rows cannot be imported.
    thread = await get_thread_store(request).get(body.thread_id, user_id=None)
    if thread is None or thread.get("user_id") != owner:
        raise HTTPException(status_code=404, detail={"code": "thread_not_found", "message": "Thread not found"})
    try:
        return await service.save_from_thread_upload(owner, body.thread_id, body.virtual_path, body.name, save_confirmed=body.save_confirmed)
    except DatasetError as exc:
        _raise_dataset_error(exc)


@router.get("")
@require_permission("threads", "read")
async def list_datasets(request: Request, service: DatasetService = Depends(get_dataset_service)) -> dict:
    try:
        rows = await service.list_datasets(_owner(request))
    except DatasetError as exc:
        _raise_dataset_error(exc)
    return {"datasets": rows, "count": len(rows)}


@storage_router.get("/usage")
@require_permission("threads", "read")
async def get_storage_usage(request: Request, service: DatasetService = Depends(get_dataset_service)) -> dict:
    try:
        return await service.get_usage(_owner(request))
    except DatasetError as exc:
        _raise_dataset_error(exc)


@storage_router.get("/reconciliation")
@require_permission("threads", "read")
async def reconcile_storage(request: Request, service: DatasetService = Depends(get_dataset_service)) -> dict:
    try:
        return await service.reconcile_storage(_owner(request))
    except DatasetError as exc:
        _raise_dataset_error(exc)


@router.get("/{dataset_id}")
@require_permission("threads", "read")
async def get_dataset(dataset_id: str, request: Request, service: DatasetService = Depends(get_dataset_service)) -> dict:
    try:
        owner = _owner(request)
        dataset = await service.get_dataset(owner, dataset_id)
        profiles = await service.list_profiles(owner, dataset_id)
        return {**dataset, "profiles": profiles}
    except DatasetError as exc:
        _raise_dataset_error(exc)


@router.post("/{dataset_id}/profiles")
@require_permission("threads", "write")
async def create_profile(
    dataset_id: str,
    body: CreateProfileRequest,
    request: Request,
    service: DatasetService = Depends(get_dataset_service),
) -> dict:
    try:
        return await service.create_profile(
            _owner(request),
            dataset_id,
            task_type=body.task_type,
            schema_status=body.schema_status,
            mapping=body.mapping,
        )
    except DatasetError as exc:
        _raise_dataset_error(exc)


@router.get("/{dataset_id}/profiles")
@require_permission("threads", "read")
async def list_profiles(dataset_id: str, request: Request, service: DatasetService = Depends(get_dataset_service)) -> dict:
    try:
        rows = await service.list_profiles(_owner(request), dataset_id)
    except DatasetError as exc:
        _raise_dataset_error(exc)
    return {"profiles": rows, "count": len(rows)}


@router.post("/{dataset_id}/profiles/{profile_id}/confirm")
@require_permission("threads", "write")
async def confirm_profile(
    dataset_id: str,
    profile_id: str,
    request: Request,
    service: DatasetService = Depends(get_dataset_service),
) -> dict:
    try:
        return await service.confirm_profile(_owner(request), dataset_id, profile_id)
    except DatasetError as exc:
        _raise_dataset_error(exc)


@router.post("/{dataset_id}/attach")
@require_permission("threads", "write")
async def attach_dataset(
    dataset_id: str,
    body: AttachDatasetRequest,
    request: Request,
    service: DatasetService = Depends(get_dataset_service),
) -> dict:
    owner = _owner(request)

    async def sync(virtual_path: str, local_path: Path) -> None:
        await _sync_attachment_to_sandbox(owner, body.thread_id, virtual_path, local_path)

    try:
        return await service.attach_to_thread(
            owner,
            dataset_id,
            body.thread_id,
            profile_id=body.profile_id,
            idempotency_key=body.idempotency_key,
            desired_filename=body.desired_filename,
            sync_callback=sync,
        )
    except DatasetError as exc:
        _raise_dataset_error(exc)


@router.get("/{dataset_id}/uses")
@require_permission("threads", "read")
async def list_dataset_uses(dataset_id: str, request: Request, service: DatasetService = Depends(get_dataset_service)) -> dict:
    try:
        rows = await service.list_uses(_owner(request), dataset_id)
    except DatasetError as exc:
        _raise_dataset_error(exc)
    return {"uses": rows, "count": len(rows)}


@router.patch("/{dataset_id}")
@require_permission("threads", "write")
async def rename_dataset(dataset_id: str, body: RenameDatasetRequest, request: Request, service: DatasetService = Depends(get_dataset_service)) -> dict:
    try:
        return await service.rename_dataset(_owner(request), dataset_id, body.name)
    except DatasetError as exc:
        _raise_dataset_error(exc)


@router.post("/{dataset_id}/archive")
@require_permission("threads", "write")
async def archive_dataset(dataset_id: str, request: Request, service: DatasetService = Depends(get_dataset_service)) -> dict:
    try:
        return await service.archive_dataset(_owner(request), dataset_id)
    except DatasetError as exc:
        _raise_dataset_error(exc)


@router.delete("/{dataset_id}")
@require_permission("threads", "write")
async def delete_dataset(
    dataset_id: str,
    body: DeleteDatasetRequest,
    request: Request,
    service: DatasetService = Depends(get_dataset_service),
) -> dict:
    try:
        return await service.delete_dataset(_owner(request), dataset_id, confirmation=body.confirmation)
    except DatasetError as exc:
        _raise_dataset_error(exc)
