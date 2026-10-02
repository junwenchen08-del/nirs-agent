"""Owner-scoped API for promoted NIR model versions."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app.gateway.authz import get_auth_context, require_permission
from app.gateway.deps import get_checkpointer, get_config
from deerflow.community.nir.datasets.sandbox_sync import sync_attachment_to_sandbox as _sync_attachment_to_sandbox
from deerflow.community.nir.models.service import ModelError, ModelService
from deerflow.config.app_config import AppConfig
from deerflow.config.paths import get_paths
from deerflow.persistence.engine import get_session_factory

router = APIRouter(prefix="/api/nir/models", tags=["nir-models"])


class PromoteModelRequest(BaseModel):
    source_thread_id: str = Field(min_length=1, max_length=64)
    model_id: str = Field(min_length=1, max_length=128)
    version: str = Field(min_length=1, max_length=128)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=256)


class AttachModelRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=64)


class DeleteModelRequest(BaseModel):
    confirmation: str = Field(min_length=1, max_length=257)


def get_model_service(config: AppConfig = Depends(get_config)) -> ModelService:
    if not config.nir_library.enabled:
        raise HTTPException(status_code=404, detail={"code": "feature_disabled", "message": "NIR model library is disabled"})
    session_factory = get_session_factory()
    if session_factory is None:
        raise HTTPException(status_code=503, detail={"code": "database_unavailable", "message": "Persistent database is unavailable"})
    return ModelService(session_factory, get_paths(), config.nir_library)


def _owner(request: Request) -> str:
    auth = get_auth_context(request)
    if auth is None or not auth.is_authenticated:
        raise HTTPException(status_code=401, detail="Authentication required")
    return str(auth.require_user().id)


def _raise_model_error(exc: ModelError) -> None:
    status = {
        "feature_disabled": 404,
        "model_not_found": 404,
        "thread_not_owned": 404,
        "invalid_identifier": 400,
        "invalid_run_id": 400,
        "invalid_idempotency_key": 400,
        "confirmation_required": 400,
        "registration_required": 409,
        "scientific_gate_failed": 409,
        "lineage_mismatch": 409,
        "artifact_unverified": 409,
        "idempotency_conflict": 409,
        "storage_conflict": 409,
        "destination_conflict": 409,
        "version_limit": 409,
        "quota_exceeded": 413,
        "library_write_limit": 413,
        "insufficient_free_space": 507,
        "sandbox_sync_failed": 503,
        "storage_error": 500,
        "archive_required": 409,
        "deletion_failed": 409,
    }.get(exc.code, 500)
    raise HTTPException(status_code=status, detail={"code": exc.code, "message": str(exc)}) from exc


async def _registered_workflow(request: Request, thread_id: str) -> Mapping[str, object]:
    checkpointer = get_checkpointer(request)
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    try:
        checkpoint_tuple = await checkpointer.aget_tuple(config)
    except Exception:
        raise HTTPException(
            status_code=503,
            detail={"code": "checkpoint_unavailable", "message": "Could not verify the registered workflow"},
        ) from None
    checkpoint = getattr(checkpoint_tuple, "checkpoint", {}) if checkpoint_tuple is not None else {}
    channel_values = checkpoint.get("channel_values", {}) if isinstance(checkpoint, Mapping) else {}
    workflow = channel_values.get("nir_workflow") if isinstance(channel_values, Mapping) else None
    if not isinstance(workflow, Mapping) or workflow.get("stage") != "registered" or workflow.get("approval_status") != "approved":
        raise HTTPException(
            status_code=409,
            detail={"code": "registration_required", "message": "The source thread has no approved registered NIR model"},
        )
    return workflow


@router.get("")
@require_permission("threads", "read")
async def list_models(request: Request, include_archived: bool = False, service: ModelService = Depends(get_model_service)) -> dict:
    try:
        rows = await service.list_models(_owner(request), include_archived=include_archived)
    except ModelError as exc:
        _raise_model_error(exc)
    return {"models": rows, "count": len(rows)}


@router.post("/promote")
@require_permission("threads", "write")
async def promote_model(body: PromoteModelRequest, request: Request, service: ModelService = Depends(get_model_service)) -> dict:
    owner = _owner(request)
    workflow = await _registered_workflow(request, body.source_thread_id)
    evidence = workflow.get("attempt_evidence")
    run_id = str(evidence.get("run_id")) if isinstance(evidence, Mapping) and evidence.get("run_id") else None
    try:
        return await service.promote_registered(
            owner,
            body.source_thread_id,
            body.model_id,
            body.version,
            workflow=workflow,
            source_run_id=run_id,
            idempotency_key=body.idempotency_key,
        )
    except ModelError as exc:
        _raise_model_error(exc)


@router.get("/{model_id}")
@require_permission("threads", "read")
async def get_model(model_id: str, request: Request, version: str | None = None, service: ModelService = Depends(get_model_service)) -> dict:
    try:
        return await service.get_model(_owner(request), model_id, version)
    except ModelError as exc:
        _raise_model_error(exc)


@router.post("/{model_id}/versions/{version}/attach")
@require_permission("threads", "write")
async def attach_model(
    model_id: str,
    version: str,
    body: AttachModelRequest,
    request: Request,
    service: ModelService = Depends(get_model_service),
) -> dict:
    owner = _owner(request)

    async def sync(virtual_path: str, local_path: Path) -> None:
        await _sync_attachment_to_sandbox(owner, body.thread_id, virtual_path, local_path)

    try:
        return await service.attach_to_thread(
            owner,
            model_id,
            version,
            body.thread_id,
            sync_callback=sync,
        )
    except ModelError as exc:
        _raise_model_error(exc)


@router.post("/{model_id}/versions/{version}/archive")
@require_permission("threads", "write")
async def archive_model(model_id: str, version: str, request: Request, service: ModelService = Depends(get_model_service)) -> dict:
    try:
        return await service.archive_model(_owner(request), model_id, version)
    except ModelError as exc:
        _raise_model_error(exc)


@router.delete("/{model_id}/versions/{version}")
@require_permission("threads", "write")
async def delete_model(
    model_id: str,
    version: str,
    body: DeleteModelRequest,
    request: Request,
    service: ModelService = Depends(get_model_service),
) -> dict:
    try:
        return await service.delete_model(
            _owner(request),
            model_id,
            version,
            confirmation=body.confirmation,
        )
    except ModelError as exc:
        _raise_model_error(exc)
