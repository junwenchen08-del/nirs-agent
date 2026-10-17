"""Owner-checked, read-only, bounded process evidence endpoints."""

from __future__ import annotations

import asyncio
import logging
import os

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response

from app.gateway.authz import require_permission
from app.gateway.deps import get_current_user
from deerflow.community.nir.process import ProcessStore
from deerflow.config.paths import get_paths

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/threads/{thread_id}/nir-process", tags=["nir-process"])


async def _store(request, thread_id):
    if os.getenv("NIR_PROCESS_ENABLED", "true").lower() == "false":
        raise HTTPException(404, "Process explorer disabled")
    user = await get_current_user(request)
    if not user:
        raise HTTPException(401, "Not authenticated")
    try:
        return ProcessStore(get_paths().thread_dir(thread_id, user_id=user) / "nir-process")
    except ValueError:
        raise HTTPException(404, "Process not found") from None


async def _read(function, *args):
    try:
        return await asyncio.to_thread(function, *args)
    except KeyError:
        raise HTTPException(404, "Process evidence not found") from None
    except ValueError as exc:
        raise HTTPException(409 if "version" in str(exc) else 404, "Process evidence version mismatch" if "version" in str(exc) else "Process evidence unavailable") from None
    except Exception:
        logger.exception("Failed to read process evidence")
        raise HTTPException(500, "Unable to read process evidence") from None


@router.get("/runs")
@require_permission("threads", "read", owner_check=True)
async def list_runs(thread_id: str, request: Request, offset: int = Query(0, ge=0, le=100000)):
    store = await _store(request, thread_id)
    return JSONResponse(await _read(store.runs, offset), headers={"Cache-Control": "private, no-store"})


@router.get("/runs/{run_id}")
@require_permission("threads", "read", owner_check=True)
async def read_run(thread_id: str, run_id: str, request: Request, offset: int = Query(0, ge=0, le=100000), attempt: str | None = Query(None, max_length=128)):
    store = await _store(request, thread_id)
    if not store.db.exists():
        raise HTTPException(404, "Process not recorded")
    payload = await _read(store.summary, run_id, offset, attempt)
    etag = f'"{run_id}-{payload["offset"]}-{payload["revision"]}"'
    headers = {"ETag": etag, "Cache-Control": "private, no-cache"}
    if request.headers.get("If-None-Match") == etag:
        return Response(status_code=304, headers=headers)
    return JSONResponse(payload, headers=headers)


@router.get("/runs/{run_id}/steps/{step_id}")
@require_permission("threads", "read", owner_check=True)
async def read_step(thread_id: str, run_id: str, step_id: str, request: Request):
    store = await _store(request, thread_id)
    return JSONResponse(await _read(store.step, run_id, step_id), headers={"Cache-Control": "private, no-store"})


@router.get("/runs/{run_id}/steps/{step_id}/candidates")
@require_permission("threads", "read", owner_check=True)
async def read_candidates(thread_id: str, run_id: str, step_id: str, request: Request, offset: int = Query(0, ge=0, le=100000)):
    store = await _store(request, thread_id)
    return JSONResponse(await _read(store.candidates, run_id, step_id, offset), headers={"Cache-Control": "private, no-store"})


@router.get("/runs/{run_id}/charts/{chart_id}")
@require_permission("threads", "read", owner_check=True)
async def read_chart(thread_id: str, run_id: str, chart_id: str, request: Request, version: str = Query(..., min_length=1, max_length=64)):
    store = await _store(request, thread_id)
    return JSONResponse(await _read(store.chart, run_id, chart_id, version), headers={"Cache-Control": "private, no-store"})


async def _autonomous(request, thread_id):
    from deerflow.community.nir.autonomous_view import AutonomousView

    return AutonomousView(await _store(request, thread_id))


@router.get("/autonomous")
@require_permission("threads", "read", owner_check=True)
async def autonomous_sessions(thread_id: str, request: Request, offset: int = Query(0, ge=0, le=100000)):
    view = await _autonomous(request, thread_id)
    return JSONResponse(await _read(view.sessions, offset), headers={"Cache-Control": "private, no-store"})


@router.get("/autonomous/{session_id}")
@require_permission("threads", "read", owner_check=True)
async def autonomous_summary(thread_id: str, session_id: str, request: Request):
    view = await _autonomous(request, thread_id)
    payload = await _read(view.summary, session_id)
    etag = f'"{session_id}-{payload["revision"]}"'
    headers = {"Cache-Control": "private, no-cache", "ETag": etag}
    if request.headers.get("If-None-Match") == etag:
        return Response(status_code=304, headers=headers)
    return JSONResponse(payload, headers=headers)


@router.get("/autonomous/{session_id}/decisions")
@require_permission("threads", "read", owner_check=True)
async def autonomous_decisions(thread_id: str, session_id: str, request: Request, offset: int = Query(0, ge=0, le=100000), limit: int = Query(10, ge=1, le=20)):
    view = await _autonomous(request, thread_id)
    return JSONResponse(await _read(view.decisions, session_id, offset, limit), headers={"Cache-Control": "private, no-store"})


@router.get("/autonomous/{session_id}/decisions/{decision_id}")
@require_permission("threads", "read", owner_check=True)
async def autonomous_decision(thread_id: str, session_id: str, decision_id: str, request: Request):
    view = await _autonomous(request, thread_id)
    return JSONResponse(await _read(view.decision, session_id, decision_id), headers={"Cache-Control": "private, no-store"})


@router.get("/autonomous/{session_id}/charts/{scope}")
@require_permission("threads", "read", owner_check=True)
async def autonomous_chart(thread_id: str, session_id: str, scope: str, request: Request):
    view = await _autonomous(request, thread_id)
    return JSONResponse(await _read(view.chart, session_id, scope), headers={"Cache-Control": "private, no-store"})


@router.get("/autonomous/{session_id}/rejections")
@require_permission("threads", "read", owner_check=True)
async def autonomous_rejections(thread_id: str, session_id: str, request: Request, offset: int = Query(0, ge=0, le=100000)):
    view = await _autonomous(request, thread_id)
    return JSONResponse(await _read(view.rejections, session_id, offset), headers={"Cache-Control": "private, no-store"})


@router.get("/autonomous/{session_id}/spectra/{experiment_id}")
@require_permission("threads", "read", owner_check=True)
async def autonomous_spectra(thread_id: str, session_id: str, experiment_id: str, request: Request):
    view = await _autonomous(request, thread_id)
    return JSONResponse(await _read(view.spectra, session_id, experiment_id), headers={"Cache-Control": "private, no-store"})


@router.get("/autonomous/{session_id}/diagnostics/{experiment_id}")
@require_permission("threads", "read", owner_check=True)
async def autonomous_round_diagnostics(thread_id: str, session_id: str, experiment_id: str, request: Request):
    view = await _autonomous(request, thread_id)
    return JSONResponse(await _read(view.diagnostics, session_id, experiment_id), headers={"Cache-Control": "private, no-store"})
