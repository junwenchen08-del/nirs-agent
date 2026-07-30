"""Export checkpointed NIR workflows as deterministic evaluation traces."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from app.gateway.authz import get_auth_context, require_permission
from app.gateway.deps import get_checkpointer, get_current_user, get_run_event_store, get_run_manager, get_thread_store
from app.gateway.utils import sanitize_log_param
from deerflow.community.nir.evaluation import NIREvalTrace, evaluate_suite, load_scenarios, trace_from_workflow
from deerflow.utils.messages import message_content_to_text

logger = logging.getLogger(__name__)
router = APIRouter(tags=["nir-evaluations"])

_SCENARIOS_PATH = Path(__file__).resolve().parents[3] / "evals" / "nir" / "scenarios.json"


class NIREvaluationEntry(BaseModel):
    """One owned thread mapped to its expected regression scenario."""

    thread_id: str = Field(min_length=1, max_length=200)
    scenario_id: str = Field(min_length=1, max_length=200)


class NIRBatchEvaluationRequest(BaseModel):
    """Bounded batch request used by the evaluation dashboard."""

    entries: list[NIREvaluationEntry] = Field(min_length=1, max_length=100)


def _activated_nir_skill(event_groups: list[list[dict]]) -> str | None:
    for events in event_groups:
        for event in events:
            content = event.get("content")
            changes = content.get("changes") if isinstance(content, Mapping) else None
            if isinstance(changes, Mapping) and changes.get("skill_name") == "nir-coordinator":
                return "nir-coordinator"
    return None


def _duration_ms(records: list[Any]) -> float:
    total = 0.0
    for record in records:
        if not record.created_at or not record.updated_at:
            continue
        try:
            created = datetime.fromisoformat(record.created_at.replace("Z", "+00:00"))
            updated = datetime.fromisoformat(record.updated_at.replace("Z", "+00:00"))
            total += max(0.0, (updated - created).total_seconds() * 1000)
        except (TypeError, ValueError):
            logger.debug("Unable to parse NIR evaluation run timestamps for %s", record.run_id, exc_info=True)
    return total


def _latest_final_response(channel_values: Mapping[str, Any]) -> str | None:
    messages = channel_values.get("messages")
    if not isinstance(messages, list):
        return None
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            if message.tool_calls or message.additional_kwargs.get("hide_from_ui"):
                continue
            text = message_content_to_text(message.content).strip()
            if text:
                return text
            continue
        if isinstance(message, Mapping):
            role = str(message.get("role") or message.get("type") or "").lower()
            additional_kwargs = message.get("additional_kwargs")
            hidden = isinstance(additional_kwargs, Mapping) and additional_kwargs.get("hide_from_ui")
            if role not in {"assistant", "ai"} or message.get("tool_calls") or hidden:
                continue
            text = message_content_to_text(message.get("content")).strip()
            if text:
                return text
    return None


async def _capture_nir_evaluation_trace(
    thread_id: str,
    request: Request,
    scenario_id: str,
) -> NIREvalTrace:
    """Build one trace from the thread's authoritative checkpoint and run events."""
    checkpointer = get_checkpointer(request)
    config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    try:
        checkpoint_tuple = await checkpointer.aget_tuple(config)
    except Exception:
        logger.exception("Failed to read NIR evaluation checkpoint for thread %s", sanitize_log_param(thread_id))
        raise HTTPException(status_code=500, detail="Failed to read NIR evaluation trace") from None
    if checkpoint_tuple is None:
        raise HTTPException(status_code=404, detail=f"Thread {thread_id} not found")

    checkpoint = getattr(checkpoint_tuple, "checkpoint", {}) or {}
    channel_values = checkpoint.get("channel_values", {})
    workflow = channel_values.get("nir_workflow") if isinstance(channel_values, Mapping) else None
    if not isinstance(workflow, Mapping):
        raise HTTPException(status_code=404, detail="Thread has no NIR workflow trace")

    run_manager = get_run_manager(request)
    user_id = await get_current_user(request)
    all_records = await run_manager.list_by_thread(thread_id, user_id=user_id, limit=100)
    records_by_id = {record.run_id: record for record in all_records}
    requested_run_ids = [str(value) for value in (workflow.get("run_ids") or []) if value]
    records = [records_by_id[run_id] for run_id in requested_run_ids if run_id in records_by_id]
    if not records and all_records:
        records = all_records[:1]

    event_store = get_run_event_store(request)
    event_groups = await asyncio.gather(*(event_store.list_events(thread_id, record.run_id, event_types=["middleware:skill_activation"], limit=100) for record in records))
    trace = trace_from_workflow(
        scenario_id=scenario_id,
        workflow=workflow,
        routed_skill=_activated_nir_skill(event_groups),
        response_text=_latest_final_response(channel_values),
        duration_ms=_duration_ms(records),
        input_tokens=sum(record.total_input_tokens for record in records),
        output_tokens=sum(record.total_output_tokens for record in records),
    )
    return trace


@router.get("/api/threads/{thread_id}/nir-evaluation-trace")
@require_permission("threads", "read", owner_check=True)
async def export_nir_evaluation_trace(
    thread_id: str,
    request: Request,
    scenario_id: str = Query(..., min_length=1, max_length=200),
) -> dict[str, Any]:
    """Export the latest checkpointed NIR workflow in CLI-compatible JSON."""
    trace = await _capture_nir_evaluation_trace(thread_id, request, scenario_id)
    return {"version": 1, "traces": [trace.to_dict()]}


@router.get("/api/nir/evaluations/scenarios")
@require_permission("threads", "read")
async def list_nir_evaluation_scenarios(request: Request) -> dict[str, Any]:
    """Return the versioned scenario catalog used by both UI and CLI scoring."""
    del request
    scenarios = load_scenarios(_SCENARIOS_PATH)
    return {
        "version": 1,
        "scenarios": [asdict(scenario) for scenario in scenarios.values()],
    }


@router.post("/api/nir/evaluations/run")
@require_permission("threads", "read")
async def run_nir_batch_evaluation(
    body: NIRBatchEvaluationRequest,
    request: Request,
) -> dict[str, Any]:
    """Capture and deterministically score a bounded set of owned NIR threads."""
    scenarios = load_scenarios(_SCENARIOS_PATH)
    scenario_ids = [entry.scenario_id for entry in body.entries]
    if len(set(scenario_ids)) != len(scenario_ids):
        raise HTTPException(status_code=422, detail="Each NIR evaluation scenario may appear only once per batch")
    unknown = [scenario_id for scenario_id in scenario_ids if scenario_id not in scenarios]
    if unknown:
        raise HTTPException(status_code=422, detail=f"Unknown NIR evaluation scenarios: {', '.join(unknown)}")

    auth = get_auth_context(request)
    user_id = str(auth.user.id) if auth and auth.user else ""
    thread_store = get_thread_store(request)
    traces: list[NIREvalTrace] = []
    for entry in body.entries:
        if not await thread_store.check_access(entry.thread_id, user_id):
            raise HTTPException(status_code=404, detail=f"Thread {entry.thread_id} not found")
        traces.append(await _capture_nir_evaluation_trace(entry.thread_id, request, entry.scenario_id))

    summary = evaluate_suite(scenarios, traces)
    return {
        "version": 1,
        "summary": summary.to_dict(),
        "traces": [trace.to_dict() for trace in traces],
    }
