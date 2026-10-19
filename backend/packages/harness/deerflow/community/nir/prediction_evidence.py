"""Bounded prediction receipts, recorded only from actual tool execution."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def _path(value: Any) -> str:
    return str(value or "").replace("\\", "/").rstrip("/")


def prediction_succeeded(workflow: Mapping[str, Any]) -> bool:
    receipt = workflow.get("prediction_evidence")
    return bool(
        workflow.get("task_type") == "prediction"
        and isinstance(receipt, Mapping)
        and receipt.get("status") == "success"
        and receipt.get("model_path") == _path(workflow.get("model_path"))
        and receipt.get("source_data_path") == _path(workflow.get("data_path"))
    )


def prediction_result_evidence(workflow: Mapping[str, Any], args: Mapping[str, Any], payload: Mapping[str, Any], *, call_id: str | None, failed: bool = False) -> dict:
    """Project a tool result; never accept a model mount or workflow note as proof.

    Input NPZ may differ from the uploaded MAT after loading. Keep both paths;
    no matrices, per-sample predictions, or unbounded drift indices are retained.
    A failed latest call replaces earlier success, including audit-write failures.
    """
    evidence = {
        "schema_version": 1,
        "status": "error" if failed or payload.get("status") == "error" else "unavailable",
        "tool_call_id": call_id,
        "source_data_path": _path(workflow.get("data_path")),
        "model_path": _path(args.get("model_path")),
        "data_path": _path(args.get("data_path")),
    }
    audit = payload.get("audit")
    count = payload.get("n_samples")
    if failed or payload.get("status") != "ok" or not isinstance(count, int) or isinstance(count, bool) or count < 1:
        return evidence
    if evidence["model_path"] != _path(workflow.get("model_path")) or not evidence["model_path"] or not evidence["data_path"]:
        return evidence
    if not isinstance(audit, Mapping) or not audit.get("event_id") or not audit.get("event_hash") or not audit.get("path"):
        return evidence
    output_path = _path(payload.get("output_path"))
    if args.get("output_path") and output_path != _path(args["output_path"]):
        return evidence
    evidence.update(status="success", n_samples=count, output_path=output_path or None)
    evidence["audit"] = {key: audit[key] for key in ("event_id", "event_hash", "path", "model_sha256", "input_sha256", "output_sha256") if audit.get(key)}
    for key in ("prediction_mean", "prediction_min", "prediction_max", "accepted_count", "needs_review_count", "mean_confidence"):
        value = payload.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value):
            evidence[key] = value
    if payload.get("task_kind") == "classification":
        evidence["task_kind"] = "classification"
    components = payload.get("predictions_summary")
    if isinstance(components, list):
        evidence["predictions_summary"] = [{key: item[key] for key in ("name", "mean", "min", "max") if key in item} for item in components[:20] if isinstance(item, Mapping)]
        evidence["n_targets"] = len(components)
    preprocessing = payload.get("preprocessing")
    if isinstance(preprocessing, Mapping):
        evidence["preprocessing"] = {key: preprocessing[key] for key in ("description", "applied", "shared") if key in preprocessing}
    drift = payload.get("drift")
    if isinstance(drift, Mapping):
        evidence["drift"] = {key: drift[key] for key in ("available", "method", "drift_score", "reason") if key in drift}
        if isinstance(drift.get("flagged_indices"), list):
            evidence["drift"]["flagged_count"] = len(drift["flagged_indices"])
    return evidence
