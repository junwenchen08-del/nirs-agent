"""Small, display-safe snapshots of NIR candidate selection evidence."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

_SELECTION_KEYS = (
    "preprocessing_selection",
    "wavelength_selection",
    "wavelength_selection_decision",
    "wavelength_selection_candidates",
    "model_selection_decision",
    "model_candidates",
)
_SCHEMA: dict[str, tuple[str, ...]] = {
    "preprocessing_selection": ("budget", "calibration_samples", "candidate_count", "candidates", "selection_rule", "selected_candidate_id", "selected_pipeline", "reason_code"),
    "wavelength_selection": ("method", "n_selected", "n_components", "reason_code"),
    "wavelength_selection_decision": ("selected_method", "reason_code", "reason", "adoption", "evaluate", "min_relative_improvement"),
    "model_selection_decision": ("selected_method", "reason_code", "reason", "adoption", "evaluate", "min_relative_improvement"),
    "adoption": ("reason_code", "relative_RMSE_improvement", "minimum_required_improvement", "reason"),
    "candidate": ("candidate_id", "method", "steps", "reasons", "cv_rmse", "val_r2", "RMSE_tuning", "RMSECV", "n_selected", "n_components", "error", "selected"),
    "step": ("method", "name"),
}


def _safe(value: Any, *, kind: str = "", depth: int = 0) -> Any:
    if depth > 5:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and math.isfinite(value):
        return value
    if isinstance(value, str):
        text = value.strip()
        # Tool paths and long prose have no place in decision snapshots.
        if len(text) > 160 or "\\" in text or "/mnt/" in text or ":/" in text:
            return None
        return text or None
    if isinstance(value, list):
        if kind not in {"candidates", "wavelength_selection_candidates", "model_candidates", "steps", "selected_pipeline", "reasons"}:
            return None
        child_kind = "candidate" if "candidates" in kind else "step" if kind in {"steps", "selected_pipeline"} else ""
        return [item for raw in value[:8] if (item := _safe(raw, kind=child_kind, depth=depth + 1)) is not None]
    if isinstance(value, Mapping) and kind in _SCHEMA:
        return {key: safe for key in _SCHEMA[kind] if key in value and (safe := _safe(value[key], kind=key, depth=depth + 1)) is not None}
    return None


def selection_decision_facts(source: Mapping[str, Any] | None) -> dict[str, Any]:
    """Project only bounded, deterministic candidate selection facts."""
    if not isinstance(source, Mapping):
        return {}
    return {key: safe for key in _SELECTION_KEYS if key in source and (safe := _safe(source[key], kind=key)) is not None}


def model_result_facts(source: Mapping[str, Any]) -> dict[str, Any]:
    """Retain bounded classification and execution labels alongside choices."""
    facts = selection_decision_facts(source)
    for key in ("task_kind", "label_name", "method"):
        if (value := _safe(source.get(key))) is not None:
            facts[key] = value
    preprocessing = source.get("preprocessing")
    if isinstance(preprocessing, str):
        if (value := _safe(preprocessing)) is not None:
            facts["preprocessing"] = value
    elif isinstance(preprocessing, list):
        facts["preprocessing"] = [value for item in preprocessing[:8] if (value := _safe(item)) is not None]
    classes = source.get("classes")
    if isinstance(classes, list):
        facts["classes"] = [value for item in classes[:20] if (value := _safe(item)) is not None]
    distribution = source.get("class_distribution")
    if isinstance(distribution, Mapping):
        facts["class_distribution"] = {name: count for raw_name, raw_count in list(distribution.items())[:20] if (name := _safe(raw_name)) is not None and (count := _safe(raw_count)) is not None}
    return facts
