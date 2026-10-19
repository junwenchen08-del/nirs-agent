"""Bounded display evidence from predictions already computed by safe search.

No estimator, file loader, fitting or Test input is accepted here. Full metrics
remain authoritative; displayed points are never used to recalculate them.
"""

import math

import numpy as np

from .process import _digest, _safe

SCOPES = {"calibration", "calibration_cv", "validation"}


def capture_round_diagnostics(y_cal, y_val, predictions, *, session_id, experiment_id, decision_id, model_hash, comparison_group_id, unit=None):
    charts = {}
    for scope in sorted(SCOPES):
        observed = np.asarray(y_val if scope == "validation" else y_cal, dtype=float).ravel()
        predicted = np.asarray(predictions[scope], dtype=float).ravel()
        if not len(observed) or observed.shape != predicted.shape or not np.all(np.isfinite(observed)) or not np.all(np.isfinite(predicted)):
            raise ValueError("Invalid original round predictions")
        indices = np.linspace(0, len(observed) - 1, min(2000, len(observed)), dtype=int)
        population = "validation" if scope == "validation" else "calibration"
        charts[scope] = {
            "scope": scope,
            "evidence_status": "available",
            "observed": observed[indices].tolist(),
            "predicted": predicted[indices].tolist(),
            "residual": (predicted[indices] - observed[indices]).tolist(),
            "sample_ids": [_digest([session_id, population, int(index)])[:16] for index in indices],
            "n_total": len(observed),
            "n_returned": len(indices),
            "sampling": "full_population" if len(indices) == len(observed) else "deterministic_evenly_spaced_max_2000",
            "unit": _safe(unit) if isinstance(unit, str) and len(unit) <= 50 else None,
        }
    return {
        "schema_version": 1,
        "evidence_status": "available",
        "session_id": session_id,
        "experiment_id": experiment_id,
        "decision_id": decision_id,
        "model_hash": model_hash,
        "comparison_group_id": comparison_group_id,
        "source": "original_search_predictions",
        "charts": charts,
    }


def validate_round_diagnostics(payload):
    if payload.get("schema_version") != 1 or payload.get("evidence_status") != "available" or not isinstance(payload.get("charts"), dict) or set(payload["charts"]) != SCOPES:
        raise ValueError("Invalid round chart contract")
    for scope, chart in payload["charts"].items():
        if not isinstance(chart, dict):
            raise ValueError("Invalid round chart record")
        unit = chart.get("unit")
        if chart.get("evidence_status") != "available" or unit is not None and (not isinstance(unit, str) or len(unit) > 50 or _safe(unit) != unit):
            raise ValueError("Invalid round chart metadata")
        arrays = [chart.get(key) for key in ("observed", "predicted", "residual")]
        if chart.get("scope") != scope or not all(isinstance(values, list) and 0 < len(values) <= 2000 and all(type(value) in {int, float} and math.isfinite(value) for value in values) for values in arrays):
            raise ValueError("Invalid round chart coordinates")
        observed, predicted, residual = arrays
        ids = chart.get("sample_ids")
        if not isinstance(ids, list) or not all(isinstance(value, str) and len(value) == 16 and all(char in "0123456789abcdef" for char in value) for value in ids):
            raise ValueError("Invalid round sample identities")
        if not len(observed) == len(predicted) == len(residual) == len(ids) == len(set(ids)) == chart.get("n_returned") or type(chart.get("n_total")) is not int or chart["n_total"] < len(observed):
            raise ValueError("Invalid round chart population")
        if not all(math.isclose(p - o, r, rel_tol=1e-9, abs_tol=1e-10) for o, p, r in zip(observed, predicted, residual, strict=True)):
            raise ValueError("Round residuals do not match predictions")
