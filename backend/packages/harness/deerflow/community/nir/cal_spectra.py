"""Bounded Cal curves captured from the actual fitted preprocessing state.

Called only by the trusted search executor. No Test or target array is accepted;
the browser reads the hash-bound cache, never this function or the model.
"""

import numpy as np

from .process import _digest


def capture_cal_spectra(X_cal, wv, model, *, session_id, experiment_id, decision_id, configuration, model_hash, axis_unit=None):
    from nir_core.preprocess.pipeline import PreprocessingPipeline

    preprocessor = model.preprocessor
    traced = isinstance(preprocessor, PreprocessingPipeline)
    steps = preprocessor.steps if traced else []
    if len(steps) > 32:
        raise ValueError("Too many preprocessing steps for bounded Cal evidence")
    rows = np.linspace(0, len(X_cal) - 1, min(12, len(X_cal)), dtype=int)
    # Bound the whole cache, not just each chart. Every snapshot keeps exactly
    # the same rows and ordered channels; training always uses full resolution.
    max_axis = min(512, 3000 // (len(steps) + 2))
    columns = np.linspace(0, X_cal.shape[1] - 1, min(max_axis, X_cal.shape[1]), dtype=int)
    raw = X_cal[rows]
    axis = np.asarray(wv) if wv is not None else np.arange(X_cal.shape[1])

    def chart(values, scope, y_label="spectral_value"):
        values = np.asarray(values, dtype=float)
        if values.shape != raw.shape or not np.all(np.isfinite(values)):
            raise ValueError("Actual preprocessing cannot supply finite same-axis Cal curves")
        return {
            "chart_schema_version": 1,
            "chart_id": _digest([experiment_id, scope])[:32],
            "run_id": "",
            "attempt_id": experiment_id,
            "step_execution_id": decision_id or "",
            "evidence_status": "available",
            "type": "spectra",
            "scope": scope,
            "x_label": "wavelength" if wv is not None else "variable",
            "x_unit": axis_unit if wv is not None else None,
            "y_label": y_label,
            "y_unit": None,
            "x": axis[columns].tolist(),
            "series": [{"id": _digest([session_id, "calibration", int(row)])[:12], "values": values[i, columns].tolist()} for i, row in enumerate(rows)],
            "n_total": len(X_cal),
            "n_returned": len(rows),
            "axis_total": X_cal.shape[1],
            "axis_returned": len(columns),
            "sampling": "even_cal_rows_and_ordered_original_axis_indices_v1",
        }

    before = chart(raw, "calibration_before")
    snapshots = []
    y_label = "spectral_value"

    def observe(index, step, values):
        nonlocal y_label
        # Use honest numerical labels, not an inferred absorbance/unit claim.
        if step.method in {"snv", "robust_snv", "rnv", "autoscale"}:
            y_label = "standardized_spectral_value"
        elif step.method == "normalize":
            y_label = "normalized_spectral_value"
        elif step.method == "mean_center":
            y_label = "centered_spectral_value"
        elif step.method in {"derivative1", "norris_derivative1"}:
            y_label = "first_derivative_value"
        elif step.method in {"derivative2", "norris_derivative2"}:
            y_label = "second_derivative_value"
        snapshots.append({"index": index, "method": step.method, "params": dict(step.params), "chart": chart(values, f"calibration_step_{index}", y_label)})

    # Traverse the real fitted pipeline once on the bounded Cal rows. No
    # prefix is independently fitted and no observer is attached to CV fits.
    processed = preprocessor.transform(raw, wv, on_step=observe) if traced else preprocessor.transform(raw, wv)
    return {
        "schema_version": 2 if traced else 1,
        "evidence_status": "available",
        "scope": "calibration",
        "session_id": session_id,
        "experiment_id": experiment_id,
        "decision_id": decision_id,
        "model_hash": model_hash,
        "configuration": configuration,
        "source": "actual_cal_fitted_preprocessing_state",
        "axis_unit_status": "declared_input_metadata" if axis_unit else "not_specified_by_input",
        "before": before,
        "after": chart(processed, "calibration_after", y_label),
        **({"steps": snapshots} if traced else {}),
    }


def validate_step_spectra(payload, configuration):
    """Validate v2 cumulative caches without data, estimators or numerical IO."""
    import math

    if payload.get("schema_version") == 1:
        return  # Preserve old immutable before/after caches without backfilling.
    steps = payload.get("steps")
    configured = configuration.get("steps")
    if payload.get("schema_version") != 2 or not isinstance(steps, list) or not isinstance(configured, list) or len(steps) != len(configured) or len(steps) > 32:
        raise ValueError("Invalid step evidence configuration")
    before = payload.get("before", {})
    if payload.get("source") != "actual_cal_fitted_preprocessing_state" or payload.get("evidence_status") != "available" or before.get("scope") != "calibration_before" or payload.get("after", {}).get("scope") != "calibration_after":
        raise ValueError("Invalid Cal evidence source")
    charts = [before, payload.get("after", {})]
    for i, step in enumerate(steps):
        if not isinstance(step, dict) or step.get("index") != i or step.get("method") != configured[i].get("method") or step.get("params") != configured[i].get("params", {}):
            raise ValueError("Step evidence identity mismatch")
        chart = step.get("chart", {})
        if chart.get("scope") != f"calibration_step_{i}":
            raise ValueError("Invalid step evidence scope")
        charts.append(chart)
    x = before.get("x", [])
    ids = [item.get("id") for item in before.get("series", [])]
    if not 2 <= len(x) <= 512 or not 1 <= len(ids) <= 12 or len(set(ids)) != len(ids) or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in x):
        raise ValueError("Invalid step evidence coordinates")
    if not all(isinstance(i, str) and i for i in ids) or type(before.get("n_total")) is not int or before["n_total"] < len(ids) or type(before.get("axis_total")) is not int or before["axis_total"] < len(x):
        raise ValueError("Invalid step evidence population")
    direction = 1 if x[1] > x[0] else -1
    if any((b - a) * direction <= 0 for a, b in zip(x, x[1:])):
        raise ValueError("Invalid step evidence axis")
    for chart in charts:
        series = chart.get("series", [])
        if (
            chart.get("type") != "spectra"
            or chart.get("attempt_id") != payload.get("experiment_id")
            or chart.get("step_execution_id") != (payload.get("decision_id") or "")
            or chart.get("x") != x
            or chart.get("x_unit") != before.get("x_unit")
            or chart.get("n_total") != before.get("n_total")
            or chart.get("axis_total") != before.get("axis_total")
            or chart.get("axis_returned") != len(x)
            or chart.get("n_returned") != len(ids)
            or [item.get("id") for item in series] != ids
            or any(len(item.get("values", [])) != len(x) or not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in item["values"]) for item in series)
        ):
            raise ValueError("Misaligned step evidence")
    final = steps[-1]["chart"] if steps else before
    if final["series"] != payload["after"]["series"]:
        raise ValueError("Final preprocessing output mismatch")
