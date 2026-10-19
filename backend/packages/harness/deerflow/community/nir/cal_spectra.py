"""Bounded Cal curves captured from the actual fitted preprocessing state.

Called only by the trusted search executor. No Test or target array is accepted;
the browser reads the hash-bound cache, never this function or the model.
"""

import numpy as np

from .process import _digest


def capture_cal_spectra(X_cal, wv, model, *, session_id, experiment_id, decision_id, configuration, model_hash, axis_unit=None):
    rows = np.linspace(0, len(X_cal) - 1, min(12, len(X_cal)), dtype=int)
    columns = np.linspace(0, X_cal.shape[1] - 1, min(512, X_cal.shape[1]), dtype=int)
    raw = X_cal[rows]
    # Reuse Cal-fitted MSC/scaling/other state; never call fit/fit_transform.
    processed = np.asarray(model.preprocessor.transform(raw, wv), dtype=float)
    if processed.shape != raw.shape or not np.all(np.isfinite(processed)):
        raise ValueError("Actual preprocessing cannot supply finite same-axis Cal curves")
    axis = np.asarray(wv) if wv is not None else np.arange(X_cal.shape[1])

    def chart(values, scope):
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
            "y_label": "spectral_value",
            "y_unit": None,
            "x": axis[columns].tolist(),
            "series": [{"id": _digest([session_id, "calibration", int(row)])[:12], "values": values[i, columns].tolist()} for i, row in enumerate(rows)],
            "n_total": len(X_cal),
            "n_returned": len(rows),
            "axis_total": X_cal.shape[1],
            "axis_returned": len(columns),
            "sampling": "even_cal_rows_and_ordered_original_axis_indices_v1",
        }

    return {
        "schema_version": 1,
        "evidence_status": "available",
        "scope": "calibration",
        "session_id": session_id,
        "experiment_id": experiment_id,
        "decision_id": decision_id,
        "model_hash": model_hash,
        "configuration": configuration,
        "source": "actual_cal_fitted_preprocessing_state",
        "axis_unit_status": "declared_input_metadata" if axis_unit else "not_specified_by_input",
        "before": chart(raw, "calibration_before"),
        "after": chart(processed, "calibration_after"),
    }
