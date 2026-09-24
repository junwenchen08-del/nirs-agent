"""NIR data I/O tools: load / inspect / predict.

These tools bridge DeerFlow's virtual sandbox paths to nir_core's data
loaders and serialised-model inference. They never return spectral matrices
to the LLM — only JSON metadata and (optionally) saved .npz / .csv paths.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import numpy as np
from langchain.tools import InjectedToolCallId, tool
from nir_core.io.resources import ResourceLimitError

from deerflow.runtime.user_context import resolve_runtime_user_id
from deerflow.tools.types import Runtime

from ._common import (
    _err,
    _load_npz_safely,
    _load_trusted_model_artifact,
    _ok,
    _resolve,
    _sha256_file,
)
from ._drift_monitor import observe_prediction_drift
from ._prediction_audit import append_prediction_audit
from ._resources import budget_for_runtime, check_spectral_data, resource_error

_PREDICTION_AUDIT_PATH = "/mnt/user-data/outputs/prediction-audit.jsonl"
_DRIFT_STATE_PATH = "/mnt/user-data/outputs/prediction-drift-state.json"
_DRIFT_ALERTS_PATH = "/mnt/user-data/outputs/prediction-drift-alerts.jsonl"


def _runtime_audit_attribution(runtime: Runtime, tool_call_id: str) -> dict[str, str | None]:
    context = getattr(runtime, "context", None)
    context = context if isinstance(context, dict) else {}
    return {
        "user_id": resolve_runtime_user_id(runtime),
        "thread_id": str(context["thread_id"]) if context.get("thread_id") else None,
        "run_id": str(context["run_id"]) if context.get("run_id") else None,
        "trace_id": (str(context["deerflow_trace_id"]) if context.get("deerflow_trace_id") else None),
        "tool_call_id": tool_call_id or None,
    }


def _prediction_summary_for_audit(payload: dict) -> dict | list[dict] | None:
    if isinstance(payload.get("predictions_summary"), list):
        return payload["predictions_summary"]
    if payload.get("task_kind") == "classification":
        return {
            "class_counts": payload.get("class_counts"),
            "accepted_count": payload.get("accepted_count"),
            "needs_review_count": payload.get("needs_review_count"),
            "mean_confidence": payload.get("mean_confidence"),
        }
    keys = ("prediction_mean", "prediction_min", "prediction_max")
    if any(key in payload for key in keys):
        return {key.removeprefix("prediction_"): payload[key] for key in keys}
    return None


def _drift_summary_for_audit(payload: dict, *, requested: bool) -> dict:
    drift = payload.get("drift")
    if not requested:
        return {"requested": False}
    if not isinstance(drift, dict):
        return {"requested": True, "available": False}
    summary = {
        "requested": True,
        "available": bool(drift.get("available")),
        "method": drift.get("method"),
        "drift_score": drift.get("drift_score"),
        "reason": drift.get("reason"),
    }
    flagged = drift.get("flagged_indices")
    if isinstance(flagged, list):
        summary["flagged_count"] = len(flagged)
    components = drift.get("per_component")
    if isinstance(components, list):
        summary["per_component"] = [
            {
                "name": component.get("name"),
                "drift_score": component.get("drift_score"),
                "flagged_count": len(component.get("flagged_indices") or []),
            }
            for component in components
            if isinstance(component, dict)
        ]
    return summary


def _drift_flagged_count(drift: dict) -> int | None:
    flagged = drift.get("flagged_indices")
    if isinstance(flagged, list):
        return len(flagged)
    components = drift.get("per_component")
    if isinstance(components, list):
        counts = [len(component.get("flagged_indices") or []) for component in components if isinstance(component, dict)]
        return max(counts, default=0)
    return None


def _unwrap_model_artifact(artifact):
    """Return model and transform metadata for plain models and NIR artifacts."""
    if isinstance(artifact, dict) and artifact.get("format") == "nir_model_artifact":
        return (
            artifact.get("model"),
            artifact.get("preprocessing") or {},
            artifact.get("wavelength_selection") or {},
            artifact.get("monitoring_reference"),
        )
    return artifact, {}, {}, None


def _apply_artifact_preprocessing(
    X: np.ndarray,
    wv: np.ndarray | None,
    preprocessing: dict,
    *,
    input_preprocessed: bool,
) -> tuple[np.ndarray, bool]:
    """Apply a fitted version-2 artifact pipeline to raw prediction spectra."""
    pipeline = preprocessing.get("pipeline") if preprocessing else None
    if input_preprocessed or pipeline is None or not preprocessing.get("apply_on_predict"):
        return X, False
    if pipeline.has_stateful_steps and not pipeline.fitted():
        raise ValueError("Model artifact contains an unfitted stateful preprocessing pipeline; cannot safely apply it to prediction spectra.")
    return np.asarray(pipeline.transform(X, wv), dtype=float), True


def _apply_artifact_wavelength_selection(X: np.ndarray, wavelength_selection: dict) -> np.ndarray:
    """Slice prediction spectra with the train-time selected wavelength indices."""
    if not wavelength_selection or wavelength_selection.get("method") in {None, "none"}:
        return X

    indices = wavelength_selection.get("selected_indices")
    if not indices:
        return X

    selected = [int(i) for i in indices]
    if X.shape[1] == len(selected):
        return X
    if max(selected) >= X.shape[1]:
        raise ValueError(f"Prediction data has {X.shape[1]} wavelengths, but the model expects original indices up to {max(selected)}.")
    return X[:, selected]


def _parse_y_cols(value: str | None) -> list[int] | None:
    """Parse a compact multi-target column selector."""
    if value is None:
        return None
    selected: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            parts = token.split(":")
            if len(parts) != 2 or not parts[0].strip() or not parts[1].strip():
                raise ValueError("y_cols ranges must use explicit 'start:stop' bounds")
            selected.extend(range(int(parts[0]), int(parts[1])))
        else:
            selected.append(int(token))
    selected = list(dict.fromkeys(selected))
    if not selected:
        raise ValueError("y_cols must select at least one column")
    return selected


def _compact_inspection_sequence(value, *, limit: int = 32):
    """Bound long schema lists before they enter the model context."""
    if not isinstance(value, list) or len(value) <= limit:
        return value
    if all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        contiguous = all(right == left + 1 for left, right in zip(value, value[1:]))
        if contiguous:
            return {"start": value[0], "stop": value[-1] + 1, "count": len(value)}
    numeric = [float(item) for item in value if isinstance(item, (int, float))]
    result = {"count": len(value), "first": value[:5], "last": value[-3:]}
    if len(numeric) == len(value):
        result["min"] = min(numeric)
        result["max"] = max(numeric)
    return result


def _categorical_column_profiles(path: str, *, row_limit: int = 1000) -> list[dict]:
    """Return bounded label/group candidates without exposing sample rows."""

    suffix = Path(path).suffix.lower()
    if suffix not in {".csv", ".tsv"}:
        return []
    import pandas as pd

    frame = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",", nrows=row_limit)
    sampled_rows = int(frame.shape[0])
    profiles: list[dict] = []
    for index, column in enumerate(frame.columns):
        values = frame[column].dropna()
        n_observed = int(values.shape[0])
        n_unique = int(values.nunique(dropna=True))
        if n_unique < 2 or n_unique > 20 or n_unique >= n_observed:
            continue
        counts = values.astype(str).value_counts().head(12)
        profiles.append(
            {
                "index": index,
                "name": str(column),
                "n_unique": n_unique,
                "class_counts": {str(label)[:80]: int(count) for label, count in counts.items()},
                "counts_truncated": n_unique > len(counts),
                "sampled_rows": sampled_rows,
                "profile_is_sampled": sampled_rows >= row_limit,
            }
        )
    return profiles


def _classification_label_row_profiles(path: str, *, row_limit: int = 25) -> list[dict]:
    """Return bounded samples-in-columns label-row candidates."""

    suffix = Path(path).suffix.lower()
    if suffix not in {".csv", ".tsv"}:
        return []
    import pandas as pd

    frame = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",", header=None, nrows=row_limit)
    if frame.shape[1] < 3:
        return []
    sample_indices = list(range(1, frame.shape[1]))
    profiles: list[dict] = []
    for row_index in range(frame.shape[0]):
        values = frame.iloc[row_index, sample_indices].dropna().astype(str).str.strip()
        values = values[values != ""]
        n_observed = int(values.shape[0])
        n_unique = int(values.nunique(dropna=True))
        if n_unique < 2 or n_unique > 20 or n_unique >= n_observed:
            continue
        counts = values.value_counts().head(12)
        min_count = int(counts.min()) if not counts.empty else 0
        if min_count < 5:
            continue
        numeric = pd.to_numeric(values, errors="coerce").notna().mean() if n_observed else 1.0
        profiles.append(
            {
                "row_index": row_index,
                "n_unique": n_unique,
                "class_counts": {str(label)[:80]: int(count) for label, count in counts.items()},
                "counts_truncated": n_unique > len(counts),
                "sample_cols": "1:",
                "wavenumber_col": 0,
                "likely_label_row": bool(numeric < 0.5),
                "sampled_columns": len(sample_indices),
            }
        )
    profiles.sort(key=lambda item: (item["likely_label_row"], item["sampled_columns"], -item["row_index"]), reverse=True)
    return profiles


@tool("nir_load_data", parse_docstring=True)
def nir_load_data_tool(
    runtime: Runtime,
    file_path: str,
    output_path: str | None = None,
    y_col: int | None = None,
    y_cols: str | None = None,
    wv_row: int | None = None,
    x_cols: str | None = None,
    subset: str | None = None,
    x_var: str | None = None,
    y_var: str | None = None,
    wv_var: str | None = None,
    transpose: bool = False,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",  # noqa: ARG001
) -> str:
    """Load a near-infrared spectral data file (.mat / .csv / .txt).

    Auto-detects the file format and internal layout (samples-in-rows vs
    samples-in-columns) and standardises it to a SpectralData container.
    Optionally saves a normalised .npz for downstream preprocessing.

    When ``y_col`` / ``wv_row`` are omitted (the common case), the loader
    auto-detects the canonical NIR "row-label + column-label" layout (empty
    corner cell, wavelength header row, reference-value column). Pass them
    explicitly ONLY when auto-detection fails — e.g. unusual CSV layouts
    without an empty corner cell, or when the first column is NOT the
    reference values. See the ``nir_inspect`` hint for guidance.

    ★ v3.7: MATLAB struct (.mat) files are now supported. Many public NIR
    datasets (Open-Nirs-Datasets, Melamine, etc.) store data as nested
    structs with multiple sub-datasets (e.g. R562 / R568 / R861 / R862).
    When ``nir_inspect`` reports ``is_struct: true`` and
    ``available_subsets: [...]``, pass the chosen subset name here via
    ``subset``. When the struct contains multiple sub-datasets and
    ``subset`` is omitted, this tool returns an error listing the available
    subsets — call ``nir_inspect`` first to see them, then re-call with
    ``subset=...``. Multiple X* blocks (X1, X2, ...) are concatenated
    column-wise automatically with their matching wn* wavelength vectors.

    ★ v3.8: ``x_cols`` lets you select which columns form the spectra block
    ``X``, skipping metadata columns. This is essential for public datasets
    like Anderson 2020 mango (CSV with 8 leading metadata columns: Set,
    Season, Region, Date, Type, Cultivar, Pop, Temp) where cols 0-7 are
    non-numeric and would otherwise leak into X as NaN. Syntax:
    ``"8:"`` (col 8 to end), ``"8:314"`` (half-open), ``":8"``,
    ``"8,9,10"`` (explicit list). When ``y_col`` is inside the selected
    range it is automatically excluded from X. Call ``nir_inspect`` first
    — when it reports ``non_numeric_columns: [0,1,2,...]`` or
    ``hint: use x_cols="N:" to skip metadata``, pass that x_cols value here.

    Args:
        file_path: Virtual path to the input file, e.g.
            ``/mnt/user-data/uploads/spectra.mat``.
        output_path: Optional virtual path for a standardised .npz output,
            e.g. ``/mnt/user-data/workspace/data.npz``.
        y_col: Optional 0-based index of the reference-value column
            (CSV/TXT only; ignored for .mat). When provided, auto-detection
            is bypassed and this column is split as y. ``None`` (default)
            triggers auto-detection.
        y_cols: Optional multi-component reference columns as a comma list
            or half-open range string. Mutually exclusive with y_col.
            Multi-component loading stores y as a two-dimensional matrix.
        wv_row: Optional 0-based index of the wavelength row (CSV/TXT only;
            ignored for .mat). When provided, auto-detection is bypassed and
            this row is split as wv. ``None`` (default) triggers
            auto-detection.
        x_cols: ★ v3.8 Optional column selector for the spectra block X
            (CSV/TXT only; ignored for .mat). Use this to skip metadata
            columns using a slice or comma-separated list. When y_col is
            inside the selected range it is automatically excluded from X.
            For the Anderson mango layout, select spectra after the metadata
            and reference columns.
        subset: Optional name of a sub-dataset inside a MATLAB struct .mat
            file (e.g. ``"R562"``). Call ``nir_inspect`` first to see the
            available subsets; when multiple exist and this is None, the
            tool returns an error listing them.
        x_var: Optional MATLAB field path for spectra, including dotted paths
            reported by nir_inspect (for example ``Experiment.block_b``).
            Use only after a low-confidence inspection asks for confirmation.
        y_var: Optional MATLAB field path for reference values.
        wv_var: Optional MATLAB field path for the wavelength/wavenumber axis.
        transpose: Set true only when nir_inspect reports that the explicitly
            selected MATLAB spectra matrix stores samples in columns.

    Returns:
        JSON summary of the loaded data (sample count, wavelength range,
        reference value range, source format). Spectral matrices are NOT
        returned — only metadata.
    """
    try:
        from nir_core.io.loaders import auto_detect_and_load, load_csv, load_mat
        from nir_core.io.sniffers import detect_format, inspect_file
        from nir_core.io.writers import save_npz

        budget = budget_for_runtime(runtime)
        real_in = _resolve(runtime, file_path, read_only=True)
        budget.check_file(real_in, stage="load_file_preflight")

        parsed_y_cols = _parse_y_cols(y_cols)
        if y_col is not None and parsed_y_cols is not None:
            return _err("y_col and y_cols are mutually exclusive")

        # When explicit y_col / wv_row / x_cols are provided for CSV/TXT files,
        # bypass auto-detection and call load_csv with the override.
        # This gives the agent an escape hatch when auto-detection fails,
        # so it never needs to fall back to writing Python scripts.
        file_format = detect_format(real_in)
        csv_override = y_col is not None or parsed_y_cols is not None or wv_row is not None or x_cols is not None
        mat_override = x_var is not None or y_var is not None or wv_var is not None or transpose
        has_override = csv_override or mat_override
        if not has_override and subset is None:
            inspected = json.loads(inspect_file(real_in))
            mapping = inspected.get("schema_mapping") or {}
            if mapping.get("status") == "needs_user_mapping":
                return _err(
                    "Field mapping confirmation required before loading. "
                    f"Candidate spectra fields: {mapping.get('x_candidates', [])}; "
                    f"candidate reference fields: {mapping.get('y_candidates', [])}; "
                    f"candidate numeric columns: {mapping.get('candidate_numeric_columns', [])}. "
                    "Call nir_inspect, ask the user to confirm the mapping, then retry "
                    "with x_var/y_var/wv_var or y_col/x_cols."
                )
        if csv_override and file_format != "mat":
            data = load_csv(real_in, y_col=y_col, y_cols=parsed_y_cols, wv_row=wv_row, x_cols=x_cols)
        elif file_format == "mat" and (subset is not None or mat_override):
            data = load_mat(
                real_in,
                x_var=x_var,
                y_var=y_var,
                wv_var=wv_var,
                subset=subset,
                transpose=transpose,
            )
        else:
            data = auto_detect_and_load(real_in)
        check_spectral_data(
            budget,
            data,
            stage="loaded_spectral_matrix",
            peak_multiplier=3.0,
        )

        if output_path:
            budget.checkpoint("save_normalized_data")
            real_out = _resolve(runtime, output_path, read_only=False)
            os.makedirs(os.path.dirname(real_out), exist_ok=True)
            save_npz(data, real_out)
        summary = data.summary()
        summary["output_saved"] = bool(output_path)
        summary["layout_override_used"] = has_override
        summary["subset_used"] = subset
        summary["x_cols_used"] = x_cols
        summary["y_cols_used"] = parsed_y_cols
        summary["x_var_used"] = x_var
        summary["y_var_used"] = y_var
        summary["wv_var_used"] = wv_var
        summary["transpose_used"] = transpose
        summary["resource_budget"] = budget.evidence()
        # ⭐ Surface the auto-detected layout so the agent can confirm y and
        # wv have been correctly separated from the raw CSV block. Without
        # these explicit fields the agent tends to "double-check" by writing
        # its own Python script — which is exactly the anti-pattern this
        # tool exists to prevent.
        summary["y_separated"] = data.y is not None
        summary["wv_separated"] = data.wv is not None
        if data.y is not None:
            y_arr = np.asarray(data.y, dtype=float)
            if y_arr.ndim == 1:
                summary["y_first_values"] = [round(float(v), 4) for v in y_arr[:5]]
                summary["y_last_values"] = [round(float(v), 4) for v in y_arr[-3:]]
            else:
                summary["y_first_values"] = np.round(y_arr[:5], 4).tolist()
                summary["y_last_values"] = np.round(y_arr[-3:], 4).tolist()
        if data.wv is not None:
            summary["wv_first_values"] = [round(float(v), 2) for v in data.wv[:3]]
            summary["wv_last_values"] = [round(float(v), 2) for v in data.wv[-3:]]
        if not summary["y_separated"]:
            summary["warning"] = (
                "No reference values (y) detected. The file may be spectra-only, "
                "or its layout does not match the auto-detected NIR pattern. "
                "Use nir_inspect to confirm the structure, then re-call "
                "nir_load_data with explicit y_col / wv_row if needed."
            )
        return _ok(summary)
    except ResourceLimitError as exc:
        return resource_error(exc)
    except MemoryError:
        return _err("内存不足: 文件过大，无法加载。请先用 nir_inspect 查看文件结构，或使用更小的数据子集。")
    except Exception as exc:  # noqa: BLE001
        return _err(f"{type(exc).__name__}: {exc}")


@tool("nir_inspect", parse_docstring=True)
def nir_inspect_tool(
    runtime: Runtime,
    file_path: str,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",  # noqa: ARG001
) -> str:
    """Inspect a spectral file's structure without fully loading it.

    Returns format, shape, estimated sample/wavelength counts, value range,
    NaN presence, and (when available) ``axis_first``, ``axis_last``, and
    ``axis_direction`` as a JSON report. Start the NIR workflow first and wait
    for that tool result before calling this tool; do not issue both calls in
    parallel. For an inspection-only workflow, a successful call completes the
    workflow automatically and no planning or modeling step should follow.

    Args:
        file_path: Virtual path to the file to inspect.
    """
    try:
        from nir_core.io.sniffers import inspect_file

        budget = budget_for_runtime(runtime)
        real_in = _resolve(runtime, file_path, read_only=True)
        budget.check_file(real_in, stage="inspect_file_preflight")
        raw = inspect_file(real_in)
        # inspect_file returns a JSON string. Parse it so we can layer extra
        # advisory fields and re-serialize.
        import json as _json

        try:
            info = _json.loads(raw)
        except (TypeError, ValueError):
            return raw

        try:
            categorical_columns = _categorical_column_profiles(real_in)
        except Exception:  # noqa: BLE001 - advisory profiling must not block inspection
            categorical_columns = []
        try:
            label_rows = _classification_label_row_profiles(real_in)
        except Exception:  # noqa: BLE001 - advisory profiling must not block inspection
            label_rows = []
        if categorical_columns:
            info["categorical_columns"] = categorical_columns
            info["classification_hint"] = (
                "Candidate class-label or grouping columns were profiled from at most 1000 rows. Confirm label_col with the user, and use a batch/sample/origin column as group_col when related spectra must stay together."
            )
        if label_rows:
            info["classification_label_rows"] = label_rows
            info["classification_hint"] = (
                "Samples-in-columns class-label rows were detected. For qualitative classification, "
                'call nir_train_classifier with label_row=<row_index>, sample_cols="1:", '
                "wavenumber_col=0, and optional group_row when replicate/sample groups must stay together."
            )

        schema_mapping = info.get("schema_mapping") or {}
        for key in ("spectral_columns", "candidate_numeric_columns", "wavelengths"):
            if key in schema_mapping:
                schema_mapping[key] = _compact_inspection_sequence(schema_mapping[key])
        if schema_mapping:
            info["schema_mapping"] = schema_mapping
        for key in ("column_names", "spectral_column_indices"):
            if key in info:
                info[key] = _compact_inspection_sequence(info[key])

        # Add layout advisory so the agent knows whether the NIR-style
        # "row-label + column-label" pattern is present without having to
        # open the file in Python.
        corner_nan = bool(info.get("corner_is_nan"))
        if info.get("labeled_matrix"):
            info["layout_pattern"] = "labeled_matrix_bundle"
        elif (info.get("schema_mapping") or {}).get("status") == "auto":
            info["layout_pattern"] = "schema_inferred"
        elif (info.get("schema_mapping") or {}).get("status") == "needs_user_mapping":
            info["layout_pattern"] = "mapping_required"
        else:
            info["layout_pattern"] = "row_label_and_column_label" if corner_nan else "plain_matrix"

        # ★ v3.8: Non-numeric metadata column advisory. When inspect detects
        # columns that are almost entirely NaN (= string columns like
        # Set/Season/Region), tell the agent exactly which x_cols value to
        # pass to nir_load_data to skip them.
        non_numeric_cols: list[int] = info.get("non_numeric_columns") or []
        metadata_hint = ""
        if non_numeric_cols:
            # Find the first numeric column index (smallest index not in
            # non_numeric_cols) — that's where spectra likely start.
            n_cols_total = info.get("shape", [0, 0])[1] if isinstance(info.get("shape"), list) else 0
            first_numeric = next(
                (i for i in range(n_cols_total) if i not in non_numeric_cols),
                None,
            )
            if first_numeric is not None and first_numeric > 0:
                metadata_hint = (
                    f" ★ v3.8: {len(non_numeric_cols)} non-numeric metadata column(s) "
                    f"detected at indices {non_numeric_cols}. These contain strings "
                    f"(e.g. Set/Season/Region/Cultivar) and would leak into X as NaN, "
                    f'breaking PLS/SVR. Pass x_cols="{first_numeric}:" to '
                    f"nir_load_data to skip them. The first numeric column is at "
                    f"index {first_numeric}."
                )
            elif non_numeric_cols:
                metadata_hint = f" ★ v3.8: Non-numeric column(s) at indices {non_numeric_cols} detected. Use x_cols with nir_load_data to select only the numeric spectra columns."

        # ★ v3.7: Struct .mat advisory — when the file is a MATLAB struct,
        # tell the agent exactly how to call nir_load_data(subset=...).
        schema_mapping = info.get("schema_mapping") or {}
        mapping_needs_confirmation = schema_mapping.get("status") == "needs_user_mapping" and not (info.get("is_struct") and info.get("available_subsets"))
        if info.get("labeled_matrix"):
            info["hint"] = (
                "MATLAB labeled matrix detected. nir_load_data will automatically "
                f"use {info.get('target_columns', [])} as reference values, exclude "
                f"metadata columns {info.get('metadata_columns', [])}, parse numeric "
                "VarLabels as the spectral axis, and preserve ObjLabels as sample names. "
                "Call nir_load_data directly without y_col, x_cols, wv_row, or Python scripts."
            )
        elif mapping_needs_confirmation:
            info["action_required"] = "confirm_field_mapping"
            x_candidates = schema_mapping.get("x_candidates") or []
            y_candidates = schema_mapping.get("y_candidates") or []
            numeric_columns = schema_mapping.get("candidate_numeric_columns") or []
            info["hint"] = (
                "The file was parsed successfully, but its field roles are ambiguous, "
                f"so modeling has been paused. Candidate spectra fields: {x_candidates}; "
                f"candidate reference fields: {y_candidates}; candidate numeric CSV "
                f"columns: {numeric_columns}. Ask the user to confirm which field/columns "
                "are spectra and which are reference values, then call nir_load_data with "
                "explicit mapping arguments. Do not write a script or guess between equal candidates."
            )
        elif schema_mapping.get("status") == "auto":
            info["hint"] = (
                "A high-confidence schema mapping was inferred from field names, numeric "
                "column headers, dimensions, and axis monotonicity. nir_load_data will use "
                f"this mapping directly: {schema_mapping}. No script or manual parsing is needed."
            )
        elif info.get("is_struct"):
            subsets = info.get("available_subsets") or []
            if subsets:
                info["hint"] = (
                    f"MATLAB struct detected with {len(subsets)} sub-datasets: "
                    f"{subsets}. Call nir_load_data with subset=<name> "
                    f"(e.g. subset='{subsets[0]}'). Each sub-dataset contains "
                    f"X fields {info.get('x_fields', [])} and Y fields "
                    f"{info.get('y_fields', [])}; multiple X* blocks will be "
                    f"concatenated column-wise with their matching wn* vectors."
                )
            else:
                info["hint"] = (
                    f"MATLAB struct detected (no nested sub-datasets). nir_load_data will auto-flatten the struct and pick X from {info.get('x_fields', [])}, Y from {info.get('y_fields', [])}, wv from {info.get('wv_fields', [])}."
                )
        elif corner_nan:
            info["hint"] = (
                "Empty/NaN corner cell detected — file uses the canonical NIR "
                "row-label + column-label layout. nir_load_data will auto-split "
                "y (first column) and wv (first row) from X. The 'structure' "
                "field above is just the raw shape heuristic and may misclassify "
                "this layout as 'samples_in_columns' because columns > rows; "
                "that is NOT a bug — the corner NaN signal is the authoritative cue."
            ) + metadata_hint
        elif non_numeric_cols:
            # Metadata columns present but no corner NaN — this is the
            # "plain matrix with metadata prefix" layout (e.g. Anderson
            # mango CSV). The hint should prioritise the x_cols advice.
            info["hint"] = (
                "No empty corner cell, but non-numeric metadata columns detected. "
                "The file has a metadata prefix (non-numeric columns) followed by "
                "numeric spectra columns. You MUST pass x_cols to nir_load_data "
                "to skip the metadata columns, otherwise they leak into X as NaN "
                "and break PLS/SVR. Also pass y_col (index of the reference-value "
                "column) and wv_row if applicable."
            ) + metadata_hint
        else:
            info["hint"] = ("No empty corner cell detected. The file may use a plain matrix layout (no separate y column or wavelength header). Pass y_col / wv_row to nir_load_data explicitly if needed.") + metadata_hint
        info["resource_budget"] = budget.evidence()
        return _json.dumps(info, ensure_ascii=False)
    except ResourceLimitError as exc:
        return resource_error(exc)
    except Exception as exc:  # noqa: BLE001
        return _err(f"{type(exc).__name__}: {exc}")


@tool("nir_predict", parse_docstring=True)
def nir_predict_tool(
    runtime: Runtime,
    model_path: str,
    data_path: str,
    output_path: str | None = None,
    detect_drift: bool = True,
    input_preprocessed: bool = False,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",  # noqa: ARG001
) -> str:
    """Predict reference values for new spectra using a trained model.

    Loads a joblib-serialised model and applies its fitted preprocessing and
    wavelength selection metadata to raw spectra before inference. Version-3
    multi-output artifacts produce one named prediction column per component.
    Optionally runs PCA Hotelling T²/Q drift detection against the monitoring
    reference persisted from the model's training domain. Every attempt is
    appended to a tamper-evident audit chain without raw spectra or row-level
    predictions. Available drift results also update a model-scoped continuous
    monitor that emits alert and recovery events only on state transitions.

    Args:
        model_path: Virtual path to the .pkl model file.
        data_path: Virtual path to a new-data .npz containing X and optional wv.
        output_path: Optional virtual path for a CSV of predictions.
        detect_drift: If True, score the transformed spectra against the
            training-domain monitoring reference. Legacy artifacts without a
            reference report drift as unavailable.
        input_preprocessed: Set True only when X already has the artifact's
            train-time preprocessing applied.

    Returns:
        JSON with predictions summary (count, mean, min, max), audit metadata,
        and optional drift info. Individual predictions are saved to
        output_path if given.
    """
    started_at = time.perf_counter()
    real_audit: str | None = None
    model_sha256: str | None = None
    data_sha256: str | None = None
    input_shape: tuple[int, int] | None = None
    artifact_format_version: object | None = None

    def finish(payload: dict, error: Exception | None = None) -> str:
        if real_audit is None:
            original = f"; original error: {type(error).__name__}: {error}" if error else ""
            return _err(f"Prediction audit storage is unavailable; prediction results were not released{original}")
        attribution = _runtime_audit_attribution(runtime, tool_call_id)
        if payload.get("status") == "ok":
            drift = payload.get("drift")
            if not detect_drift:
                payload["drift_monitoring"] = {
                    "status": "disabled",
                    "reason": "drift_detection_not_requested",
                }
            elif not isinstance(drift, dict) or not drift.get("available"):
                payload["drift_monitoring"] = {
                    "status": "unavailable",
                    "reason": (drift.get("reason", "drift_result_unavailable") if isinstance(drift, dict) else "drift_result_unavailable"),
                }
            elif model_sha256 is None or input_shape is None:
                payload["drift_monitoring"] = {
                    "status": "error",
                    "error_type": "MissingPredictionLineage",
                }
            else:
                try:
                    output_directory = Path(real_audit).parent
                    monitoring = observe_prediction_drift(
                        state_path=output_directory / Path(_DRIFT_STATE_PATH).name,
                        alerts_path=output_directory / Path(_DRIFT_ALERTS_PATH).name,
                        model_sha256=model_sha256,
                        model_path=model_path,
                        data_sha256=data_sha256,
                        drift_score=float(drift["drift_score"]),
                        n_samples=input_shape[0],
                        flagged_count=_drift_flagged_count(drift),
                        attribution=attribution,
                    )
                    monitoring["state_path"] = _DRIFT_STATE_PATH
                    monitoring["alerts_path"] = _DRIFT_ALERTS_PATH
                    payload["drift_monitoring"] = monitoring
                except Exception as monitoring_error:  # noqa: BLE001
                    payload["drift_monitoring"] = {
                        "status": "error",
                        "error_type": type(monitoring_error).__name__,
                    }
        event_id = str(uuid.uuid4())
        event = {
            "schema_version": 1,
            "event_id": event_id,
            "occurred_at": datetime.now(UTC).isoformat(),
            "duration_ms": round((time.perf_counter() - started_at) * 1000, 3),
            "status": "success" if payload.get("status") == "ok" else "error",
            "attribution": attribution,
            "model": {
                "path": model_path,
                "sha256": model_sha256,
                "artifact_format_version": artifact_format_version,
            },
            "input": {
                "path": data_path,
                "sha256": data_sha256,
                "n_samples": input_shape[0] if input_shape else None,
                "n_wavelengths": input_shape[1] if input_shape else None,
                "already_preprocessed": input_preprocessed,
            },
            "output_path": payload.get("output_path") or output_path,
            "prediction_summary": _prediction_summary_for_audit(payload),
            "drift": _drift_summary_for_audit(payload, requested=detect_drift),
            "drift_monitoring": payload.get("drift_monitoring"),
            "error": ({"type": type(error).__name__, "message": str(error)[:1000]} if error is not None else None),
        }
        try:
            persisted = append_prediction_audit(real_audit, event)
        except Exception as audit_error:  # noqa: BLE001
            original = f"; original error: {type(error).__name__}: {error}" if error else ""
            return _err(f"Prediction audit persistence failed; prediction results were not released: {type(audit_error).__name__}{original}")
        payload["audit"] = {
            "event_id": event_id,
            "event_hash": persisted["event_hash"],
            "path": _PREDICTION_AUDIT_PATH,
        }
        return _ok(payload)

    try:
        real_audit = _resolve(runtime, _PREDICTION_AUDIT_PATH, read_only=False)
        real_model = _resolve(runtime, model_path, read_only=True)
        real_data = _resolve(runtime, data_path, read_only=True)
        model_sha256 = _sha256_file(real_model)
        data_sha256 = _sha256_file(real_data)
        artifact = _load_trusted_model_artifact(real_model, model_path)
        if isinstance(artifact, dict):
            artifact_format_version = artifact.get("version")
        data_dict = _load_npz_safely(real_data)
        X = np.asarray(data_dict["X"], dtype=float)
        if X.ndim != 2:
            raise ValueError("Prediction X must be a two-dimensional spectra matrix.")
        input_shape = (int(X.shape[0]), int(X.shape[1]))
        wv = np.asarray(data_dict["wv"], dtype=float).ravel() if data_dict.get("wv") is not None and data_dict["wv"].size else None

        if isinstance(artifact, dict) and artifact.get("multi_output") is True:
            models = artifact.get("models") or []
            names = [str(name) for name in artifact.get("component_names") or []]
            selections = artifact.get("wavelength_selection") or []
            monitoring_references = artifact.get("monitoring_reference") or []
            preprocessing_config = artifact.get("preprocessing") or {}
            n_targets = int(artifact.get("n_targets", len(models)))
            if not (len(models) == len(names) == len(selections) == n_targets):
                raise ValueError("Multi-output artifact has inconsistent model, name, or wavelength-selection counts.")

            predictions: list[np.ndarray] = []
            preprocessing_applied: list[bool] = []
            drift_results: list[dict] = []
            for index, model_item in enumerate(models):
                if isinstance(preprocessing_config, list):
                    if len(preprocessing_config) != n_targets:
                        raise ValueError("Multi-output artifact has inconsistent preprocessing metadata.")
                    component_preprocessing = preprocessing_config[index]
                else:
                    component_preprocessing = preprocessing_config
                X_component, applied = _apply_artifact_preprocessing(
                    X,
                    wv,
                    component_preprocessing,
                    input_preprocessed=input_preprocessed,
                )
                X_component = _apply_artifact_wavelength_selection(X_component, selections[index])
                predictions.append(np.asarray(model_item.predict(X_component), dtype=float).ravel())
                preprocessing_applied.append(applied)
                if detect_drift and len(monitoring_references) == n_targets and monitoring_references[index]:
                    from nir_core.utils.drift import compute_reference_drift

                    component_drift = compute_reference_drift(monitoring_references[index], X_component)
                    drift_results.append(
                        {
                            "name": names[index],
                            "drift_score": component_drift["drift_score"],
                            "flagged_indices": component_drift["flagged_indices"].tolist(),
                            "t2_limit": component_drift["t2_limit"],
                            "q_limit": component_drift["q_limit"],
                        }
                    )

            y_pred_multi = np.column_stack(predictions)
            result = {
                "status": "ok",
                "n_samples": int(X.shape[0]),
                "n_targets": n_targets,
                "component_names": names,
                "predictions_summary": [
                    {
                        "name": name,
                        "mean": float(np.mean(y_pred_multi[:, index])),
                        "min": float(np.min(y_pred_multi[:, index])),
                        "max": float(np.max(y_pred_multi[:, index])),
                    }
                    for index, name in enumerate(names)
                ],
                "preprocessing": {
                    "shared": (bool(preprocessing_config[0].get("shared", False)) if isinstance(preprocessing_config, list) and preprocessing_config else not isinstance(preprocessing_config, list)),
                    "applied": preprocessing_applied,
                },
                "wavelength_selection": selections,
            }
            if detect_drift:
                result["drift"] = (
                    {
                        "available": True,
                        "method": "pca_t2_q",
                        "per_component": drift_results,
                        "drift_score": max((item["drift_score"] for item in drift_results), default=0.0),
                    }
                    if drift_results
                    else {"available": False, "reason": "legacy_artifact_has_no_training_reference"}
                )
            if output_path:
                real_out = _resolve(runtime, output_path, read_only=False)
                os.makedirs(os.path.dirname(real_out), exist_ok=True)
                import csv

                with open(real_out, "w", newline="", encoding="utf-8") as file:
                    writer = csv.writer(file)
                    writer.writerow(["sample_index", *names])
                    for index, values in enumerate(y_pred_multi):
                        writer.writerow([index, *(float(value) for value in values)])
                result["output_path"] = output_path
            return finish(result)

        if isinstance(artifact, dict) and artifact.get("task_kind") == "classification":
            model = artifact.get("model")
            if model is None:
                raise ValueError("Classification artifact is missing the fitted model object.")
            preprocessing = artifact.get("preprocessing") or {}
            wavelength_selection = artifact.get("wavelength_selection") or {}
            monitoring_reference = artifact.get("monitoring_reference")
            X, preprocessing_applied = _apply_artifact_preprocessing(
                X,
                wv,
                preprocessing,
                input_preprocessed=input_preprocessed,
            )
            X = _apply_artifact_wavelength_selection(X, wavelength_selection)
            predicted = np.asarray(model.predict(X)).astype(str).ravel()
            classes = [str(value) for value in artifact.get("classes") or getattr(model, "classes_", [])]
            if not classes:
                raise ValueError("Classification artifact has no class vocabulary.")
            model_classes = [str(value) for value in getattr(model, "classes_", classes)]
            if model_classes != classes:
                raise ValueError("Classification artifact class vocabulary does not match the fitted model.")

            probabilities: np.ndarray | None = None
            predict_proba = getattr(model, "predict_proba", None)
            if callable(predict_proba):
                probabilities = np.asarray(predict_proba(X), dtype=float)
            if probabilities is None or probabilities.shape != (X.shape[0], len(classes)):
                raise ValueError("Classification model must provide one probability per persisted class.")
            confidence = np.max(probabilities, axis=1)
            if probabilities.shape[1] > 1:
                ordered = np.sort(probabilities, axis=1)
                margin = ordered[:, -1] - ordered[:, -2]
            else:
                margin = confidence.copy()

            drift_payload: dict | None = None
            drift_flagged: set[int] = set()
            if detect_drift:
                if monitoring_reference:
                    from nir_core.utils.drift import compute_reference_drift

                    drift = compute_reference_drift(monitoring_reference, X)
                    flagged_indices = drift["flagged_indices"].tolist()
                    drift_flagged = {int(index) for index in flagged_indices}
                    drift_payload = {
                        "available": True,
                        "method": drift["method"],
                        "drift_score": drift["drift_score"],
                        "flagged_indices": flagged_indices,
                        "t2_flagged_indices": drift["t2_flagged_indices"].tolist(),
                        "q_flagged_indices": drift["q_flagged_indices"].tolist(),
                        "t2_limit": drift["t2_limit"],
                        "q_limit": drift["q_limit"],
                    }
                else:
                    drift_payload = {"available": False, "reason": "legacy_artifact_has_no_training_reference"}

            policy = artifact.get("decision_policy") or {}
            confidence_threshold = float(policy.get("min_confidence", 0.5))
            margin_threshold = float(policy.get("min_margin", 0.05))
            accepted = np.asarray(
                [bool(confidence[index] >= confidence_threshold and margin[index] >= margin_threshold and index not in drift_flagged) for index in range(X.shape[0])],
                dtype=bool,
            )
            class_counts = {label: int(np.count_nonzero(predicted == label)) for label in classes if np.any(predicted == label)}
            result = {
                "status": "ok",
                "task_kind": "classification",
                "n_samples": int(X.shape[0]),
                "classes": classes,
                "class_counts": class_counts,
                "accepted_count": int(np.count_nonzero(accepted)),
                "needs_review_count": int(np.count_nonzero(~accepted)),
                "mean_confidence": float(np.mean(confidence)),
                "decision_policy": {
                    "min_confidence": confidence_threshold,
                    "min_margin": margin_threshold,
                    "drifted_samples_require_review": True,
                },
                "preprocessing": {
                    "description": preprocessing.get("description", "none"),
                    "applied": preprocessing_applied,
                },
                "wavelength_selection": wavelength_selection or None,
            }
            if drift_payload is not None:
                result["drift"] = drift_payload
            if output_path:
                real_out = _resolve(runtime, output_path, read_only=False)
                os.makedirs(os.path.dirname(real_out), exist_ok=True)
                import csv

                with open(real_out, "w", newline="", encoding="utf-8") as file:
                    writer = csv.writer(file)
                    writer.writerow(["sample_index", "predicted_class", "confidence", "margin", "decision"])
                    for index, label in enumerate(predicted):
                        writer.writerow(
                            [
                                index,
                                label,
                                float(confidence[index]),
                                float(margin[index]),
                                "accepted" if accepted[index] else "needs_review",
                            ]
                        )
                result["output_path"] = output_path
            return finish(result)

        model, preprocessing, wavelength_selection, monitoring_reference = _unwrap_model_artifact(artifact)
        if model is None:
            raise ValueError("Model artifact is missing the fitted model object.")
        X, preprocessing_applied = _apply_artifact_preprocessing(
            X,
            wv,
            preprocessing,
            input_preprocessed=input_preprocessed,
        )
        X = _apply_artifact_wavelength_selection(X, wavelength_selection)

        # Predict (sklearn-style model).
        y_pred = np.asarray(model.predict(X)).ravel()

        result = {
            "status": "ok",
            "n_samples": int(X.shape[0]),
            "prediction_mean": float(np.mean(y_pred)),
            "prediction_min": float(np.min(y_pred)),
            "prediction_max": float(np.max(y_pred)),
            "preprocessing": {
                "description": preprocessing.get("description", "none"),
                "applied": preprocessing_applied,
            },
            "wavelength_selection": wavelength_selection or None,
        }

        if detect_drift:
            if monitoring_reference:
                from nir_core.utils.drift import compute_reference_drift

                drift = compute_reference_drift(monitoring_reference, X)
                result["drift"] = {
                    "available": True,
                    "method": drift["method"],
                    "drift_score": drift["drift_score"],
                    "flagged_indices": drift["flagged_indices"].tolist(),
                    "t2_flagged_indices": drift["t2_flagged_indices"].tolist(),
                    "q_flagged_indices": drift["q_flagged_indices"].tolist(),
                    "t2_limit": drift["t2_limit"],
                    "q_limit": drift["q_limit"],
                }
            else:
                result["drift"] = {"available": False, "reason": "legacy_artifact_has_no_training_reference"}

        if output_path:
            real_out = _resolve(runtime, output_path, read_only=False)
            os.makedirs(os.path.dirname(real_out), exist_ok=True)
            import csv

            with open(real_out, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["sample_index", "prediction"])
                for i, v in enumerate(y_pred):
                    w.writerow([i, float(v)])
            result["output_path"] = output_path

        return finish(result)
    except Exception as exc:  # noqa: BLE001
        return finish(
            {"status": "error", "error": f"{type(exc).__name__}: {exc}"},
            exc,
        )
