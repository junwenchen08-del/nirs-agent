"""NIR data I/O tools: load / inspect / predict.

These tools bridge DeerFlow's virtual sandbox paths to nir_core's data
loaders and serialised-model inference. They never return spectral matrices
to the LLM — only JSON metadata and (optionally) saved .npz / .csv paths.
"""

from __future__ import annotations

import os
from typing import Annotated

import numpy as np
from langchain.tools import InjectedToolCallId, tool

from deerflow.tools.types import Runtime

from ._common import _err, _ok, _resolve


def _unwrap_model_artifact(artifact):
    """Return model and transform metadata for plain models and NIR artifacts."""
    if isinstance(artifact, dict) and artifact.get("format") == "nir_model_artifact":
        return (
            artifact.get("model"),
            artifact.get("preprocessing") or {},
            artifact.get("wavelength_selection") or {},
        )
    return artifact, {}, {}


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


@tool("nir_load_data", parse_docstring=True)
def nir_load_data_tool(
    runtime: Runtime,
    file_path: str,
    output_path: str | None = None,
    y_col: int | None = None,
    wv_row: int | None = None,
    x_cols: str | None = None,
    subset: str | None = None,
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
        wv_row: Optional 0-based index of the wavelength row (CSV/TXT only;
            ignored for .mat). When provided, auto-detection is bypassed and
            this row is split as wv. ``None`` (default) triggers
            auto-detection.
        x_cols: ★ v3.8 Optional column selector for the spectra block X
            (CSV/TXT only; ignored for .mat). Use this to skip metadata
            columns. Syntax: ``"8:"`` (col 8 to end), ``"8:314"`` (half-open
            range), ``":8"`` (cols 0-7), ``"8,9,10"`` (explicit list). When
            y_col is inside the selected range it is automatically excluded
            from X. Example for the Anderson mango dataset with 8 metadata
            cols + DM at col 8 + 306 spectra cols: pass
            ``y_col=8, wv_row=0, x_cols="9:"``.
        subset: Optional name of a sub-dataset inside a MATLAB struct .mat
            file (e.g. ``"R562"``). Call ``nir_inspect`` first to see the
            available subsets; when multiple exist and this is None, the
            tool returns an error listing them.

    Returns:
        JSON summary of the loaded data (sample count, wavelength range,
        reference value range, source format). Spectral matrices are NOT
        returned — only metadata.
    """
    try:
        from nir_core.io.loaders import auto_detect_and_load, load_csv, load_mat
        from nir_core.io.sniffers import detect_format
        from nir_core.io.writers import save_npz

        real_in = _resolve(runtime, file_path, read_only=True)

        # When explicit y_col / wv_row / x_cols are provided for CSV/TXT files,
        # bypass auto-detection and call load_csv with the override.
        # This gives the agent an escape hatch when auto-detection fails,
        # so it never needs to fall back to writing Python scripts.
        has_override = y_col is not None or wv_row is not None or x_cols is not None
        if has_override and detect_format(real_in) != "mat":
            data = load_csv(real_in, y_col=y_col, wv_row=wv_row, x_cols=x_cols)
        elif subset is not None and detect_format(real_in) == "mat":
            # Struct .mat with explicit subset selection.
            data = load_mat(real_in, subset=subset)
        else:
            data = auto_detect_and_load(real_in)

        if output_path:
            real_out = _resolve(runtime, output_path, read_only=False)
            os.makedirs(os.path.dirname(real_out), exist_ok=True)
            save_npz(data, real_out)
        summary = data.summary()
        summary["output_saved"] = bool(output_path)
        summary["layout_override_used"] = has_override
        summary["subset_used"] = subset
        summary["x_cols_used"] = x_cols
        # ⭐ Surface the auto-detected layout so the agent can confirm y and
        # wv have been correctly separated from the raw CSV block. Without
        # these explicit fields the agent tends to "double-check" by writing
        # its own Python script — which is exactly the anti-pattern this
        # tool exists to prevent.
        summary["y_separated"] = data.y is not None
        summary["wv_separated"] = data.wv is not None
        if data.y is not None:
            summary["y_first_values"] = [round(float(v), 4) for v in data.y[:5]]
            summary["y_last_values"] = [round(float(v), 4) for v in data.y[-3:]]
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

    Returns format, shape, estimated sample/wavelength counts, value range
    and NaN presence as a JSON report. Useful for previewing data before
    committing to a full load.

    Args:
        file_path: Virtual path to the file to inspect.
    """
    try:
        from nir_core.io.sniffers import inspect_file

        real_in = _resolve(runtime, file_path, read_only=True)
        raw = inspect_file(real_in)
        # inspect_file returns a JSON string. Parse it so we can layer extra
        # advisory fields and re-serialize.
        import json as _json

        try:
            info = _json.loads(raw)
        except (TypeError, ValueError):
            return raw

        # Add layout advisory so the agent knows whether the NIR-style
        # "row-label + column-label" pattern is present without having to
        # open the file in Python.
        corner_nan = bool(info.get("corner_is_nan"))
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
        if info.get("is_struct"):
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
        return _json.dumps(info, ensure_ascii=False)
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
    wavelength selection metadata to raw spectra before inference.
    Optionally runs Mahalanobis drift detection against the training
    distribution (if the metrics JSON contains training stats).

    Args:
        model_path: Virtual path to the .pkl model file.
        data_path: Virtual path to a new-data .npz containing X and optional wv.
        output_path: Optional virtual path for a CSV of predictions.
        detect_drift: If True, compute Mahalanobis drift of the new spectra
            against the input data's own distribution (as a proxy when no
            training reference is available).
        input_preprocessed: Set True only when X already has the artifact's
            train-time preprocessing applied.

    Returns:
        JSON with predictions summary (count, mean, min, max) and optional
        drift info. Individual predictions are saved to output_path if given.
    """
    try:
        import joblib

        real_model = _resolve(runtime, model_path, read_only=True)
        real_data = _resolve(runtime, data_path, read_only=True)
        model, preprocessing, wavelength_selection = _unwrap_model_artifact(joblib.load(real_model))
        if model is None:
            return _err("Model artifact is missing the fitted model object.")
        data_dict = dict(np.load(real_data, allow_pickle=True))
        X = np.asarray(data_dict["X"], dtype=float)
        wv = np.asarray(data_dict["wv"], dtype=float).ravel() if data_dict.get("wv") is not None and data_dict["wv"].size else None
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
            from nir_core.utils.drift import compute_mahalanobis_drift

            # Use the input data itself as the reference distribution
            # (best available proxy when no separate training matrix exists).
            drift = compute_mahalanobis_drift(X, X, threshold=3.0)
            result["drift"] = {
                "note": "Self-reference drift (no training matrix supplied); supply training X for a true drift estimate.",
                "drift_score": float(drift["drift_score"]),
            }

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

        return _ok(result)
    except Exception as exc:  # noqa: BLE001
        return _err(f"{type(exc).__name__}: {exc}")
