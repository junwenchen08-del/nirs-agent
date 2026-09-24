"""Supervised qualitative NIR classification training tool."""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Annotated

import numpy as np
from langchain.tools import InjectedToolCallId, tool
from nir_core.io.resources import ResourceLimitError

from deerflow.tools.types import Runtime

from ._common import (
    _bind_model_metrics,
    _err,
    _json_default,
    _model_evidence,
    _ok,
    _parse_pipeline_step,
    _resolve,
    _sha256_file,
    _write_trusted_model_artifact,
)
from ._resources import budget_for_runtime, resource_error
from ._science_gate import reproducibility_evidence, science_gate_error, split_science_gate
from .artifacts import _build_preprocessing_artifact


def _resolve_column(frame, selector: str, *, role: str) -> tuple[int, str]:
    normalized = str(selector).strip()
    if normalized in frame.columns:
        index = int(frame.columns.get_loc(normalized))
        return index, str(frame.columns[index])
    try:
        index = int(normalized)
    except ValueError as exc:
        raise ValueError(f"{role} column {selector!r} was not found") from exc
    if not 0 <= index < frame.shape[1]:
        raise ValueError(f"{role} column index {index} is outside 0..{frame.shape[1] - 1}")
    return index, str(frame.columns[index])


def _feature_indices(frame, *, label_index: int, group_index: int | None, x_cols: str | None) -> list[int]:
    from nir_core.io.loaders import _parse_x_cols

    excluded = {label_index, *([group_index] if group_index is not None else [])}
    if x_cols is not None and str(x_cols).strip():
        selected = [index for index in _parse_x_cols(str(x_cols), frame.shape[1]) if index not in excluded]
    else:
        selected = []
        for index, column in enumerate(frame.columns):
            if index in excluded:
                continue
            numeric = frame[column]
            try:
                numeric = numeric.astype(float)
            except (TypeError, ValueError):
                continue
            if np.isfinite(numeric.to_numpy(dtype=float)).all():
                selected.append(index)
    if not selected:
        raise ValueError("No numeric spectral columns were selected; provide x_cols explicitly")
    return selected


def _wavelengths_from_headers(headers: list[str]) -> np.ndarray | None:
    try:
        wavelengths = np.asarray([float(value) for value in headers], dtype=float)
    except ValueError:
        return None
    return wavelengths if np.isfinite(wavelengths).all() else None


def _parse_index_selector(value: str | None, n_columns: int) -> list[int] | None:
    if value is None or not str(value).strip():
        return None
    from nir_core.io.loaders import _parse_x_cols

    return _parse_x_cols(str(value), n_columns)


def _string_values(values) -> np.ndarray:
    import pandas as pd

    series = pd.Series(values, dtype="string")
    if series.isna().any() or (series.str.strip() == "").any():
        raise ValueError("Classification labels contain missing or empty values")
    return series.astype(str).str.strip().to_numpy()


def _numeric_share(values) -> float:
    total = 0
    numeric = 0
    for value in values:
        text = str(value).strip()
        if not text or text.lower() == "nan":
            continue
        total += 1
        try:
            float(text)
        except ValueError:
            continue
        numeric += 1
    return numeric / total if total else 0.0


def _find_columnwise_feature_rows(raw, *, start_row: int, sample_indices: list[int], wavenumber_col: int | None) -> list[int]:
    rows: list[int] = []
    for row_index in range(start_row, raw.shape[0]):
        if wavenumber_col is not None:
            try:
                float(str(raw.iat[row_index, wavenumber_col]).strip())
            except ValueError:
                continue
        numeric = raw.iloc[row_index, sample_indices].apply(lambda value: str(value).strip()).replace("", np.nan)
        converted = numeric.apply(lambda value: float(value) if str(value).strip().lower() != "nan" else np.nan)
        if converted.isna().any():
            continue
        rows.append(row_index)
    if not rows:
        raise ValueError("No numeric spectral rows were found for the columnwise classification layout")
    return rows


def _label_row_candidates(raw, *, sample_indices: list[int], min_count_per_class: int = 5) -> list[dict]:
    candidates: list[dict] = []
    search_rows = min(raw.shape[0], 25)
    for row_index in range(search_rows):
        labels = _string_values(raw.iloc[row_index, sample_indices])
        counts = Counter(labels)
        n_unique = len(counts)
        if n_unique < 2 or n_unique > 20 or min(counts.values(), default=0) < min_count_per_class:
            continue
        data_start = row_index + 1
        try:
            feature_rows = _find_columnwise_feature_rows(
                raw,
                start_row=data_start,
                sample_indices=sample_indices,
                wavenumber_col=0,
            )
        except Exception:  # noqa: BLE001 - this is an advisory detector
            continue
        nonnumeric_score = 1.0 - _numeric_share(labels)
        candidates.append(
            {
                "row_index": row_index,
                "n_unique": n_unique,
                "class_counts": dict(sorted(counts.items())),
                "data_start_row": min(feature_rows),
                "n_feature_rows": len(feature_rows),
                "score": (nonnumeric_score, len(feature_rows), -row_index),
            }
        )
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates


def _classification_arrays_from_rowwise(
    frame,
    *,
    label_col: str,
    group_col: str | None,
    x_cols: str | None,
) -> dict:
    label_index, label_name = _resolve_column(frame, label_col, role="label")
    group_index: int | None = None
    group_name: str | None = None
    if group_col is not None and str(group_col).strip():
        group_index, group_name = _resolve_column(frame, group_col, role="group")
        if group_index == label_index:
            raise ValueError("label_col and group_col must be different columns")
    selected_indices = _feature_indices(
        frame,
        label_index=label_index,
        group_index=group_index,
        x_cols=x_cols,
    )
    headers = [str(frame.columns[index]) for index in selected_indices]
    X = frame.iloc[:, selected_indices].apply(lambda column: column.astype(float)).to_numpy(dtype=float)
    y = _string_values(frame.iloc[:, label_index])
    groups = None
    if group_index is not None:
        groups = _string_values(frame.iloc[:, group_index])
    return {
        "layout": "samples_in_rows",
        "X": X,
        "y": y,
        "groups": groups,
        "headers": headers,
        "wv": _wavelengths_from_headers(headers),
        "label_name": label_name,
        "group_name": group_name,
        "layout_parameters": {
            "label_col": label_name,
            "group_col": group_name,
            "x_cols": x_cols,
        },
    }


def _classification_arrays_from_columnwise(
    path: str,
    *,
    label_row: int | None,
    group_row: int | None,
    sample_cols: str | None,
    wavenumber_col: int | None,
) -> dict:
    import pandas as pd

    raw = pd.read_csv(path, header=None)
    if raw.shape[1] < 3:
        raise ValueError("Columnwise classification layout requires one wavelength column and at least two sample columns")
    selected_sample_indices = _parse_index_selector(sample_cols, raw.shape[1])
    if selected_sample_indices is None:
        selected_sample_indices = [index for index in range(raw.shape[1]) if index != wavenumber_col]
    selected_sample_indices = [index for index in selected_sample_indices if index != wavenumber_col]
    if len(selected_sample_indices) < 2:
        raise ValueError("Columnwise classification layout requires at least two sample columns")

    inferred = False
    if label_row is None:
        candidates = _label_row_candidates(raw, sample_indices=selected_sample_indices)
        if not candidates:
            raise ValueError("Could not infer a class-label row; pass label_row explicitly")
        label_row = int(candidates[0]["row_index"])
        inferred = True
    if not 0 <= int(label_row) < raw.shape[0]:
        raise ValueError(f"label_row index {label_row} is outside 0..{raw.shape[0] - 1}")
    if group_row is not None and not 0 <= int(group_row) < raw.shape[0]:
        raise ValueError(f"group_row index {group_row} is outside 0..{raw.shape[0] - 1}")
    if group_row is not None and int(group_row) == int(label_row):
        raise ValueError("label_row and group_row must be different rows")

    y = _string_values(raw.iloc[int(label_row), selected_sample_indices])
    groups = _string_values(raw.iloc[int(group_row), selected_sample_indices]) if group_row is not None else None
    feature_start = max(int(label_row), int(group_row) if group_row is not None else -1) + 1
    feature_rows = _find_columnwise_feature_rows(
        raw,
        start_row=feature_start,
        sample_indices=selected_sample_indices,
        wavenumber_col=wavenumber_col,
    )
    X = raw.iloc[feature_rows, selected_sample_indices].apply(lambda column: column.astype(float)).to_numpy(dtype=float).T
    if wavenumber_col is None:
        headers = [f"feature_{index}" for index in feature_rows]
        wv = None
    else:
        headers = [str(raw.iat[row_index, wavenumber_col]).strip() for row_index in feature_rows]
        wv = _wavelengths_from_headers(headers)
    return {
        "layout": "samples_in_columns",
        "X": X,
        "y": y,
        "groups": groups,
        "headers": headers,
        "wv": wv,
        "label_name": f"row_{int(label_row)}",
        "group_name": f"row_{int(group_row)}" if group_row is not None else None,
        "layout_parameters": {
            "label_row": int(label_row),
            "label_row_inferred": inferred,
            "group_row": int(group_row) if group_row is not None else None,
            "sample_cols": sample_cols or f"0:{raw.shape[1]} excluding wavenumber_col",
            "wavenumber_col": wavenumber_col,
            "data_start_row": min(feature_rows),
        },
    }


def _classification_dataset_gate(X: np.ndarray, labels: np.ndarray, wv: np.ndarray | None) -> dict:
    errors: list[dict] = []
    if X.ndim != 2 or X.shape[0] != labels.size:
        errors.append({"code": "shape_mismatch", "message": "X must be 2D with one class label per row"})
    if not np.isfinite(X).all():
        errors.append({"code": "non_finite_spectra", "message": "Spectra contain NaN or infinite values"})
    classes, counts = np.unique(labels, return_counts=True)
    if classes.size < 2:
        errors.append({"code": "single_class", "message": "Classification requires at least two classes"})
    if counts.size and int(counts.min()) < 5:
        errors.append({"code": "class_too_small", "message": "Every class needs at least 5 samples for stable calibration, tuning, and holdout selection"})
    seen: dict[bytes, str] = {}
    conflicts = 0
    for row, label in zip(np.ascontiguousarray(X), labels, strict=True):
        key = row.tobytes()
        previous = seen.setdefault(key, str(label))
        if previous != str(label):
            conflicts += 1
    if conflicts:
        errors.append({"code": "conflicting_duplicate_spectra", "message": f"Found {conflicts} duplicate spectra with conflicting labels"})
    if wv is not None:
        if wv.size != X.shape[1] or not np.isfinite(wv).all():
            errors.append({"code": "invalid_wavelength_axis", "message": "Wavelength axis is non-finite or does not match X columns"})
        elif np.any(np.diff(wv) == 0):
            errors.append({"code": "duplicate_wavelengths", "message": "Wavelength axis contains duplicate values"})
    return {
        "passed": not errors,
        "errors": errors,
        "n_samples": int(X.shape[0]) if X.ndim == 2 else 0,
        "n_wavelengths": int(X.shape[1]) if X.ndim == 2 else 0,
        "n_classes": int(classes.size),
        "class_distribution": {str(label): int(count) for label, count in zip(classes, counts, strict=True)},
    }


def _model_probabilities(model, X: np.ndarray) -> np.ndarray | None:
    predict_proba = getattr(model, "predict_proba", None)
    if not callable(predict_proba):
        return None
    probabilities = np.asarray(predict_proba(X), dtype=float)
    return probabilities if probabilities.ndim == 2 else None


def _quality_gate(
    metrics: dict,
    *,
    min_balanced_accuracy: float,
    min_macro_f1: float,
    min_class_recall: float,
) -> dict:
    per_class = metrics.get("per_class") or {}
    minimum_recall = min((float(value.get("recall", 0.0)) for value in per_class.values()), default=0.0)
    checks = {
        "balanced_accuracy": float(metrics.get("balanced_accuracy", 0.0)) >= float(min_balanced_accuracy),
        "macro_f1": float(metrics.get("macro_f1", 0.0)) >= float(min_macro_f1),
        "minimum_class_recall": minimum_recall >= float(min_class_recall),
    }
    passed = all(checks.values())
    score = min(float(metrics.get("balanced_accuracy", 0.0)), float(metrics.get("macro_f1", 0.0)), minimum_recall)
    grade = "A" if passed and score >= 0.9 else ("B" if passed else ("C" if score >= 0.5 else "F"))
    return {
        "passed": passed,
        "grade": grade,
        "action": "eligible_for_review" if passed else "improve_or_collect_more_data",
        "minimum_class_recall": minimum_recall,
        "checks": checks,
        "thresholds_used": {
            "min_balanced_accuracy": float(min_balanced_accuracy),
            "min_macro_f1": float(min_macro_f1),
            "min_class_recall": float(min_class_recall),
        },
    }


@tool("nir_train_classifier", parse_docstring=True)
def nir_train_classifier_tool(
    runtime: Runtime,
    file_path: str,
    label_col: str | None = None,
    x_cols: str | None = None,
    group_col: str | None = None,
    label_row: int | None = None,
    group_row: int | None = None,
    sample_cols: str | None = None,
    wavenumber_col: int | None = 0,
    pipeline_steps: str = '["snv", "autoscale"]',
    methods: str = '["pls_da", "logistic", "linear_svm"]',
    tuning_ratio: float = 0.2,
    test_ratio: float = 0.2,
    random_state: int = 42,
    max_components: int = 10,
    min_balanced_accuracy: float = 0.7,
    min_macro_f1: float = 0.7,
    min_class_recall: float = 0.6,
    min_confidence: float = 0.5,
    min_margin: float = 0.05,
    model_output: str = "/mnt/user-data/outputs/qualitative_model.pkl",
    metrics_output: str = "/mnt/user-data/outputs/qualitative_metrics.json",
    report_output: str = "/mnt/user-data/outputs/qualitative_report.md",
    tool_call_id: Annotated[str, InjectedToolCallId] = "",  # noqa: ARG001
) -> str:
    """Train a leakage-safe supervised NIR classifier from a labelled CSV.

    The tool accepts string, integer, or boolean class labels, performs a
    deterministic class-stratified three-way split, fits preprocessing on
    calibration rows only, selects a bounded model family on tuning rows, and
    evaluates the selected family once on the holdout partition. When
    ``group_col`` is supplied, no group may cross partitions.
    It also supports common public spectroscopy CSV files where samples are
    columns and wavelengths are rows: pass ``label_row`` (or omit both
    ``label_col`` and ``label_row`` to infer a bounded categorical label row),
    ``sample_cols="1:"``, and optional ``group_row``.

    Args:
        file_path: Virtual path to a labelled CSV table.
        label_col: Class-label column name or zero-based column index for
            samples-in-rows CSV files. May be omitted when using or inferring
            ``label_row`` for samples-in-columns CSV files.
        x_cols: Optional rowwise spectral-column selector such as ``"2:"``.
            When omitted, finite numeric columns other than label/group are
            used.
        group_col: Optional sample, batch, origin, or replicate group column.
        label_row: Optional zero-based class-label row for samples-in-columns
            CSV files, e.g. ``2`` for BBSRC-style FTIR coffee data.
        group_row: Optional zero-based sample/batch/group row for
            samples-in-columns CSV files.
        sample_cols: Optional columnwise sample-column selector such as
            ``"1:"``. Defaults to all columns except ``wavenumber_col``.
        wavenumber_col: Optional zero-based wavelength/wavenumber column for
            samples-in-columns CSV files; pass null when no wavelength column
            exists.
        pipeline_steps: JSON preprocessing list fitted inside data partitions.
        methods: JSON list from ``pls_da``, ``logistic``, ``linear_svm``, and
            ``rbf_svm``.
        tuning_ratio: Fraction reserved for model selection.
        test_ratio: Fraction reserved for one-time holdout evaluation.
        random_state: Reproducible split and estimator seed.
        max_components: Maximum PLS-DA latent-variable count.
        min_balanced_accuracy: Holdout quality threshold.
        min_macro_f1: Holdout macro-F1 quality threshold.
        min_class_recall: Minimum allowed holdout recall for every class.
        min_confidence: Prediction acceptance confidence threshold.
        min_margin: Prediction acceptance top-two probability margin.
        model_output: Virtual output path for the version-4 model artifact.
        metrics_output: Virtual output path for metrics and split evidence.
        report_output: Virtual output path for the Markdown report.

    Returns:
        JSON with selected method, class distribution, holdout metrics, quality
        decision, cryptographic evidence, and generated artifact paths.
    """
    try:
        probability_thresholds = {
            "min_balanced_accuracy": min_balanced_accuracy,
            "min_macro_f1": min_macro_f1,
            "min_class_recall": min_class_recall,
            "min_confidence": min_confidence,
            "min_margin": min_margin,
        }
        invalid_thresholds = [name for name, value in probability_thresholds.items() if not np.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0]
        if invalid_thresholds:
            return _err("Classification thresholds must be finite values in [0, 1]: " + ", ".join(invalid_thresholds))
        if not 0.0 < float(tuning_ratio) < 1.0 or not 0.0 < float(test_ratio) < 1.0 or float(tuning_ratio) + float(test_ratio) >= 1.0:
            return _err("tuning_ratio and test_ratio must be positive and sum to less than 1")
        if int(max_components) < 1:
            return _err("max_components must be at least 1")

        import pandas as pd
        from nir_core.model.classification import (
            build_classifier,
            classification_metrics,
            select_classifier,
            stratified_three_way_split,
        )
        from nir_core.preprocess.pipeline import PreprocessingPipeline, validate_pipeline
        from nir_core.utils.drift import fit_monitoring_reference

        budget = budget_for_runtime(runtime)
        real_input = _resolve(runtime, file_path, read_only=True)
        budget.check_file(real_input, stage="classification_file_preflight")

        use_columnwise = label_row is not None or (label_col is None or not str(label_col).strip())
        if use_columnwise:
            arrays = _classification_arrays_from_columnwise(
                real_input,
                label_row=label_row,
                group_row=group_row,
                sample_cols=sample_cols,
                wavenumber_col=wavenumber_col,
            )
        else:
            frame = pd.read_csv(real_input)
            arrays = _classification_arrays_from_rowwise(
                frame,
                label_col=str(label_col),
                group_col=group_col,
                x_cols=x_cols,
            )
        X = arrays["X"]
        y = arrays["y"]
        groups = arrays["groups"]
        headers = arrays["headers"]
        wv = arrays["wv"]
        label_name = arrays["label_name"]
        group_name = arrays["group_name"]
        input_layout = arrays["layout"]
        layout_parameters = arrays["layout_parameters"]
        budget.check_array(X, target_count=1, peak_multiplier=8.0, stage="classification_matrix_preflight")

        dataset_validation = _classification_dataset_gate(X, y, wv)
        if validation_error := science_gate_error(dataset_validation):
            return _err(validation_error)
        split = stratified_three_way_split(
            y,
            groups=groups,
            tuning_ratio=float(tuning_ratio),
            test_ratio=float(test_ratio),
            random_state=int(random_state),
        )
        calibration = np.asarray(split.calibration, dtype=int)
        tuning = np.asarray(split.tuning, dtype=int)
        holdout = np.asarray(split.holdout, dtype=int)
        split_validation = split_science_gate(
            calibration=X[calibration],
            tuning=X[tuning],
            holdout=X[holdout],
        )
        if validation_error := science_gate_error(split_validation):
            return _err(validation_error)

        raw_steps = json.loads(pipeline_steps)
        if not isinstance(raw_steps, list):
            return _err("pipeline_steps must be a JSON list")
        steps = [_parse_pipeline_step(value) for value in raw_steps]
        valid, reason = validate_pipeline(steps)
        if not valid:
            return _err(f"Invalid preprocessing pipeline: {reason}")
        method_values = json.loads(methods)
        if not isinstance(method_values, list) or not method_values:
            return _err("methods must be a non-empty JSON list")
        method_values = [str(value).strip().lower() for value in method_values]

        selection_pipeline = PreprocessingPipeline(steps).fit(X[calibration], wv)
        X_calibration = selection_pipeline.transform(X[calibration], wv)
        X_tuning = selection_pipeline.transform(X[tuning], wv)
        selection = select_classifier(
            X_calibration,
            y[calibration],
            X_tuning,
            y[tuning],
            methods=method_values,
            random_state=int(random_state),
            max_components=int(max_components),
        )

        final_indices = np.sort(np.concatenate([calibration, tuning]))
        final_pipeline = PreprocessingPipeline(steps).fit(X[final_indices], wv)
        X_final = final_pipeline.transform(X[final_indices], wv)
        X_holdout = final_pipeline.transform(X[holdout], wv)
        chosen = next(item for item in selection.candidates if item["method"] == selection.method)
        final_model = build_classifier(
            selection.method,
            random_state=int(random_state),
            max_components=int(max_components),
            n_components=int(chosen["n_components"]) if chosen.get("n_components") is not None else None,
        ).fit(X_final, y[final_indices])
        classes = [str(value) for value in final_model.classes_]

        train_prediction = np.asarray(final_model.predict(X_final)).astype(str)
        holdout_prediction = np.asarray(final_model.predict(X_holdout)).astype(str)
        train_probabilities = _model_probabilities(final_model, X_final)
        holdout_probabilities = _model_probabilities(final_model, X_holdout)
        train_metrics = classification_metrics(
            y[final_indices],
            train_prediction,
            classes=classes,
            probabilities=train_probabilities,
        )
        test_metrics = classification_metrics(
            y[holdout],
            holdout_prediction,
            classes=classes,
            probabilities=holdout_probabilities,
        )
        quality = _quality_gate(
            test_metrics,
            min_balanced_accuracy=float(min_balanced_accuracy),
            min_macro_f1=float(min_macro_f1),
            min_class_recall=float(min_class_recall),
        )

        training_data_hash = _sha256_file(real_input)
        split_payload = {
            "strategy": split.strategy,
            "random_state": int(random_state),
            "group_column": group_name,
            "ratios": {
                "calibration": float(1.0 - tuning_ratio - test_ratio),
                "tuning": float(tuning_ratio),
                "holdout": float(test_ratio),
            },
        }
        partition_payload = {
            "calibration": {"n_samples": int(calibration.size), "sample_indices": calibration.tolist()},
            "tuning": {"n_samples": int(tuning.size), "sample_indices": tuning.tolist()},
            "holdout": {"n_samples": int(holdout.size), "sample_indices": holdout.tolist()},
        }
        if groups is not None:
            for name, indices in (("calibration", calibration), ("tuning", tuning), ("holdout", holdout)):
                partition_payload[name]["group_values"] = sorted(set(groups[indices]))
        metrics = {
            "task_kind": "classification",
            "training_data_hash": training_data_hash,
            "scientific_validation": {
                "schema_version": 1,
                "passed": True,
                "dataset": dataset_validation,
                "partition_separation": split_validation,
            },
            "reproducibility": reproducibility_evidence(
                random_state=int(random_state),
                protocol="classification_three_way_holdout",
                input_sha256=training_data_hash,
                parameters={
                    "input_layout": input_layout,
                    "label_col": label_name if input_layout == "samples_in_rows" else None,
                    "group_col": group_name if input_layout == "samples_in_rows" else None,
                    "label_row": layout_parameters.get("label_row"),
                    "label_row_inferred": layout_parameters.get("label_row_inferred"),
                    "group_row": layout_parameters.get("group_row"),
                    "x_cols": x_cols,
                    "sample_cols": sample_cols,
                    "wavenumber_col": wavenumber_col,
                    "pipeline_steps": raw_steps,
                    "methods": method_values,
                    "tuning_ratio": float(tuning_ratio),
                    "test_ratio": float(test_ratio),
                    "max_components": int(max_components),
                },
            ),
            "protocol": "classification_three_way_holdout",
            "validation_scope": "independent_holdout_not_external",
            "label_name": label_name,
            "input_layout": input_layout,
            "layout_parameters": layout_parameters,
            "classes": classes,
            "class_distribution": dict(sorted(Counter(y).items())),
            "method": selection.method,
            "n_samples": int(X.shape[0]),
            "n_wavelengths": int(X.shape[1]),
            "preprocessing": final_pipeline.description(),
            "preprocessing_steps": raw_steps,
            "wavelength_headers": headers,
            "split": split_payload,
            "partitions": partition_payload,
            "candidate_results": selection.candidates,
            "train": train_metrics,
            "test": test_metrics,
            "quality": quality,
        }

        monitoring_reference = fit_monitoring_reference(X_final) if X_final.shape[0] >= 3 else None
        artifact = {
            "format": "nir_model_artifact",
            "version": 4,
            "task_kind": "classification",
            "model": final_model,
            "method": selection.method,
            "classes": classes,
            "label_name": label_name,
            "preprocessing": _build_preprocessing_artifact(
                final_pipeline,
                final_pipeline.description(),
            ),
            "wavelength_selection": {
                "method": "none",
                "n_original": int(X.shape[1]),
                "n_selected": int(X.shape[1]),
                "selected_indices": None,
            },
            "decision_policy": {
                "min_confidence": float(min_confidence),
                "min_margin": float(min_margin),
                "unknown_action": "needs_review",
            },
            "monitoring_reference": monitoring_reference,
        }
        real_model = _resolve(runtime, model_output, read_only=False)
        real_metrics = _resolve(runtime, metrics_output, read_only=False)
        real_report = _resolve(runtime, report_output, read_only=False)
        for path in (real_model, real_metrics, real_report):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        _write_trusted_model_artifact(artifact, real_model)
        with open(real_metrics, "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, ensure_ascii=False, indent=2, default=_json_default)
        _bind_model_metrics(real_model, real_metrics, training_data_hash=training_data_hash)
        report = [
            "# Qualitative NIR classification report",
            "",
            f"- Label: `{label_name}`",
            f"- Input layout: `{input_layout}`",
            f"- Classes: {', '.join(classes)}",
            f"- Samples: {X.shape[0]}",
            f"- Split: `{split.strategy}` ({calibration.size}/{tuning.size}/{holdout.size})",
            "- Validation scope: independent holdout generated from this dataset; not external validation",
            f"- Preprocessing: {final_pipeline.description()}",
            f"- Selected model: `{selection.method}`",
            f"- Holdout balanced accuracy: {test_metrics['balanced_accuracy']:.5f}",
            f"- Holdout macro-F1: {test_metrics['macro_f1']:.5f}",
            f"- Holdout MCC: {test_metrics['mcc']:.5f}",
            f"- Minimum class recall: {quality['minimum_class_recall']:.5f}",
            f"- Quality grade: `{quality['grade']}`",
            "",
            "## Confusion matrix",
            "",
            "```json",
            json.dumps(test_metrics["confusion_matrix"], ensure_ascii=False),
            "```",
        ]
        with open(real_report, "w", encoding="utf-8") as handle:
            handle.write("\n".join(report) + "\n")

        return _ok(
            {
                "status": "ok",
                "task_kind": "classification",
                "protocol": metrics["protocol"],
                "validation_scope": metrics["validation_scope"],
                "label_name": label_name,
                "input_layout": input_layout,
                "layout_parameters": layout_parameters,
                "classes": classes,
                "class_distribution": metrics["class_distribution"],
                "split": split_payload,
                "partitions": {name: {"n_samples": value["n_samples"]} for name, value in partition_payload.items()},
                "preprocessing": final_pipeline.description(),
                "method": selection.method,
                "candidate_results": selection.candidates,
                "holdout": test_metrics,
                "grade": quality["grade"],
                "passed": quality["passed"],
                "action": quality["action"],
                "thresholds_used": quality["thresholds_used"],
                "model_path": model_output,
                "metrics_path": metrics_output,
                "report": report_output,
                "evidence": _model_evidence(
                    real_model,
                    real_metrics,
                    training_data_hash=training_data_hash,
                    protocol=metrics["protocol"],
                    validation_scope=metrics["validation_scope"],
                ),
                "resource_budget": budget.evidence(),
            }
        )
    except ResourceLimitError as exc:
        return resource_error(exc)
    except Exception as exc:  # noqa: BLE001
        return _err(f"{type(exc).__name__}: {exc}")


__all__ = ["nir_train_classifier_tool"]
