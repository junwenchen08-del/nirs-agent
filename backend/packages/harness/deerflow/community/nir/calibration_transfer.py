"""Separate NIR calibration-transfer tools backed by Chemotools adaptation."""

from __future__ import annotations

import json
import os
from pathlib import PurePosixPath
from typing import Annotated, Any

import numpy as np
from langchain.tools import InjectedToolCallId, tool

from deerflow.tools.types import Runtime

from ._common import (
    _err,
    _load_npz_safely,
    _load_trusted_model_artifact,
    _load_trusted_transfer_artifact,
    _ok,
    _resolve,
    _sha256_file,
    _write_trusted_transfer_artifact,
)


def _required_array(data: dict[str, np.ndarray], name: str, *, path_label: str):
    value = data.get(name)
    if value is None or np.asarray(value).size == 0:
        raise ValueError(f"{path_label} must contain a non-empty {name!r} array.")
    return np.asarray(value)


def _pair_loaded_data(
    source_data: dict[str, np.ndarray],
    target_data: dict[str, np.ndarray],
    *,
    pairing_mode: str,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    from nir_core.calibration_transfer import pair_transfer_samples

    source = np.asarray(_required_array(source_data, "X", path_label="source"), dtype=float)
    target = np.asarray(_required_array(target_data, "X", path_label="target"), dtype=float)
    normalized_mode = str(pairing_mode).strip().lower()
    if normalized_mode == "sample_names":
        source_ids = _required_array(
            source_data,
            "sample_names",
            path_label="source",
        ).astype(str)
        target_ids = _required_array(
            target_data,
            "sample_names",
            path_label="target",
        ).astype(str)
        return pair_transfer_samples(
            source,
            target,
            source_ids=source_ids.tolist(),
            target_ids=target_ids.tolist(),
        )
    if normalized_mode == "row_order":
        if source.shape[0] != target.shape[0]:
            raise ValueError("row_order pairing requires equal source and target row counts.")
        ids = tuple(f"row-{index}" for index in range(source.shape[0]))
        return source.copy(), target.copy(), ids
    raise ValueError("pairing_mode must be 'sample_names' or explicit 'row_order'.")


def _source_and_target_axes(
    source_data: dict[str, np.ndarray],
    target_data: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    source_wv = np.asarray(
        _required_array(source_data, "wv", path_label="source"),
        dtype=float,
    ).ravel()
    target_wv = np.asarray(
        _required_array(target_data, "wv", path_label="target"),
        dtype=float,
    ).ravel()
    return source_wv, target_wv


def _single_target_y(
    data: dict[str, np.ndarray],
    *,
    expected_rows: int,
    path_label: str,
) -> np.ndarray:
    y = np.asarray(
        _required_array(data, "y", path_label=path_label),
        dtype=float,
    )
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    if y.ndim != 1 or y.shape[0] != expected_rows or not np.isfinite(y).all():
        raise ValueError(f"{path_label} y must be a finite single-target vector aligned with the paired samples.")
    return y


def _row_fingerprints(values: np.ndarray) -> set[bytes]:
    matrix = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return {matrix[index].tobytes() for index in range(matrix.shape[0])}


def _aligned_target(transfer, X_target: np.ndarray, target_wv: np.ndarray) -> np.ndarray:
    aligner = transfer.axis_aligner_
    return aligner.transform(X_target, target_wv) if aligner is not None else np.asarray(X_target, dtype=float)


def _parse_parameters(parameters_json: str | None) -> dict[str, Any]:
    if parameters_json is None or not str(parameters_json).strip():
        return {}
    parsed = json.loads(parameters_json)
    if not isinstance(parsed, dict):
        raise ValueError("parameters_json must decode to a JSON object.")
    return parsed


def _require_output_artifact_path(path: str) -> None:
    virtual = PurePosixPath(str(path).replace("\\", "/"))
    if virtual.parts[:4] != ("/", "mnt", "user-data", "outputs"):
        raise ValueError("Calibration-transfer artifacts must be written under /mnt/user-data/outputs.")


def _predict_reference_model(
    artifact,
    X: np.ndarray,
    source_wv: np.ndarray,
) -> np.ndarray:
    from .io_tools import (
        _apply_artifact_preprocessing,
        _apply_artifact_wavelength_selection,
        _unwrap_model_artifact,
    )

    if isinstance(artifact, dict) and (artifact.get("multi_output") is True or artifact.get("task_kind") == "classification"):
        raise ValueError("Calibration-transfer prediction validation currently requires a single-target regression artifact.")
    model, preprocessing, wavelength_selection, _ = _unwrap_model_artifact(artifact)
    if model is None or not callable(getattr(model, "predict", None)):
        raise ValueError("Reference model artifact has no usable regression model.")
    processed, _ = _apply_artifact_preprocessing(
        np.asarray(X, dtype=float),
        np.asarray(source_wv, dtype=float),
        preprocessing,
        input_preprocessed=False,
    )
    selected = _apply_artifact_wavelength_selection(processed, wavelength_selection)
    return np.asarray(model.predict(selected), dtype=float).ravel()


@tool("nir_list_calibration_transfer_methods", parse_docstring=True)
def nir_list_calibration_transfer_methods_tool(
    runtime: Runtime,  # noqa: ARG001
    tool_call_id: Annotated[str, InjectedToolCallId] = "",  # noqa: ARG001
) -> str:
    """List the separate Chemotools calibration-transfer methods and parameters.

    Args:
        tool_call_id: Injected tool-call identifier.

    Returns:
        JSON containing DS, PDS, and SST. These methods require paired spectra
        from two instruments and are never ordinary preprocessing candidates.
    """
    try:
        from nir_core.calibration_transfer import (
            TRANSFER_CATALOG_VERSION,
            list_transfer_methods,
        )

        methods = list_transfer_methods()
        return _ok(
            {
                "status": "ok",
                "catalog_version": TRANSFER_CATALOG_VERSION,
                "count": len(methods),
                "methods": methods,
            }
        )
    except Exception as exc:  # noqa: BLE001
        return _err(
            f"{type(exc).__name__}: {exc}",
            code="nir_calibration_transfer_catalog_unavailable",
        )


@tool("nir_fit_calibration_transfer", parse_docstring=True)
def nir_fit_calibration_transfer_tool(
    runtime: Runtime,
    source_path: str,
    target_path: str,
    output_path: str,
    source_instrument_id: str,
    target_instrument_id: str,
    method: str = "ds",
    parameters_json: str | None = None,
    pairing_mode: str = "sample_names",
    validation_fraction: float = 0.25,
    random_state: int = 42,
    minimum_improvement_percent: float = 5.0,
    reference_model_path: str | None = None,
    reference_model_id: str | None = None,
    validation_source_path: str | None = None,
    validation_target_path: str | None = None,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",  # noqa: ARG001
) -> str:
    """Fit and validate a target-to-source calibration transfer.

    Args:
        source_path: Reference/source-instrument NPZ with X, wv, and sample_names.
        target_path: Target-instrument NPZ for the same physical samples.
        output_path: Trusted output artifact path under /mnt/user-data/outputs.
        source_instrument_id: Stable ID of the instrument owning the reference model.
        target_instrument_id: Stable ID of the instrument being adapted.
        method: ds, pds, or sst.
        parameters_json: Method parameters returned by the transfer catalog.
        pairing_mode: sample_names (default) or explicitly confirmed row_order.
        validation_fraction: Internal paired holdout fraction from 0.1 to 0.5.
        random_state: Deterministic paired split seed.
        minimum_improvement_percent: Minimum spectral-RMSE improvement to validate.
        reference_model_path: Optional trusted source-instrument regression model.
        reference_model_id: Optional reference-model identity to bind in provenance.
        validation_source_path: Optional independent paired source-instrument NPZ.
        validation_target_path: Matching independent paired target-instrument NPZ.

    Returns:
        JSON with pairing evidence, validation scope, spectral/prediction metrics,
        exact direction/provider metadata, and artifact path. Production approval
        requires both independent paired files and a trusted reference model.
    """
    try:
        from nir_core.calibration_transfer import (
            CalibrationTransfer,
            evaluate_prediction_transfer,
            evaluate_spectral_transfer,
            paired_train_validation_indices,
        )

        _require_output_artifact_path(output_path)
        parameters = _parse_parameters(parameters_json)
        if not np.isfinite(minimum_improvement_percent) or not (0.0 <= minimum_improvement_percent <= 100.0):
            raise ValueError("minimum_improvement_percent must be between 0 and 100.")
        has_validation_source = bool(validation_source_path and str(validation_source_path).strip())
        has_validation_target = bool(validation_target_path and str(validation_target_path).strip())
        if has_validation_source != has_validation_target:
            raise ValueError("validation_source_path and validation_target_path must be provided together.")
        real_source = _resolve(runtime, source_path, read_only=True)
        real_target = _resolve(runtime, target_path, read_only=True)
        real_output = _resolve(runtime, output_path, read_only=False)
        source_data = _load_npz_safely(real_source)
        target_data = _load_npz_safely(real_target)
        try:
            source, target, sample_ids = _pair_loaded_data(
                source_data,
                target_data,
                pairing_mode=pairing_mode,
            )
        except (KeyError, TypeError, ValueError) as exc:
            return _err(
                f"{type(exc).__name__}: {exc}",
                code="nir_calibration_transfer_pairing_invalid",
            )
        source_wv, target_wv = _source_and_target_axes(source_data, target_data)
        if has_validation_source:
            real_validation_source = _resolve(
                runtime,
                str(validation_source_path),
                read_only=True,
            )
            real_validation_target = _resolve(
                runtime,
                str(validation_target_path),
                read_only=True,
            )
            validation_source_data = _load_npz_safely(real_validation_source)
            validation_target_data = _load_npz_safely(real_validation_target)
            try:
                validation_source, validation_target, validation_sample_ids = _pair_loaded_data(
                    validation_source_data,
                    validation_target_data,
                    pairing_mode=pairing_mode,
                )
            except (KeyError, TypeError, ValueError) as exc:
                return _err(
                    f"{type(exc).__name__}: {exc}",
                    code="nir_calibration_transfer_validation_pairing_invalid",
                )
            validation_source_wv, validation_target_wv = _source_and_target_axes(
                validation_source_data,
                validation_target_data,
            )
            if not np.array_equal(validation_source_wv, source_wv):
                raise ValueError("Independent validation source wv must exactly match the training source axis.")
            if not np.array_equal(validation_target_wv, target_wv):
                raise ValueError("Independent validation target wv must exactly match the training target axis.")
            if pairing_mode == "sample_names" and set(sample_ids).intersection(validation_sample_ids):
                raise ValueError("Independent validation sample_names must not overlap the transfer-training sample_names.")
            if _row_fingerprints(source).intersection(_row_fingerprints(validation_source)) or _row_fingerprints(target).intersection(_row_fingerprints(validation_target)):
                raise ValueError("Independent validation spectra must not duplicate transfer-training rows.")
            fit_source = source
            fit_target = target
            fit_sample_ids = sample_ids
            scope = "independent_paired_validation"
            selection_warning = "Independent paired spectra were not used to fit the transfer."
            validation_y_data = validation_source_data
            validation_indices = None
        else:
            train, validation = paired_train_validation_indices(
                source.shape[0],
                validation_fraction=validation_fraction,
                random_state=random_state,
            )
            fit_source = source[train]
            fit_target = target[train]
            fit_sample_ids = tuple(sample_ids[index] for index in train)
            validation_source = source[validation]
            validation_target = target[validation]
            validation_sample_ids = tuple(sample_ids[index] for index in validation)
            scope = "internal_paired_holdout"
            selection_warning = "This internal holdout is useful for development but cannot by itself approve a production transfer."
            validation_y_data = source_data
            validation_indices = validation

        validation_transfer = CalibrationTransfer(
            method=method,
            source_instrument_id=source_instrument_id,
            target_instrument_id=target_instrument_id,
            parameters=parameters,
        ).fit(
            fit_target,
            fit_source,
            target_wv=target_wv,
            source_wv=source_wv,
            sample_ids=fit_sample_ids,
        )
        transformed_validation = validation_transfer.transform(
            validation_target,
            target_wv=target_wv,
            target_instrument_id=target_instrument_id,
        )
        aligned_validation = _aligned_target(
            validation_transfer,
            validation_target,
            target_wv,
        )
        spectral = evaluate_spectral_transfer(
            validation_source,
            aligned_validation,
            transformed_validation,
        )
        spectral_passed = bool(spectral["after"]["rmse"] < spectral["before"]["rmse"] and spectral["improvement_percent"] >= minimum_improvement_percent)
        prediction = None
        reference_model_binding = None
        prediction_passed = None
        if reference_model_path:
            if not reference_model_id or not str(reference_model_id).strip():
                raise ValueError("reference_model_id is required when reference_model_path is provided.")
            real_reference_model = _resolve(
                runtime,
                reference_model_path,
                read_only=True,
            )
            reference_model = _load_trusted_model_artifact(
                real_reference_model,
                reference_model_path,
            )
            y = _single_target_y(
                validation_y_data,
                expected_rows=(source.shape[0] if validation_indices is not None else validation_source.shape[0]),
                path_label=("source" if validation_indices is not None else "validation source"),
            )
            y_validation = y[validation_indices] if validation_indices is not None else y
            prediction = evaluate_prediction_transfer(
                y_validation,
                _predict_reference_model(
                    reference_model,
                    aligned_validation,
                    source_wv,
                ),
                _predict_reference_model(
                    reference_model,
                    transformed_validation,
                    source_wv,
                ),
            )
            prediction_passed = bool(prediction["after"]["rmsep"] < prediction["before"]["rmsep"] and prediction["improvement_percent"] >= minimum_improvement_percent)
            reference_model_binding = {
                "model_id": str(reference_model_id).strip(),
                "artifact_path": reference_model_path,
                "artifact_sha256": _sha256_file(real_reference_model),
                "source_instrument_id": str(source_instrument_id).strip(),
            }

        passed = spectral_passed and (prediction_passed if prediction_passed is not None else True)
        if not passed:
            validation_status = "rejected"
        elif prediction_passed is True:
            validation_status = "production_validated" if scope == "independent_paired_validation" else "model_validated_internal"
        else:
            validation_status = "spectrally_validated"

        final_transfer = CalibrationTransfer(
            method=method,
            source_instrument_id=source_instrument_id,
            target_instrument_id=target_instrument_id,
            parameters=parameters,
        ).fit(
            target,
            source,
            target_wv=target_wv,
            source_wv=source_wv,
            sample_ids=sample_ids,
        )
        manifest = final_transfer.manifest()
        validation_record = {
            "scope": scope,
            "selection_warning": selection_warning,
            "random_state": (int(random_state) if scope == "internal_paired_holdout" else None),
            "train_pairs": int(fit_source.shape[0]),
            "validation_pairs": int(validation_source.shape[0]),
            "minimum_improvement_percent": float(minimum_improvement_percent),
            "spectral": spectral,
            "spectral_passed": spectral_passed,
            "prediction": prediction,
            "prediction_passed": prediction_passed,
            "passed": passed,
        }
        if reference_model_binding is not None:
            manifest["reference_model_binding"] = reference_model_binding
        artifact = {
            "format": "nir_calibration_transfer_artifact",
            "version": 1,
            "validation_status": validation_status,
            "production_qualification": (
                "approved"
                if validation_status == "production_validated"
                else ("requires_independent_validation" if validation_status == "model_validated_internal" else ("requires_reference_model_validation" if validation_status == "spectrally_validated" else "rejected"))
            ),
            "transfer": final_transfer,
            "manifest": manifest,
            "validation": validation_record,
            "reference_model_binding": reference_model_binding,
        }
        os.makedirs(os.path.dirname(real_output), exist_ok=True)
        _write_trusted_transfer_artifact(artifact, real_output)
        return _ok(
            {
                "status": "ok",
                "artifact_path": output_path,
                "validation_status": validation_status,
                "production_qualification": artifact["production_qualification"],
                "pairing": {
                    "mode": pairing_mode,
                    "paired_samples": len(sample_ids),
                    "unique_ids_verified": pairing_mode == "sample_names",
                    "independent_validation_pairs": (len(validation_sample_ids) if scope == "independent_paired_validation" else 0),
                },
                "manifest": manifest,
                "reference_model_binding": reference_model_binding,
                "validation": validation_record,
            }
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return _err(
            f"{type(exc).__name__}: {exc}",
            code="nir_calibration_transfer_invalid",
        )
    except Exception as exc:  # noqa: BLE001
        return _err(
            f"{type(exc).__name__}: {exc}",
            code="nir_calibration_transfer_failed",
        )


@tool("nir_apply_calibration_transfer", parse_docstring=True)
def nir_apply_calibration_transfer_tool(
    runtime: Runtime,
    artifact_path: str,
    target_path: str,
    output_path: str,
    target_instrument_id: str,
    tool_call_id: Annotated[str, InjectedToolCallId] = "",  # noqa: ARG001
) -> str:
    """Apply a validated transfer artifact to spectra from its bound target instrument.

    Args:
        artifact_path: Trusted transfer artifact under /mnt/user-data/outputs.
        target_path: Target-instrument NPZ containing X and its original wv axis.
        output_path: NPZ path for spectra represented on the source instrument axis.
        target_instrument_id: Must match the artifact's target instrument binding.

    Returns:
        JSON with exact transfer direction, output shape, and qualification status.
    """
    try:
        real_artifact = _resolve(runtime, artifact_path, read_only=True)
        real_target = _resolve(runtime, target_path, read_only=True)
        real_output = _resolve(runtime, output_path, read_only=False)
        artifact = _load_trusted_transfer_artifact(real_artifact, artifact_path)
        if artifact.get("validation_status") == "rejected":
            return _err(
                "Rejected calibration-transfer artifacts cannot be applied.",
                code="nir_calibration_transfer_not_validated",
            )
        transfer = artifact["transfer"]
        target_data = _load_npz_safely(real_target)
        target = np.asarray(
            _required_array(target_data, "X", path_label="target"),
            dtype=float,
        )
        target_wv = np.asarray(
            _required_array(target_data, "wv", path_label="target"),
            dtype=float,
        ).ravel()
        transformed = transfer.transform(
            target,
            target_wv=target_wv,
            target_instrument_id=target_instrument_id,
        )
        output_data = dict(target_data)
        output_data["X"] = transformed
        output_data["wv"] = transfer.source_wavelengths_
        os.makedirs(os.path.dirname(real_output), exist_ok=True)
        np.savez_compressed(real_output, **output_data)
        return _ok(
            {
                "status": "ok",
                "output_path": output_path,
                "output_shape": list(transformed.shape),
                "direction": artifact["manifest"]["direction"],
                "validation_status": artifact["validation_status"],
                "production_qualification": artifact["production_qualification"],
            }
        )
    except (KeyError, TypeError, ValueError) as exc:
        return _err(
            f"{type(exc).__name__}: {exc}",
            code="nir_calibration_transfer_apply_invalid",
        )
    except Exception as exc:  # noqa: BLE001
        return _err(
            f"{type(exc).__name__}: {exc}",
            code="nir_calibration_transfer_apply_failed",
        )


@tool("nir_evaluate_calibration_transfer", parse_docstring=True)
def nir_evaluate_calibration_transfer_tool(
    runtime: Runtime,
    artifact_path: str,
    source_path: str,
    target_path: str,
    pairing_mode: str = "sample_names",
    tool_call_id: Annotated[str, InjectedToolCallId] = "",  # noqa: ARG001
) -> str:
    """Evaluate a transfer artifact on a separate paired source/target dataset.

    Args:
        artifact_path: Trusted transfer artifact under /mnt/user-data/outputs.
        source_path: Independent reference/source-instrument NPZ.
        target_path: Matching target-instrument NPZ.
        pairing_mode: sample_names (default) or explicitly confirmed row_order.

    Returns:
        JSON with before/after spectral RMSE, bias, slope, R2, and improvement.
    """
    try:
        from nir_core.calibration_transfer import evaluate_spectral_transfer

        real_artifact = _resolve(runtime, artifact_path, read_only=True)
        real_source = _resolve(runtime, source_path, read_only=True)
        real_target = _resolve(runtime, target_path, read_only=True)
        artifact = _load_trusted_transfer_artifact(real_artifact, artifact_path)
        transfer = artifact["transfer"]
        source_data = _load_npz_safely(real_source)
        target_data = _load_npz_safely(real_target)
        source, target, sample_ids = _pair_loaded_data(
            source_data,
            target_data,
            pairing_mode=pairing_mode,
        )
        source_wv, target_wv = _source_and_target_axes(source_data, target_data)
        if not np.array_equal(source_wv, transfer.source_wavelengths_):
            raise ValueError("Evaluation source axis does not match the artifact source axis.")
        transformed = transfer.transform(target, target_wv=target_wv)
        spectral = evaluate_spectral_transfer(
            source,
            _aligned_target(transfer, target, target_wv),
            transformed,
        )
        return _ok(
            {
                "status": "ok",
                "scope": "provided_paired_evaluation_set",
                "paired_samples": len(sample_ids),
                "direction": artifact["manifest"]["direction"],
                "spectral": spectral,
                "prediction_validation": "not_performed",
            }
        )
    except (KeyError, TypeError, ValueError) as exc:
        return _err(
            f"{type(exc).__name__}: {exc}",
            code="nir_calibration_transfer_evaluation_invalid",
        )
    except Exception as exc:  # noqa: BLE001
        return _err(
            f"{type(exc).__name__}: {exc}",
            code="nir_calibration_transfer_evaluation_failed",
        )


__all__ = [
    "nir_apply_calibration_transfer_tool",
    "nir_evaluate_calibration_transfer_tool",
    "nir_fit_calibration_transfer_tool",
    "nir_list_calibration_transfer_methods_tool",
]
