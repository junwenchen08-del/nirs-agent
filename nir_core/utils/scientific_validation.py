"""Scientific validity and reproducibility gates for NIR calibration.

The functions in this module are deliberately independent from DeerFlow.  They
return JSON-serialisable reports so every training entry point can persist the
same evidence and model registration can enforce it without rerunning training.
"""

from __future__ import annotations

import hashlib
import platform
from collections.abc import Mapping
from importlib import metadata
from typing import Any

import numpy as np

from nir_core.utils.spectral_axis import summarize_spectral_axis


def _package_version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "unknown"


def _array_digest(*arrays: np.ndarray | None) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        if array is None:
            digest.update(b"<none>")
            continue
        normalized = np.ascontiguousarray(np.asarray(array))
        digest.update(str(normalized.dtype).encode("ascii"))
        digest.update(repr(normalized.shape).encode("ascii"))
        digest.update(normalized.tobytes())
    return digest.hexdigest()


def _duplicate_groups(X: np.ndarray) -> list[list[int]]:
    groups: dict[bytes, list[int]] = {}
    contiguous = np.ascontiguousarray(X)
    for index, row in enumerate(contiguous):
        groups.setdefault(row.tobytes(), []).append(index)
    return [indices for indices in groups.values() if len(indices) > 1]


def validate_calibration_dataset(
    X: np.ndarray,
    y: np.ndarray,
    wv: np.ndarray | None = None,
    *,
    minimum_samples: int = 6,
) -> dict[str, Any]:
    """Validate alignment, numeric integrity, wavelength axis, and duplicates.

    Exact duplicate spectra with conflicting reference values are blocking.
    Repeated spectra with equal references are retained but reported so callers
    can decide whether they are legitimate replicate measurements.
    """

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    if X.ndim != 2:
        errors.append(
            {
                "code": "spectra_not_2d",
                "message": f"Spectral matrix must be 2-D; observed shape {X.shape}.",
            }
        )
        return {
            "schema_version": 1,
            "passed": False,
            "errors": errors,
            "warnings": warnings,
        }
    if y.ndim == 1:
        y_matrix = y[:, None]
    elif y.ndim == 2:
        y_matrix = y
    else:
        errors.append(
            {
                "code": "targets_not_1d_or_2d",
                "message": f"Reference values must be 1-D or 2-D; observed shape {y.shape}.",
            }
        )
        y_matrix = np.empty((0, 0), dtype=float)

    n_samples, n_wavelengths = X.shape
    if y_matrix.shape[0] != n_samples:
        errors.append(
            {
                "code": "sample_target_misalignment",
                "message": (
                    f"Spectra contain {n_samples} samples but reference values contain {y_matrix.shape[0]} rows."
                ),
            }
        )
    if n_samples < int(minimum_samples):
        errors.append(
            {
                "code": "insufficient_samples",
                "message": (
                    f"At least {int(minimum_samples)} samples are required; observed {n_samples}."
                ),
            }
        )
    if n_wavelengths < 2:
        errors.append(
            {
                "code": "insufficient_wavelengths",
                "message": "At least two spectral variables are required.",
            }
        )
    nonfinite_x = int(np.size(X) - np.isfinite(X).sum())
    if nonfinite_x:
        errors.append(
            {
                "code": "nonfinite_spectra",
                "message": f"Spectral matrix contains {nonfinite_x} non-finite values.",
                "count": nonfinite_x,
            }
        )
    nonfinite_y = int(np.size(y_matrix) - np.isfinite(y_matrix).sum())
    if nonfinite_y:
        errors.append(
            {
                "code": "nonfinite_targets",
                "message": f"Reference values contain {nonfinite_y} non-finite values.",
                "count": nonfinite_y,
            }
        )

    constant_targets: list[int] = []
    if y_matrix.shape[0] == n_samples and np.isfinite(y_matrix).all():
        constant_targets = [
            int(index)
            for index in range(y_matrix.shape[1])
            if np.ptp(y_matrix[:, index]) <= 1e-12
        ]
        if constant_targets:
            errors.append(
                {
                    "code": "constant_targets",
                    "message": (
                        f"Reference values have zero usable variation for target columns {constant_targets}."
                    ),
                    "target_indices": constant_targets,
                }
            )

    dead_columns = (
        np.flatnonzero(np.ptp(X, axis=0) <= 1e-12).astype(int).tolist()
        if X.size and np.isfinite(X).all()
        else []
    )
    if dead_columns:
        warnings.append(
            {
                "code": "constant_wavelength_columns",
                "message": (
                    f"{len(dead_columns)} spectral columns have no variation and carry no calibration information."
                ),
                "indices": dead_columns[:50],
            }
        )

    wavelength_direction = "missing"
    if wv is not None:
        wv = np.asarray(wv, dtype=float).ravel()
        axis_direction = str(summarize_spectral_axis(wv)["axis_direction"])
        if wv.size != n_wavelengths:
            errors.append(
                {
                    "code": "wavelength_length_mismatch",
                    "message": (
                        f"Wavelength axis has {wv.size} values but spectra have {n_wavelengths} columns."
                    ),
                }
            )
        elif not np.isfinite(wv).all():
            errors.append(
                {
                    "code": "nonfinite_wavelengths",
                    "message": "Wavelength axis contains non-finite values.",
                }
            )
        else:
            differences = np.diff(wv)
            if np.any(differences == 0):
                errors.append(
                    {
                        "code": "duplicate_wavelengths",
                        "message": "Wavelength axis contains duplicate values.",
                    }
                )
                wavelength_direction = "invalid"
            elif axis_direction == "ascending":
                wavelength_direction = "ascending"
            elif axis_direction == "descending":
                wavelength_direction = "descending"
            else:
                errors.append(
                    {
                        "code": "nonmonotonic_wavelengths",
                        "message": (
                            "Wavelength axis must be strictly ascending or strictly descending to preserve spectral alignment."
                        ),
                    }
                )
                wavelength_direction = "invalid"

    duplicate_groups = _duplicate_groups(X) if np.isfinite(X).all() else []
    conflicting_groups: list[list[int]] = []
    consistent_groups: list[list[int]] = []
    if y_matrix.shape[0] == n_samples and np.isfinite(y_matrix).all():
        for indices in duplicate_groups:
            references = y_matrix[indices]
            if np.allclose(
                references,
                references[0],
                rtol=1e-7,
                atol=1e-10,
                equal_nan=False,
            ):
                consistent_groups.append(indices)
            else:
                conflicting_groups.append(indices)
    if conflicting_groups:
        errors.append(
            {
                "code": "conflicting_duplicate_spectra",
                "message": (
                    f"{len(conflicting_groups)} exact duplicate spectral groups have conflicting reference values."
                ),
                "groups": conflicting_groups[:20],
            }
        )
    if consistent_groups:
        warnings.append(
            {
                "code": "duplicate_spectra",
                "message": (
                    f"{len(consistent_groups)} exact duplicate spectral groups were found; keep them together during splitting if they are replicates."
                ),
                "groups": consistent_groups[:20],
            }
        )

    return {
        "schema_version": 1,
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "summary": {
            "n_samples": int(n_samples),
            "n_wavelengths": int(n_wavelengths),
            "n_targets": int(y_matrix.shape[1]) if y_matrix.ndim == 2 else 0,
            "wavelength_direction": wavelength_direction,
            "duplicate_group_count": len(duplicate_groups),
            "conflicting_duplicate_group_count": len(conflicting_groups),
            "constant_wavelength_count": len(dead_columns),
        },
        "data_fingerprint": _array_digest(X, y_matrix, wv),
    }


def validate_partition_separation(
    partitions: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Reject exact spectra duplicated across calibration/tuning/test splits."""

    names = list(partitions)
    normalized: dict[str, np.ndarray] = {}
    errors: list[dict[str, Any]] = []
    for name, values in partitions.items():
        array = np.asarray(values, dtype=float)
        if array.ndim != 2:
            errors.append(
                {
                    "code": "partition_not_2d",
                    "message": f"Partition {name!r} must be 2-D.",
                }
            )
        normalized[name] = np.ascontiguousarray(array)

    overlaps: list[dict[str, Any]] = []
    for left_index, left_name in enumerate(names):
        left = normalized[left_name]
        if left.ndim != 2:
            continue
        left_rows: dict[bytes, list[int]] = {}
        for row_index, row in enumerate(left):
            left_rows.setdefault(row.tobytes(), []).append(row_index)
        for right_name in names[left_index + 1 :]:
            right = normalized[right_name]
            if right.ndim != 2 or right.shape[1] != left.shape[1]:
                continue
            pairs: list[dict[str, Any]] = []
            for row_index, row in enumerate(right):
                for matching_index in left_rows.get(row.tobytes(), []):
                    pairs.append(
                        {
                            left_name: int(matching_index),
                            right_name: int(row_index),
                        }
                    )
            if pairs:
                overlaps.append(
                    {
                        "left": left_name,
                        "right": right_name,
                        "count": len(pairs),
                        "sample_pairs": pairs[:20],
                    }
                )
    if overlaps:
        errors.append(
            {
                "code": "cross_partition_duplicate_spectra",
                "message": (
                    "Exact duplicate spectra occur across data partitions; evaluation would be optimistically biased."
                ),
                "overlaps": overlaps,
            }
        )

    return {
        "schema_version": 1,
        "passed": not errors,
        "errors": errors,
        "partitions": {
            name: int(values.shape[0])
            for name, values in normalized.items()
            if values.ndim == 2
        },
    }


def build_reproducibility_manifest(
    *,
    random_state: int,
    protocol: str,
    input_sha256: str,
    parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the minimum environment and seed record required for replay."""

    if len(input_sha256) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in input_sha256
    ):
        raise ValueError("input_sha256 must be a 64-character hexadecimal digest")
    return {
        "schema_version": 1,
        "protocol": str(protocol),
        "random_state": int(random_state),
        "input_sha256": input_sha256.lower(),
        "deterministic_split": True,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": _package_version("scipy"),
            "scikit_learn": _package_version("scikit-learn"),
            "nir_core": _package_version("nir-core"),
        },
        "parameters": dict(parameters or {}),
    }
