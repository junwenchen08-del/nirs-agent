"""Shared scientific-validity contract for every NIR training entry point."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def dataset_science_gate(
    X: np.ndarray,
    y: np.ndarray,
    wv: np.ndarray | None,
) -> dict[str, Any]:
    from nir_core.utils.scientific_validation import validate_calibration_dataset

    return validate_calibration_dataset(X, y, wv)


def split_science_gate(
    *,
    calibration: np.ndarray,
    tuning: np.ndarray,
    holdout: np.ndarray,
) -> dict[str, Any]:
    from nir_core.utils.scientific_validation import validate_partition_separation

    return validate_partition_separation(
        {
            "calibration": calibration,
            "tuning": tuning,
            "holdout": holdout,
        }
    )


def science_gate_error(report: Mapping[str, Any]) -> str | None:
    if report.get("passed") is True:
        return None
    errors = report.get("errors")
    if not isinstance(errors, list):
        return "Scientific validation failed without structured diagnostics."
    messages = [str(item.get("message") or item.get("code") or "unknown validation error") for item in errors if isinstance(item, Mapping)]
    return "Scientific validation failed: " + "; ".join(messages)


def reproducibility_evidence(
    *,
    random_state: int,
    protocol: str,
    input_sha256: str,
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    from nir_core.utils.scientific_validation import build_reproducibility_manifest

    return build_reproducibility_manifest(
        random_state=random_state,
        protocol=protocol,
        input_sha256=input_sha256,
        parameters=parameters,
    )
