from __future__ import annotations

import numpy as np

from nir_core.utils.scientific_validation import (
    build_reproducibility_manifest,
    validate_calibration_dataset,
    validate_partition_separation,
)


def _valid_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(7)
    X = rng.normal(size=(12, 5))
    y = np.linspace(1.0, 3.0, 12)
    wv = np.linspace(900.0, 1700.0, 5)
    return X, y, wv


def test_valid_dataset_produces_stable_scientific_evidence() -> None:
    X, y, wv = _valid_data()
    first = validate_calibration_dataset(X, y, wv)
    second = validate_calibration_dataset(X.copy(), y.copy(), wv.copy())

    assert first["passed"] is True
    assert first["errors"] == []
    assert first["summary"]["wavelength_direction"] == "ascending"
    assert first["data_fingerprint"] == second["data_fingerprint"]


def test_conflicting_duplicate_spectra_are_blocking() -> None:
    X, y, wv = _valid_data()
    X[4] = X[1]
    y[4] = y[1] + 10.0

    report = validate_calibration_dataset(X, y, wv)

    assert report["passed"] is False
    assert "conflicting_duplicate_spectra" in {
        item["code"] for item in report["errors"]
    }


def test_nonmonotonic_wavelength_axis_is_blocking() -> None:
    X, y, wv = _valid_data()
    wv[[2, 3]] = wv[[3, 2]]

    report = validate_calibration_dataset(X, y, wv)

    assert report["passed"] is False
    assert "nonmonotonic_wavelengths" in {item["code"] for item in report["errors"]}


def test_equal_replicates_are_reported_without_failing_dataset_gate() -> None:
    X, y, wv = _valid_data()
    X[4] = X[1]
    y[4] = y[1]

    report = validate_calibration_dataset(X, y, wv)

    assert report["passed"] is True
    assert "duplicate_spectra" in {item["code"] for item in report["warnings"]}


def test_cross_partition_duplicate_spectra_are_blocking() -> None:
    X, _, _ = _valid_data()
    report = validate_partition_separation(
        {
            "calibration": X[:6],
            "tuning": np.vstack([X[1], X[6:8]]),
            "holdout": X[8:],
        }
    )

    assert report["passed"] is False
    assert report["errors"][0]["code"] == "cross_partition_duplicate_spectra"


def test_reproducibility_manifest_records_seed_versions_and_parameters() -> None:
    manifest = build_reproducibility_manifest(
        random_state=42,
        protocol="three_way_holdout",
        input_sha256="a" * 64,
        parameters={"test_ratio": 0.2},
    )

    assert manifest["random_state"] == 42
    assert manifest["input_sha256"] == "a" * 64
    assert manifest["versions"]["numpy"]
    assert manifest["parameters"] == {"test_ratio": 0.2}
