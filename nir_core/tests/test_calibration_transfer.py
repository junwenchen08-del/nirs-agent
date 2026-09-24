"""Scientific and safety contracts for Chemotools calibration transfer."""

from __future__ import annotations

import joblib
import numpy as np
import pytest


def _paired_spectra(
    *,
    n_samples: int = 48,
    n_wavelengths: int = 31,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2048)
    wv = np.linspace(1000.0, 1800.0, n_wavelengths)
    latent = rng.normal(size=(n_samples, 5))
    loadings = rng.normal(size=(5, n_wavelengths))
    source = latent @ loadings
    target = source * 1.25
    return source, target, wv


def test_pairing_reorders_target_by_unique_sample_ids():
    from nir_core.calibration_transfer import pair_transfer_samples

    source, target, _ = _paired_spectra(n_samples=4)
    source_ids = ["A", "B", "C", "D"]
    target_ids = ["C", "A", "D", "B"]
    reordered = target[[2, 0, 3, 1]]

    paired_source, paired_target, paired_ids = pair_transfer_samples(
        source,
        reordered,
        source_ids=source_ids,
        target_ids=target_ids,
    )

    np.testing.assert_array_equal(paired_source, source)
    np.testing.assert_array_equal(paired_target, target)
    assert paired_ids == tuple(source_ids)


@pytest.mark.parametrize(
    ("source_ids", "target_ids", "message"),
    [
        (["A", "A"], ["A", "B"], "unique"),
        (["A", "B"], ["A", "C"], "same sample IDs"),
        (["A"], ["A", "B"], "row count"),
    ],
)
def test_pairing_fails_closed_for_ambiguous_or_mismatched_ids(
    source_ids,
    target_ids,
    message,
):
    from nir_core.calibration_transfer import pair_transfer_samples

    source = np.ones((2, 5))
    target = np.ones((2, 5))
    with pytest.raises(ValueError, match=message):
        pair_transfer_samples(
            source,
            target,
            source_ids=source_ids,
            target_ids=target_ids,
        )


@pytest.mark.parametrize(
    ("method", "params"),
    [
        ("ds", {}),
        ("pds", {"window_length": 3, "n_components": 2, "storage": "band"}),
        ("sst", {"n_components": 3, "with_mean": True, "with_std": False}),
    ],
)
def test_chemotools_transfer_methods_improve_independent_paired_spectra(
    method,
    params,
):
    from nir_core.calibration_transfer import (
        CalibrationTransfer,
        evaluate_spectral_transfer,
        paired_train_validation_indices,
    )

    source, target, wv = _paired_spectra()
    train, validation = paired_train_validation_indices(
        source.shape[0],
        validation_fraction=0.25,
        random_state=7,
    )
    transfer = CalibrationTransfer(
        method=method,
        source_instrument_id="reference-a",
        target_instrument_id="target-b",
        parameters=params,
    ).fit(
        target[train],
        source[train],
        target_wv=wv,
        source_wv=wv,
    )

    transformed = transfer.transform(
        target[validation],
        target_wv=wv,
        target_instrument_id="target-b",
    )
    metrics = evaluate_spectral_transfer(
        source[validation],
        target[validation],
        transformed,
    )

    assert metrics["after"]["rmse"] < metrics["before"]["rmse"]
    assert metrics["improvement_percent"] > 20.0
    manifest = transfer.manifest()
    assert manifest["direction"] == "target-b_to_reference-a"
    assert manifest["provider"] == "chemotools"
    assert manifest["provider_version"] == "0.4.4"
    assert manifest["source_axis_hash"] == manifest["target_axis_hash"]


def test_transfer_aligns_target_axis_without_extrapolation():
    from nir_core.calibration_transfer import CalibrationTransfer

    source, _, source_wv = _paired_spectra(n_wavelengths=25)
    target_wv = np.linspace(900.0, 1900.0, 41)
    target = np.vstack([np.interp(target_wv, source_wv, row) for row in source]) * 1.25
    transfer = CalibrationTransfer(
        method="ds",
        source_instrument_id="source",
        target_instrument_id="target",
    ).fit(
        target,
        source,
        target_wv=target_wv,
        source_wv=source_wv,
    )

    transformed = transfer.transform(target, target_wv=target_wv)
    assert transformed.shape == source.shape
    assert transfer.manifest()["axis_alignment"]["applied"] is True
    assert transfer.manifest()["axis_alignment"]["allow_extrapolation"] is False


def test_transfer_rejects_axis_extrapolation_and_wrong_instrument():
    from nir_core.calibration_transfer import CalibrationTransfer

    source, target, wv = _paired_spectra()
    transfer = CalibrationTransfer(
        method="ds",
        source_instrument_id="source",
        target_instrument_id="target",
    ).fit(target, source, target_wv=wv, source_wv=wv)

    with pytest.raises(ValueError, match="instrument"):
        transfer.transform(
            target,
            target_wv=wv,
            target_instrument_id="another-target",
        )
    with pytest.raises(ValueError, match="outside"):
        CalibrationTransfer(
            method="ds",
            source_instrument_id="source",
            target_instrument_id="target",
        ).fit(
            target[:, 2:-2],
            source,
            target_wv=wv[2:-2],
            source_wv=wv,
        )


def test_transfer_round_trip_preserves_binding_and_predictions(tmp_path):
    from nir_core.calibration_transfer import CalibrationTransfer

    source, target, wv = _paired_spectra()
    transfer = CalibrationTransfer(
        method="ds",
        source_instrument_id="source",
        target_instrument_id="target",
    ).fit(target, source, target_wv=wv, source_wv=wv)
    expected = transfer.transform(target, target_wv=wv)
    path = tmp_path / "transfer.joblib"
    joblib.dump(transfer, path)
    loaded = joblib.load(path)

    assert loaded.manifest() == transfer.manifest()
    np.testing.assert_allclose(loaded.transform(target, target_wv=wv), expected)


def test_transfer_catalog_exposes_parameters_without_entering_preprocessing():
    from nir_core.calibration_transfer import describe_transfer_method
    from nir_core.preprocess.registry import METHOD_REGISTRY

    assert {"ds", "pds", "sst"}.isdisjoint(METHOD_REGISTRY)
    assert (
        describe_transfer_method("pds")["parameter_schema"]["window_length"]["default"]
        == 25
    )
    assert (
        describe_transfer_method("sst")["parameter_schema"]["with_std"]["default"]
        is False
    )


@pytest.mark.parametrize(
    ("method", "parameters", "exception", "message"),
    [
        ("ds", {"n_components": 2}, ValueError, "Unknown parameters"),
        ("pds", {"window_length": 0}, ValueError, "at least 1"),
        ("pds", {"scale": "false"}, TypeError, "boolean"),
        ("pds", {"storage": "sparse"}, ValueError, "dense.*band"),
        ("sst", {"n_components": 1.5}, TypeError, "integer"),
        ("sst", {"with_std": 1}, TypeError, "boolean"),
    ],
)
def test_transfer_parameter_schema_fails_closed(
    method,
    parameters,
    exception,
    message,
):
    from nir_core.calibration_transfer import CalibrationTransfer

    with pytest.raises(exception, match=message):
        CalibrationTransfer(
            method=method,
            source_instrument_id="source",
            target_instrument_id="target",
            parameters=parameters,
        )


def test_prediction_transfer_metrics_require_rmsep_improvement():
    from nir_core.calibration_transfer import evaluate_prediction_transfer

    y_true = np.linspace(0.0, 10.0, 20)
    before = y_true * 1.3 + 1.0
    after = y_true + 0.05
    metrics = evaluate_prediction_transfer(y_true, before, after)

    assert metrics["after"]["rmsep"] < metrics["before"]["rmsep"]
    assert metrics["after"]["bias"] == pytest.approx(0.05)
    assert metrics["after"]["slope"] == pytest.approx(1.0)
    assert metrics["improvement_percent"] > 90.0
