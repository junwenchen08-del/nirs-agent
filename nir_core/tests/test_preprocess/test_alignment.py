"""Tests for wavelength-axis validation and resampling."""

from __future__ import annotations

import numpy as np
import pytest

from nir_core.preprocess.alignment import SpectralAxisAligner, resample_spectra


def test_resample_linear_spectra_matches_analytic_signal():
    source_wv = np.linspace(900.0, 1800.0, 91)
    target_wv = np.linspace(950.0, 1750.0, 161)
    X = np.vstack([2.0 + 0.01 * source_wv, -1.0 + 0.02 * source_wv])

    out = resample_spectra(X, source_wv, target_wv)

    expected = np.vstack([2.0 + 0.01 * target_wv, -1.0 + 0.02 * target_wv])
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_resample_accepts_descending_source_axis():
    source_wv = np.linspace(1800.0, 900.0, 91)
    target_wv = np.linspace(950.0, 1750.0, 81)
    X = (0.5 * source_wv)[None, :]
    out = resample_spectra(X, source_wv, target_wv)
    np.testing.assert_allclose(out[0], 0.5 * target_wv, atol=1e-12)


def test_resample_rejects_extrapolation_by_default():
    source_wv = np.linspace(1000.0, 1700.0, 80)
    target_wv = np.linspace(900.0, 1800.0, 100)
    with pytest.raises(ValueError, match="outside"):
        resample_spectra(np.ones((2, 80)), source_wv, target_wv)


def test_resample_can_explicitly_enable_extrapolation():
    source_wv = np.linspace(1000.0, 1700.0, 80)
    target_wv = np.linspace(950.0, 1750.0, 90)
    X = (3.0 + 0.02 * source_wv)[None, :]
    out = resample_spectra(X, source_wv, target_wv, allow_extrapolation=True)
    np.testing.assert_allclose(out[0], 3.0 + 0.02 * target_wv, atol=1e-10)


@pytest.mark.parametrize(
    "source_wv",
    [
        np.array([1000.0, 1100.0, 1100.0, 1200.0]),
        np.array([1000.0, 1200.0, 1100.0, 1300.0]),
        np.array([1000.0, np.nan, 1200.0, 1300.0]),
    ],
)
def test_resample_rejects_invalid_source_axis(source_wv):
    with pytest.raises(ValueError):
        resample_spectra(
            np.ones((2, source_wv.size)), source_wv, np.array([1050.0, 1150.0])
        )


def test_axis_aligner_is_serializable_and_reusable(tmp_path):
    import joblib

    source_wv = np.linspace(900.0, 1800.0, 91)
    target_wv = np.linspace(950.0, 1750.0, 81)
    X = np.sin(source_wv / 200.0)[None, :]
    aligner = SpectralAxisAligner(target_wv=target_wv).fit(source_wv)
    expected = aligner.transform(X, source_wv)
    path = tmp_path / "aligner.joblib"
    joblib.dump(aligner, path)
    loaded = joblib.load(path)
    actual = loaded.transform(X, source_wv)
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_array_equal(loaded.target_wavelengths_, target_wv)


def test_axis_aligner_rejects_source_axis_changed_after_fit():
    source_wv = np.linspace(900.0, 1800.0, 91)
    target_wv = np.linspace(950.0, 1750.0, 81)
    aligner = SpectralAxisAligner(target_wv=target_wv).fit(source_wv)

    with pytest.raises(ValueError, match="does not match"):
        aligner.transform(np.ones((2, 91)), source_wv + 0.01)
