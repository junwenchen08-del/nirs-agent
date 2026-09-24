"""Numerical and safety contracts for Chemotools compatibility adapters."""

from __future__ import annotations

import numpy as np
import pytest

from nir_core.preprocess.baseline import asls as native_asls
from nir_core.preprocess.scaling import autoscale as native_autoscale
from nir_core.preprocess.scaling import mean_center as native_mean_center
from nir_core.preprocess.scaling import normalize as native_normalize
from nir_core.preprocess.scatter import msc as native_msc
from nir_core.preprocess.scatter import snv as native_snv
from nir_core.preprocess.smoothing import sg_derivative as native_sg_derivative
from nir_core.preprocess.smoothing import sg_smooth as native_sg_smooth


@pytest.fixture
def spectra() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(74)
    wv = np.linspace(1000.0, 1800.0, 101)
    base = 0.4 + 0.06 * np.sin(np.linspace(0, 8, wv.size))
    X = np.vstack(
        [
            base * (0.8 + 0.08 * index)
            + 0.01 * (index - 2)
            + rng.normal(0.0, 0.001, wv.size)
            for index in range(6)
        ]
    )
    return X, wv


def test_chemotools_snv_matches_native_and_preserves_1d(spectra):
    from nir_core.preprocess.providers.chemotools import snv

    X, _ = spectra
    np.testing.assert_allclose(snv(X), native_snv(X), rtol=0.0, atol=1e-12)
    assert snv(X[0]).shape == X[0].shape


def test_chemotools_snv_preserves_project_constant_row_safety():
    from nir_core.preprocess.providers.chemotools import snv

    X = np.vstack([np.ones(31), np.linspace(0.0, 1.0, 31)])
    result = snv(X)
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(result[0], np.zeros(31))


def test_chemotools_msc_matches_native_with_training_reference(spectra):
    from nir_core.preprocess.providers.chemotools import msc

    X, _ = spectra
    reference = X[:4].mean(axis=0)
    np.testing.assert_allclose(
        msc(X[4:], reference=reference),
        native_msc(X[4:], reference=reference),
        rtol=1e-11,
        atol=1e-12,
    )


def test_chemotools_msc_preserves_constant_reference_fallback():
    from nir_core.preprocess.providers.chemotools import msc

    X = np.vstack([np.ones(21), np.linspace(0.0, 1.0, 21)])
    reference = np.ones(21)
    np.testing.assert_allclose(msc(X, reference), native_msc(X, reference))


def test_chemotools_sg_smooth_matches_native_interp_edges(spectra):
    from nir_core.preprocess.providers.chemotools import sg_smooth

    X, _ = spectra
    np.testing.assert_allclose(
        sg_smooth(X, window=11, order=2),
        native_sg_smooth(X, window=11, order=2),
        rtol=0.0,
        atol=3e-15,
    )


@pytest.mark.parametrize("deriv", [1, 2])
def test_chemotools_sg_derivative_matches_native_index_scale(spectra, deriv):
    from nir_core.preprocess.providers.chemotools import sg_derivative

    X, _ = spectra
    np.testing.assert_allclose(
        sg_derivative(X, window=11, order=3, deriv=deriv),
        native_sg_derivative(X, window=11, order=3, deriv=deriv),
        rtol=0.0,
        atol=3e-15,
    )


def test_chemotools_sg_derivative_matches_native_wavelength_scale(spectra):
    from nir_core.preprocess.providers.chemotools import sg_derivative

    X, wv = spectra
    expected = native_sg_derivative(
        X,
        window=11,
        order=2,
        deriv=1,
        spacing_mode="wavelength",
        wv=wv,
    )
    actual = sg_derivative(
        X,
        window=11,
        order=2,
        deriv=1,
        spacing_mode="wavelength",
        wv=wv,
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=3e-15)


def test_chemotools_sg_derivative_rejects_irregular_physical_axis(spectra):
    from nir_core.preprocess.providers.chemotools import sg_derivative

    X, wv = spectra
    wv[50] += 0.5
    with pytest.raises(ValueError, match="equally spaced wavelength"):
        sg_derivative(X, spacing_mode="wavelength", wv=wv)


def test_chemotools_asls_matches_native_within_solver_tolerance(spectra):
    from nir_core.preprocess.providers.chemotools import asls

    X, _ = spectra
    np.testing.assert_allclose(
        asls(X, lambda_=1e5, p=0.001),
        native_asls(X, lambda_=1e5, p=0.001),
        rtol=1e-7,
        atol=5e-8,
    )


def test_chemotools_scaling_adapters_match_native_and_handle_constants():
    from nir_core.preprocess.providers.chemotools import (
        autoscale,
        mean_center,
        normalize,
    )

    X = np.array(
        [
            [1.0, 2.0, 5.0],
            [1.0, 4.0, 5.0],
            [1.0, 6.0, 5.0],
        ]
    )
    np.testing.assert_allclose(mean_center(X), native_mean_center(X), atol=1e-12)
    np.testing.assert_allclose(autoscale(X), native_autoscale(X), atol=1e-12)
    np.testing.assert_allclose(
        normalize(X, norm="l1"), native_normalize(X, norm="l1"), atol=1e-12
    )
    np.testing.assert_allclose(
        normalize(X, norm="l2"), native_normalize(X, norm="l2"), atol=1e-12
    )
    assert np.isfinite(autoscale(X)).all()


def test_chemotools_emsc_matches_direct_transformer_on_regular_axis(spectra):
    from chemotools.scatter import ExtendedMultiplicativeScatterCorrection

    from nir_core.preprocess.providers.chemotools import emsc

    X, wv = spectra
    expected = ExtendedMultiplicativeScatterCorrection(
        method="median",
        order=2,
    ).fit_transform(X)
    actual = emsc(X, wv=wv, method="median", polynomial_order=2)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


def test_chemotools_emsc_rejects_irregular_physical_axis(spectra):
    from nir_core.preprocess.providers.chemotools import emsc

    X, wv = spectra
    wv[40] += 0.25
    with pytest.raises(ValueError, match="regular wavelength axis"):
        emsc(X, wv=wv)


def test_chemotools_airpls_matches_direct_transformer(spectra):
    from chemotools.baseline import AirPls

    from nir_core.preprocess.providers.chemotools import airpls

    X, _ = spectra
    expected = AirPls(
        lam=1e5,
        nr_iterations=30,
        solver_type="banded",
        max_iter_after_warmstart=20,
        n_jobs=1,
    ).fit_transform(X)
    actual = airpls(X, lambda_=1e5, max_iters=30)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("deriv", [1, 2])
def test_chemotools_norris_williams_uses_upstream_kernel_with_physical_scaling(
    spectra,
    deriv,
):
    from chemotools.derivative import NorrisWilliams

    from nir_core.preprocess.providers.chemotools import norris_williams_derivative

    X, _ = spectra
    gap = 3
    delta = 2.5
    direct = NorrisWilliams(
        window_length=5,
        gap_size=gap,
        deriv=deriv,
        mode="nearest",
    ).fit_transform(X)
    if deriv == 1:
        scale = gap / ((gap - 1) * delta)
    else:
        half_gap = (gap - 1) / 2.0
        scale = gap / ((half_gap * delta) ** 2)
    np.testing.assert_allclose(
        norris_williams_derivative(
            X,
            gap=gap,
            segment=5,
            deriv=deriv,
            delta=delta,
        ),
        direct * scale,
        rtol=0.0,
        atol=1e-12,
    )


def test_chemotools_detrend_matches_polynomial_correction_on_regular_axis(spectra):
    from chemotools.baseline import PolynomialCorrection

    from nir_core.preprocess.providers.chemotools import detrend

    X, wv = spectra
    expected = PolynomialCorrection(order=2, indices=None).fit_transform(X)
    np.testing.assert_allclose(detrend(X, wv=wv), expected, rtol=0.0, atol=1e-12)


def test_chemotools_detrend_rejects_irregular_physical_axis(spectra):
    from nir_core.preprocess.providers.chemotools import detrend

    X, wv = spectra
    wv[20] += 0.25
    with pytest.raises(ValueError, match="regular wavelength axis"):
        detrend(X, wv=wv)


def test_chemotools_rnv_matches_robust_normal_variate(spectra):
    from chemotools.scatter import RobustNormalVariate

    from nir_core.preprocess.providers.chemotools import rnv

    X, _ = spectra
    expected = RobustNormalVariate(
        percentile=30.0,
        epsilon=1e-9,
        n_jobs=1,
    ).fit_transform(X)
    np.testing.assert_allclose(
        rnv(X, percentile=30.0, epsilon=1e-9),
        expected,
        rtol=0.0,
        atol=1e-12,
    )


def test_new_chemotools_baseline_and_smoothing_adapters_match_upstream(spectra):
    from chemotools.baseline import ArPls, RubberbandCorrection
    from chemotools.smooth import MedianFilter, WhittakerSmooth

    from nir_core.preprocess.providers.chemotools import (
        arpls,
        median_filter,
        rubberband,
        whittaker_smooth,
    )

    X, _ = spectra
    np.testing.assert_allclose(
        arpls(X, lambda_=1e4, ratio=0.02, max_iters=30),
        ArPls(
            lam=1e4,
            ratio=0.02,
            nr_iterations=30,
            solver_type="banded",
            max_iter_after_warmstart=20,
            n_jobs=1,
        ).fit_transform(X),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        rubberband(X),
        RubberbandCorrection(n_jobs=1).fit_transform(X),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        median_filter(X, window_length=5, mode="reflect"),
        MedianFilter(window_length=5, mode="reflect", n_jobs=1).fit_transform(X),
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        whittaker_smooth(X, lambda_=1e4),
        WhittakerSmooth(
            lam=1e4,
            weights=None,
            solver_type="banded",
            n_jobs=1,
        ).fit_transform(X),
        rtol=0.0,
        atol=1e-12,
    )


def test_chemotools_normalize_rejects_native_only_max_norm():
    from nir_core.preprocess.providers.chemotools import normalize

    with pytest.raises(ValueError, match="l1.*l2"):
        normalize(np.ones((2, 5)), norm="max")


def test_chemotools_adapters_do_not_mutate_inputs(spectra):
    from nir_core.preprocess.providers.chemotools import (
        airpls,
        arpls,
        asls,
        autoscale,
        detrend,
        emsc,
        mean_center,
        median_filter,
        msc,
        normalize,
        norris_williams_derivative,
        rnv,
        rubberband,
        sg_smooth,
        snv,
        whittaker_smooth,
    )

    X, _ = spectra
    original = X.copy()
    for transform in (
        snv,
        msc,
        sg_smooth,
        asls,
        mean_center,
        autoscale,
        normalize,
        emsc,
        airpls,
        norris_williams_derivative,
        detrend,
        rnv,
        arpls,
        rubberband,
        median_filter,
        whittaker_smooth,
    ):
        transform(X)
        np.testing.assert_array_equal(X, original)
