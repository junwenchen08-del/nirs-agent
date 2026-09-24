"""Tests for robust standard-normal-variate preprocessing."""

from __future__ import annotations

import numpy as np

from nir_core.preprocess.scatter import robust_snv


def test_robust_snv_has_zero_row_median_and_unit_robust_scale(rng):
    X = rng.normal(2.0, 0.4, size=(12, 101))
    X[:, 50] += 20.0

    out = robust_snv(X)

    np.testing.assert_allclose(np.median(out, axis=1), 0.0, atol=1e-12)
    robust_scale = 1.4826 * np.median(
        np.abs(out - np.median(out, axis=1, keepdims=True)), axis=1
    )
    np.testing.assert_allclose(robust_scale, 1.0, rtol=1e-10, atol=1e-10)


def test_robust_snv_is_less_distorted_by_single_outlier_than_snv(rng):
    from nir_core.preprocess.scatter import snv

    x = np.linspace(-1.0, 1.0, 101) + rng.normal(0.0, 0.01, 101)
    contaminated = x.copy()
    contaminated[50] = 100.0

    robust_clean = robust_snv(x)
    robust_contaminated = robust_snv(contaminated)
    classical_clean = snv(x)
    classical_contaminated = snv(contaminated)
    keep = np.ones(x.size, dtype=bool)
    keep[50] = False

    robust_error = np.mean(np.abs(robust_clean[keep] - robust_contaminated[keep]))
    classical_error = np.mean(
        np.abs(classical_clean[keep] - classical_contaminated[keep])
    )
    assert robust_error < classical_error


def test_robust_snv_constant_rows_become_zero():
    X = np.vstack([np.ones(20), np.full(20, 7.0)])
    out = robust_snv(X)
    np.testing.assert_allclose(out, 0.0)
    assert np.isfinite(out).all()


def test_robust_snv_accepts_1d_and_does_not_mutate(rng):
    x = rng.normal(size=51)
    original = x.copy()
    out = robust_snv(x)
    assert out.shape == x.shape
    np.testing.assert_array_equal(x, original)
