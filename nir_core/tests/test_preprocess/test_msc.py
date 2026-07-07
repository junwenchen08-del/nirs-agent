"""Tests for :func:`nir_core.preprocess.scatter.msc`."""

from __future__ import annotations

import numpy as np
import pytest

from nir_core.preprocess.scatter import msc


def test_msc_matches_manual_fit(rng):
    """MSC corrected spectrum should match a manual OLS fit x = a + b*ref."""
    X = rng.normal(1.0, 0.3, size=(15, 100))
    X += np.linspace(0, 0.5, 100)  # shared structure
    ref = X.mean(axis=0)
    out = msc(X)

    # Manual per-row fit.
    A = np.column_stack([np.ones(100), ref])
    expected = np.empty_like(X)
    for i in range(X.shape[0]):
        coef, *_ = np.linalg.lstsq(A, X[i], rcond=None)
        a, b = coef
        expected[i] = (X[i] - a) / b
    np.testing.assert_allclose(out, expected, rtol=1e-8, atol=1e-10)


def test_msc_default_reference_is_mean(rng):
    """When reference is None the mean spectrum should be used."""
    X = rng.normal(size=(10, 50))
    # Compute MSC with explicit mean reference and default; should match.
    ref = X.mean(axis=0)
    out_default = msc(X)
    out_explicit = msc(X, reference=ref)
    np.testing.assert_allclose(out_default, out_explicit, rtol=1e-12)


def test_msc_custom_reference(rng):
    """Passing a custom reference should change the output vs default."""
    X = rng.normal(1.0, 0.2, size=(12, 80))
    ref = X[0]  # use first sample as reference
    out_custom = msc(X, reference=ref)
    out_default = msc(X)
    # Outputs should differ (custom reference is not the mean).
    assert not np.allclose(out_custom, out_default)
    # Verify manual fit against the custom reference.
    A = np.column_stack([np.ones(80), ref])
    expected = np.empty_like(X)
    for i in range(X.shape[0]):
        coef, *_ = np.linalg.lstsq(A, X[i], rcond=None)
        a, b = coef
        expected[i] = (X[i] - a) / b
    np.testing.assert_allclose(out_custom, expected, rtol=1e-8, atol=1e-10)


def test_msc_preserves_shape(rng):
    X = rng.normal(size=(8, 60))
    assert msc(X).shape == X.shape


def test_msc_does_not_mutate_input(rng):
    X = rng.normal(size=(5, 40))
    X_copy = X.copy()
    _ = msc(X)
    np.testing.assert_array_equal(X, X_copy)


def test_msc_zero_slope_row_returns_reference():
    """A row identical to the reference (b=1, a=0) should be unchanged up to ref."""
    ref = np.linspace(0, 1, 50)
    X = np.tile(ref, (3, 1))  # every row == ref => a=0, b=1, output == ref
    out = msc(X, reference=ref)
    np.testing.assert_allclose(out, np.tile(ref, (3, 1)), atol=1e-10)


def test_msc_accepts_1d_input(rng):
    x = rng.normal(1.0, 0.2, size=60)
    out_1d = msc(x)
    out_2d = msc(x[np.newaxis, :])[0]
    assert out_1d.shape == x.shape
    np.testing.assert_allclose(out_1d, out_2d, rtol=1e-10)
