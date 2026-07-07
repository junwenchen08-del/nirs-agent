"""Tests for :func:`nir_core.preprocess.scatter.snv`."""

from __future__ import annotations

import numpy as np
import pytest

from nir_core.preprocess.scatter import snv


def test_snv_matches_manual_computation(rng):
    """SNV should equal (X - row_mean) / row_std computed by numpy."""
    X = rng.normal(0.5, 0.2, size=(20, 100)) + np.linspace(0, 1, 100)
    out = snv(X)

    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True, ddof=0)
    expected = (X - mean) / std

    np.testing.assert_allclose(out, expected, rtol=1e-10)


def test_snv_row_mean_zero_std_one(rng):
    """After SNV each row mean ~0 and std ~1."""
    X = rng.normal(1.0, 0.3, size=(50, 200)) + np.sin(np.linspace(0, 6, 200))
    out = snv(X)
    np.testing.assert_allclose(out.mean(axis=1), 0.0, atol=1e-10)
    np.testing.assert_allclose(out.std(axis=1, ddof=0), 1.0, atol=1e-10)


def test_snv_constant_row_does_not_error():
    """A row with zero std should become zeros, not inf/nan."""
    X = np.ones((3, 10))
    X[0] = 5.0  # constant row
    X[1] = np.linspace(0, 1, 10)  # varying row
    X[2] = 7.0  # another constant row
    out = snv(X)
    assert not np.any(np.isnan(out))
    assert not np.any(np.isinf(out))
    # constant rows -> all zeros
    np.testing.assert_allclose(out[0], 0.0)
    np.testing.assert_allclose(out[2], 0.0)


def test_snv_preserves_shape(rng):
    X = rng.normal(size=(10, 50))
    assert snv(X).shape == X.shape


def test_snv_does_not_mutate_input(rng):
    X = rng.normal(size=(5, 30))
    X_copy = X.copy()
    _ = snv(X)
    np.testing.assert_array_equal(X, X_copy)


def test_snv_accepts_1d_input(rng):
    x = rng.normal(2.0, 0.5, size=80)
    out_1d = snv(x)
    out_2d = snv(x[np.newaxis, :])[0]
    assert out_1d.shape == x.shape
    np.testing.assert_allclose(out_1d, out_2d, rtol=1e-12)
