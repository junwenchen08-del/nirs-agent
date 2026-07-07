"""Tests for :func:`nir_core.preprocess.smoothing.sg_smooth`."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import savgol_filter

from nir_core.preprocess.smoothing import sg_smooth


def test_sg_smooth_matches_scipy(rng):
    """sg_smooth output should match scipy.signal.savgol_filter directly."""
    X = rng.normal(size=(10, 200))
    window, order = 11, 2
    out = sg_smooth(X, window=window, order=order)
    expected = savgol_filter(X, window_length=window, polyorder=order, deriv=0, axis=1)
    np.testing.assert_allclose(out, expected, rtol=1e-12)


def test_sg_smooth_reduces_noise():
    """Smoothing should reduce the variance of pure noise."""
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.1, size=(5, 500))
    smoothed = sg_smooth(noise, window=21, order=3)
    # Smoothed signal has lower std than the raw noise.
    assert smoothed.std() < noise.std()


def test_sg_smooth_preserves_shape(rng):
    X = rng.normal(size=(7, 150))
    assert sg_smooth(X).shape == X.shape


def test_sg_smooth_does_not_mutate_input(rng):
    X = rng.normal(size=(4, 80))
    X_copy = X.copy()
    _ = sg_smooth(X, window=11, order=2)
    np.testing.assert_array_equal(X, X_copy)


def test_sg_smooth_even_window_raises(rng):
    X = rng.normal(size=(3, 50))
    with pytest.raises(ValueError, match="odd"):
        sg_smooth(X, window=10, order=2)


def test_sg_smooth_window_le_order_raises(rng):
    """window must be strictly greater than order."""
    X = rng.normal(size=(3, 50))
    # window == order is invalid (need order < window).
    with pytest.raises(ValueError):
        sg_smooth(X, window=5, order=5)
    # window < order is invalid.
    with pytest.raises(ValueError):
        sg_smooth(X, window=5, order=7)


def test_sg_smooth_accepts_1d(rng):
    x = rng.normal(size=100)
    out = sg_smooth(x, window=11, order=2)
    assert out.shape == x.shape
    expected = savgol_filter(x, 11, 2, deriv=0)
    np.testing.assert_allclose(out, expected, rtol=1e-12)
