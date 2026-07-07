"""Tests for :func:`nir_core.preprocess.smoothing.sg_derivative`."""

from __future__ import annotations

import numpy as np
import pytest

from nir_core.preprocess.smoothing import sg_derivative


def test_first_derivative_monotonic_data_positive():
    """First derivative of a strictly increasing signal should be >= 0."""
    x = np.linspace(0, 10, 200)
    signal = np.exp(x / 5)  # strictly increasing
    X = signal[np.newaxis, :]
    d1 = sg_derivative(X, window=11, order=2, deriv=1)
    # In the interior (avoid edge effects) derivative should be positive.
    interior = d1[0, 20:-20]
    assert np.all(interior > 0)


def test_first_derivative_linear_is_constant():
    """First derivative of a linear ramp should be ~constant in the interior.

    Note: scipy's savgol_filter deriv returns the derivative with respect to
    the sample index, so for signal slope m sampled with spacing dx the
    returned derivative equals m * dx.
    """
    x = np.linspace(0, 10, 301)
    signal = 3.0 * x + 1.0
    d1 = sg_derivative(signal, window=11, order=3, deriv=1)
    interior = d1[20:-20]
    dx = x[1] - x[0]
    np.testing.assert_allclose(interior, 3.0 * dx, atol=1e-6)


def test_second_derivative_sign_for_peak():
    """Second derivative of a Gaussian peak should be negative at the peak."""
    x = np.linspace(-5, 5, 301)
    peak = np.exp(-x ** 2)
    d2 = sg_derivative(peak, window=15, order=4, deriv=2)
    # At the center (x=0) a Gaussian has negative curvature.
    center_idx = np.argmin(np.abs(x))
    assert d2[center_idx] < 0


def test_second_derivative_sign_for_valley():
    """Second derivative of an inverted Gaussian (valley) should be positive at center."""
    x = np.linspace(-5, 5, 301)
    valley = -np.exp(-x ** 2)
    d2 = sg_derivative(valley, window=15, order=4, deriv=2)
    center_idx = np.argmin(np.abs(x))
    assert d2[center_idx] > 0


def test_derivative_deriv_greater_than_order_raises(rng):
    X = rng.normal(size=(3, 100))
    with pytest.raises(ValueError, match="deriv must be"):
        sg_derivative(X, window=11, order=2, deriv=3)


def test_derivative_invalid_window_raises(rng):
    X = rng.normal(size=(3, 100))
    with pytest.raises(ValueError, match="odd"):
        sg_derivative(X, window=10, order=2, deriv=1)


def test_derivative_preserves_shape(rng):
    X = rng.normal(size=(5, 120))
    assert sg_derivative(X, window=11, order=3, deriv=2).shape == X.shape


def test_derivative_does_not_mutate_input(rng):
    X = rng.normal(size=(4, 90))
    X_copy = X.copy()
    _ = sg_derivative(X, window=11, order=2, deriv=1)
    np.testing.assert_array_equal(X, X_copy)
