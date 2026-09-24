"""Tests for the Norris-Williams gap-segment derivative."""

from __future__ import annotations

import numpy as np
import pytest

from nir_core.preprocess.smoothing import norris_williams_derivative


def test_first_derivative_of_linear_signal_is_one_interior():
    delta = 2.0
    x = np.arange(101, dtype=float) * delta
    out = norris_williams_derivative(x, gap=3, segment=5, deriv=1, delta=delta)
    np.testing.assert_allclose(out[6:-6], 1.0, atol=1e-12)


def test_second_derivative_of_quadratic_signal_is_two_interior():
    delta = 0.5
    axis = np.arange(121, dtype=float) * delta
    x = axis**2
    out = norris_williams_derivative(x, gap=4, segment=5, deriv=2, delta=delta)
    np.testing.assert_allclose(out[8:-8], 2.0, atol=1e-10)


def test_norris_williams_reduces_high_frequency_noise(rng):
    axis = np.linspace(0.0, 10.0, 301)
    clean = np.sin(axis)
    noisy = clean + rng.normal(0.0, 0.08, axis.size)
    raw_derivative = norris_williams_derivative(
        noisy, gap=2, segment=1, deriv=1, delta=axis[1] - axis[0]
    )
    smoothed_derivative = norris_williams_derivative(
        noisy, gap=2, segment=9, deriv=1, delta=axis[1] - axis[0]
    )
    target = np.cos(axis)
    raw_error = np.mean((raw_derivative[10:-10] - target[10:-10]) ** 2)
    smoothed_error = np.mean((smoothed_derivative[10:-10] - target[10:-10]) ** 2)
    assert smoothed_error < raw_error


def test_norris_williams_preserves_shape_and_input(rng):
    X = rng.normal(size=(3, 101))
    original = X.copy()
    out = norris_williams_derivative(X, gap=3, segment=5, deriv=1)
    assert out.shape == X.shape
    np.testing.assert_array_equal(X, original)


@pytest.mark.parametrize(
    "params",
    [
        {"gap": 0},
        {"gap": 2.5},
        {"segment": 0},
        {"segment": 4},
        {"deriv": 0},
        {"deriv": 3},
        {"delta": 0.0},
    ],
)
def test_norris_williams_rejects_invalid_parameters(params):
    with pytest.raises(ValueError):
        norris_williams_derivative(np.arange(30.0), **params)


def test_norris_williams_rejects_signal_too_short_for_gap():
    with pytest.raises(ValueError, match="too short"):
        norris_williams_derivative(np.arange(7.0), gap=4)
