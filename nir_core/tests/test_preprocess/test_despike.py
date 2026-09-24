"""Tests for isolated-spike removal."""

from __future__ import annotations

import numpy as np
import pytest

from nir_core.preprocess.despike import despike


def test_despike_repairs_positive_and_negative_isolated_spikes():
    x = np.linspace(0.0, 4.0 * np.pi, 201)
    clean = np.sin(x)
    contaminated = clean.copy()
    contaminated[60] += 12.0
    contaminated[140] -= 10.0

    corrected = despike(contaminated, window=5, z_threshold=6.0)

    assert abs(corrected[60] - clean[60]) < 0.1
    assert abs(corrected[140] - clean[140]) < 0.1
    keep = np.ones(clean.size, dtype=bool)
    keep[[60, 140]] = False
    np.testing.assert_allclose(corrected[keep], clean[keep], atol=1e-12)


def test_despike_leaves_smooth_spectrum_unchanged():
    x = np.linspace(-2.0, 2.0, 301)
    smooth = np.exp(-(x**2) / 0.2)
    corrected = despike(smooth, window=5, z_threshold=8.0)
    np.testing.assert_allclose(corrected, smooth, atol=1e-12)


def test_despike_preserves_broad_absorption_peak():
    x = np.linspace(-3.0, 3.0, 301)
    broad_peak = np.exp(-(x**2) / 0.4)
    corrected = despike(broad_peak, window=5, z_threshold=6.0)
    assert corrected.max() > 0.99 * broad_peak.max()


def test_despike_supports_batches_and_does_not_mutate(rng):
    X = rng.normal(0.0, 0.02, size=(4, 101))
    X[:, 50] += 5.0
    original = X.copy()
    out = despike(X)
    assert out.shape == X.shape
    assert np.isfinite(out).all()
    np.testing.assert_array_equal(X, original)


@pytest.mark.parametrize(
    ("window", "z_threshold"),
    [(2, 6.0), (4, 6.0), (5, 0.0), (5, -1.0)],
)
def test_despike_rejects_invalid_parameters(window, z_threshold):
    with pytest.raises(ValueError):
        despike(np.arange(20.0), window=window, z_threshold=z_threshold)
