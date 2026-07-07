"""Tests for baseline-removal algorithms (airPLS, asLS, detrend)."""

from __future__ import annotations

import numpy as np
import pytest

from nir_core.preprocess.baseline import airpls, asls, detrend


def _make_signal_with_baseline(n=300, seed=0):
    """Build a 1D signal = Gaussian peaks + smooth baseline drift."""
    x = np.linspace(0, 10, n)
    rng = np.random.default_rng(seed)
    peaks = (
        1.0 * np.exp(-((x - 3) ** 2) / 0.2)
        + 0.8 * np.exp(-((x - 6) ** 2) / 0.3)
        + 0.5 * np.exp(-((x - 8) ** 2) / 0.15)
    )
    baseline = 0.5 + 0.3 * np.sin(x / 2) + 0.05 * x  # smooth drift
    signal = peaks + baseline + rng.normal(0, 0.005, n)
    return x, signal, peaks, baseline


class TestAirpls:
    def test_airpls_removes_baseline_drift(self):
        """After airPLS the residual baseline should be near-constant."""
        x, signal, peaks, baseline = _make_signal_with_baseline()
        corrected = airpls(signal[np.newaxis, :], lambda_=1e6, max_iters=30)[0]
        # The corrected signal's lower envelope should be near 0 (baseline removed).
        # Take the 10th percentile as a proxy for the lower envelope.
        low = np.percentile(corrected, 5)
        # Without baseline correction, the lower envelope is ~baseline min (~0.5).
        # After correction it should be much closer to 0.
        assert abs(low) < 0.2

    def test_airpls_preserves_peak_locations(self):
        """airPLS should not destroy the signal peaks."""
        x, signal, peaks, baseline = _make_signal_with_baseline()
        corrected = airpls(signal[np.newaxis, :], lambda_=1e6, max_iters=30)[0]
        # Peaks of the corrected signal should align with peaks of the input peaks.
        # Use a wider tolerance window because baseline correction can shift the
        # exact local maximum by a sample or two.
        peak_centers = [3.0, 6.0]
        for c in peak_centers:
            idx = np.argmin(np.abs(x - c))
            window = corrected[max(0, idx - 10): idx + 11]
            # The peak value should be close to the window maximum.
            assert corrected[idx] >= window.max() - 0.05

    def test_airpls_2d_input(self, rng):
        """airPLS should process a 2D batch spectrum-by-spectrum."""
        X = rng.normal(0, 0.01, size=(5, 200)) + np.linspace(0, 1, 200)
        out = airpls(X, lambda_=1e5, max_iters=20)
        assert out.shape == X.shape
        assert not np.any(np.isnan(out))

    def test_airpls_does_not_mutate_input(self, rng):
        X = rng.normal(size=(3, 100)) + np.linspace(0, 0.5, 100)
        X_copy = X.copy()
        _ = airpls(X, lambda_=1e5, max_iters=10)
        np.testing.assert_array_equal(X, X_copy)

    def test_airpls_accepts_1d(self):
        x, signal, _, _ = _make_signal_with_baseline()
        out_1d = airpls(signal, lambda_=1e6, max_iters=20)
        out_2d = airpls(signal[np.newaxis, :], lambda_=1e6, max_iters=20)[0]
        np.testing.assert_allclose(out_1d, out_2d, rtol=1e-10)


class TestAsls:
    def test_asls_removes_baseline(self):
        x, signal, peaks, baseline = _make_signal_with_baseline()
        corrected = asls(signal[np.newaxis, :], lambda_=1e5, p=0.01)[0]
        low = np.percentile(corrected, 5)
        assert abs(low) < 0.2

    def test_asls_preserves_peaks(self):
        x, signal, peaks, baseline = _make_signal_with_baseline()
        corrected = asls(signal[np.newaxis, :], lambda_=1e5, p=0.01)[0]
        peak_idx = np.argmax(np.exp(-((x - 3) ** 2) / 0.2))
        window = corrected[max(0, peak_idx - 10): peak_idx + 11]
        assert corrected[peak_idx] >= window.max() - 0.05

    def test_asls_2d_input(self, rng):
        X = rng.normal(0, 0.01, size=(4, 150)) + np.linspace(0, 1, 150)
        out = asls(X, lambda_=1e4, p=0.01)
        assert out.shape == X.shape
        assert not np.any(np.isnan(out))

    def test_asls_does_not_mutate_input(self, rng):
        X = rng.normal(size=(3, 100))
        X_copy = X.copy()
        _ = asls(X)
        np.testing.assert_array_equal(X, X_copy)


class TestDetrend:
    def test_detrend_removes_quadratic_trend(self):
        """A pure quadratic trend should be (nearly) fully removed."""
        x = np.linspace(0, 10, 200)
        trend = 2.0 + 0.5 * x + 0.1 * x ** 2
        signal = trend[np.newaxis, :]
        out = detrend(signal, wv=x)
        # After detrending the residual should be near zero everywhere.
        np.testing.assert_allclose(out, 0.0, atol=1e-8)

    def test_detrend_linear_trend_removed(self):
        x = np.linspace(0, 10, 100)
        trend = 1.0 + 3.0 * x
        out = detrend(trend, wv=x)
        np.testing.assert_allclose(out, 0.0, atol=1e-8)

    def test_detrend_without_wv_uses_column_index(self):
        """When wv is None the column index is used as the regressor."""
        n = 100
        idx = np.arange(n, dtype=float)
        trend = 1.0 + 0.5 * idx + 0.01 * idx ** 2
        out = detrend(trend)
        np.testing.assert_allclose(out, 0.0, atol=1e-8)

    def test_detrend_preserves_peak_signal(self):
        """Detrending should preserve a peak on top of a quadratic baseline."""
        x = np.linspace(0, 10, 300)
        peak = np.exp(-((x - 5) ** 2) / 0.5)
        trend = 1.0 + 0.3 * x + 0.02 * x ** 2
        signal = peak + trend
        out = detrend(signal, wv=x)
        # The peak maximum should still be present near x=5.
        peak_idx = np.argmin(np.abs(x - 5))
        assert out[peak_idx] > np.percentile(out, 50)

    def test_detrend_2d_input(self, rng):
        X = rng.normal(0, 0.01, size=(5, 150))
        x = np.linspace(0, 10, 150)
        X = X + 0.5 * x + 0.02 * x ** 2
        out = detrend(X, wv=x)
        assert out.shape == X.shape

    def test_detrend_does_not_mutate_input(self, rng):
        X = rng.normal(size=(3, 100))
        X_copy = X.copy()
        _ = detrend(X)
        np.testing.assert_array_equal(X, X_copy)
