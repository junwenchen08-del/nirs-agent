"""Tests for :mod:`nir_core.diagnostics`."""

from __future__ import annotations

import numpy as np
import pytest

from nir_core.diagnostics import compute_residual_diagnostics


class TestComputeResidualDiagnostics:
    """Tests for compute_residual_diagnostics with relative thresholds."""

    def test_flat_trend_low_variance(self):
        """Good predictions → flat trend, low variance, few outliers."""
        rng = np.random.default_rng(42)
        y_true = np.linspace(1, 10, 50)
        y_pred = y_true + rng.normal(0, 0.05, size=50)  # tiny noise
        diag = compute_residual_diagnostics(y_true, y_pred)
        assert diag["residual_trend"] == "flat"
        assert diag["residual_variance"] == "low"
        assert diag["outlier_ratio"] < 0.15  # at most ~7 of 50

    def test_upward_trend(self):
        """Systematic under-prediction at high y → upward trend."""
        y_true = np.linspace(1, 10, 50)
        y_pred = y_true - 0.5 * (y_true - 5)  # increasingly wrong at high y
        diag = compute_residual_diagnostics(y_true, y_pred)
        assert diag["residual_trend"] == "upward"

    def test_downward_trend(self):
        """Systematic over-prediction at high y → downward trend."""
        y_true = np.linspace(1, 10, 50)
        y_pred = y_true + 0.5 * (y_true - 5)
        diag = compute_residual_diagnostics(y_true, y_pred)
        assert diag["residual_trend"] == "downward"

    def test_high_variance(self):
        """Large residuals relative to y → high variance."""
        rng = np.random.default_rng(123)
        y_true = np.linspace(1, 10, 50)
        y_pred = y_true + rng.normal(0, 3.0, size=50)  # huge noise
        diag = compute_residual_diagnostics(y_true, y_pred)
        assert diag["residual_variance"] == "high"

    def test_outlier_detection(self):
        """A few extreme residuals → outlier_count > 0."""
        y_true = np.linspace(1, 10, 50)
        y_pred = y_true.copy()
        y_pred[5] += 100  # huge outlier
        y_pred[25] -= 100  # huge outlier
        diag = compute_residual_diagnostics(y_true, y_pred)
        assert diag["outlier_count"] >= 2
        assert diag["outlier_ratio"] > 0.0

    def test_scale_invariance(self):
        """Diagnostics should be identical for y scaled by a constant."""
        y_true = np.linspace(1, 10, 50)
        y_pred = y_true + 0.3 * (y_true - 5)  # upward trend
        diag1 = compute_residual_diagnostics(y_true, y_pred)
        # Scale both by 100 (e.g. moisture 0-100 vs 0-1)
        diag2 = compute_residual_diagnostics(y_true * 100, y_pred * 100)
        assert diag1["residual_trend"] == diag2["residual_trend"]
        assert diag1["residual_variance"] == diag2["residual_variance"]

    def test_single_sample(self):
        """A single sample returns flat/low defaults."""
        diag = compute_residual_diagnostics(np.array([5.0]), np.array([4.9]))
        assert diag["residual_trend"] == "flat"
        assert diag["residual_variance"] == "low"
        assert diag["outlier_count"] == 0

    def test_zero_variance_y(self):
        """All y_true identical → graceful handling."""
        y_true = np.full(10, 5.0)
        y_pred = np.full(10, 5.0)
        diag = compute_residual_diagnostics(y_true, y_pred)
        assert diag["residual_trend"] == "flat"
        assert diag["outlier_count"] == 0

    def test_shape_mismatch_raises(self):
        """Mismatched shapes raise ValueError."""
        with pytest.raises(ValueError, match="Shape mismatch"):
            compute_residual_diagnostics(np.array([1, 2, 3]), np.array([1, 2]))

    def test_return_keys(self):
        """Return dict has all expected keys."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.1, 3.9, 4.8])
        diag = compute_residual_diagnostics(y_true, y_pred)
        expected_keys = {
            "residual_trend",
            "residual_variance",
            "outlier_ratio",
            "outlier_count",
            "residual_std",
        }
        assert set(diag.keys()) == expected_keys
