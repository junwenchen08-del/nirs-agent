"""Tests for nir_core.plotting.model_diag."""

from __future__ import annotations

import base64

import numpy as np
import pytest

from nir_core.plotting.model_diag import (
    plot_drift_heatmap,
    plot_predicted_vs_reference,
    plot_residuals,
)


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _assert_valid_png_b64(b64_str: str) -> None:
    """Decode a base64 string and assert the bytes start with the PNG magic."""
    assert isinstance(b64_str, str)
    assert len(b64_str) > 0
    raw = base64.b64decode(b64_str)
    assert raw[:8] == PNG_MAGIC, "Decoded bytes are not a PNG."


def test_plot_predicted_vs_reference_returns_png(rng: np.random.Generator) -> None:
    """plot_predicted_vs_reference returns a valid base64 PNG with linear+noise data."""
    x = np.linspace(0, 10, 50)
    y_ref = 2 * x + 1
    y_pred = y_ref + rng.normal(0.0, 0.2, size=y_ref.size)
    b64 = plot_predicted_vs_reference(y_ref, y_pred, title="测试 P vs R")
    _assert_valid_png_b64(b64)


def test_plot_predicted_vs_reference_length_mismatch_raises() -> None:
    """Length mismatch between y_ref and y_pred should raise ValueError."""
    y_ref = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0])
    with pytest.raises(ValueError):
        plot_predicted_vs_reference(y_ref, y_pred)


def test_plot_residuals_returns_png(rng: np.random.Generator) -> None:
    """plot_residuals returns a valid base64 PNG."""
    x = np.linspace(0, 10, 50)
    y_ref = 2 * x + 1
    y_pred = y_ref + rng.normal(0.0, 0.3, size=y_ref.size)
    b64 = plot_residuals(y_ref, y_pred)
    _assert_valid_png_b64(b64)


def test_plot_residuals_length_mismatch_raises() -> None:
    """Length mismatch should raise ValueError."""
    y_ref = np.ones(5)
    y_pred = np.ones(3)
    with pytest.raises(ValueError):
        plot_residuals(y_ref, y_pred)


def test_plot_drift_heatmap_returns_png() -> None:
    """plot_drift_heatmap returns a valid base64 PNG for constructed inputs."""
    rng = np.random.default_rng(42)
    n_samples = 50
    distances = rng.chisquare(df=5, size=n_samples)
    # Inject a few large-distance outliers.
    distances[5] = 40.0
    distances[20] = 45.0
    flagged = np.array([5, 20], dtype=int)
    wv = np.linspace(1100.0, 2500.0, 100)
    b64 = plot_drift_heatmap(distances, wv, flagged)
    _assert_valid_png_b64(b64)


def test_plot_drift_heatmap_no_flagged() -> None:
    """Empty flagged array should still produce a valid PNG."""
    rng = np.random.default_rng(0)
    distances = rng.standard_normal(30) ** 2
    wv = np.linspace(1000.0, 2000.0, 50)
    b64 = plot_drift_heatmap(distances, wv, np.array([], dtype=int))
    _assert_valid_png_b64(b64)


def test_plot_predicted_vs_reference_linear_data(
    linear_data: tuple[np.ndarray, np.ndarray],
    rng: np.random.Generator,
) -> None:
    """Use the shared linear_data fixture (x, y) with added noise."""
    x, y = linear_data
    y_pred = y + rng.normal(0.0, 0.1, size=y.size)
    b64 = plot_predicted_vs_reference(y, y_pred)
    _assert_valid_png_b64(b64)
